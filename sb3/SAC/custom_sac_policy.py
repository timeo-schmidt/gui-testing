from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
import gymnasium as gym
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs

from stable_baselines3.sac.policies import Actor, SACPolicy

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from PIL import Image, ImageDraw, ImageFilter
import numpy as np


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Custom Action Distribution
class MaskedSquashedDiagGaussianDistribution(SquashedDiagGaussianDistribution):
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim, epsilon)

    # def sample(self) -> th.Tensor:
    #     # Reparametrization trick to pass gradients
    #     self.gaussian_actions = super().sample()

    #     # For each action, draw 1000 samples from the normal distribution
        
    #     screen_size = th.tensor(self.mask.shape[2:]).to(self.gaussian_actions.device)
    #     zero_tensor = th.tensor([0,0]).to(self.gaussian_actions.device).type(th.long)

    #     # If the mask is active, ensure that the sampling is according to the mask
    #     if self.mask is not None and self.mask_valid:
    #         for i, a in enumerate(self.gaussian_actions):
    #             # Check if the action is valid
    #             # Convert the action to the screen click coordinates
    #             scaled_action = (((a + 1) / 2) * th.tensor(screen_size)).type(th.long).clamp(th.tensor(zero_tensor), screen_size - 1)
    #             z=0
    #             for _ in range(10000):
    #                 z+=1
    #                 if self.mask[i, -1, scaled_action[0], scaled_action[1]].item() != 255:
    #                     # If the action is invalid, sample again
    #                     self.gaussian_actions[i] = super().sample()[i]
    #                     scaled_action = (((self.gaussian_actions[i] + 1) / 2) * th.tensor(screen_size)).type(th.long).clamp(th.tensor(zero_tensor), screen_size - 1)
    #                 else:
    #                     break
    #             if z == 10000:
    #                 print("ERROR: Could not find a valid action")

    #     return th.tanh(self.gaussian_actions)
        
    def masked_actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, mask: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.mask = mask
        # Check that the mask does have at least 10% of clickable values
        self.mask_valid = True
        # Sample actions from the normal distribution
        afp =  super().actions_from_params(mean_actions, log_std, deterministic=deterministic)
        self.mask_valid = False
        return afp
    
# Custom Feature Extractor
class CustomCNNExtractor(NatureCNN):

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        
        # # Set the obstervation space to only the screenshot
        # observation_space = observation_space["screenshot"]

        # Call the parent constructor
        super().__init__(
            observation_space,
            features_dim,
            normalized_image=normalized_image
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations["screenshot"]
        return super().forward(observations)


class MaskedActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )

        # Overriding the action distribution
        action_dim = get_action_dim(self.action_space)
        self.action_dist = MaskedSquashedDiagGaussianDistribution(action_dim)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        x =  self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

        # Get the shape of the mask
        mask = obs["clickable_elements"][:, -1, :, :] # Keep the most recent frame only for action masking

        # Create a copy of x
        masked_actions = x.clone()

        # Iterate through the batch
        for i in range(x.shape[0]):
            # Get all the valid actions
            indices = th.nonzero(mask[i] == 255)

            # If there are no valid actions, just use the original action
            if(indices.shape[0] == 0):
                print("there are no valid actions!")
                continue

            indices = indices.float()
            # Normalize the 2nd element (height)
            indices[:,0] = (indices[:,0] / (mask[i].shape[0] - 1)) * 2 - 1
            # Normalize the 3rd element (width)
            indices[:,1] = (indices[:,1] / (mask[i].shape[1] - 1)) * 2 - 1

            obs_dist = SquashedDiagGaussianDistribution(action_dim=2)
            obs_dist.proba_distribution(mean_actions[i], log_std[i])

            # compute log probabilities of the positions
            log_probs = obs_dist.log_prob(indices)

            # to get actual probabilities, you can exponentiate the log probabilities
            probs = th.exp(log_probs)

            action_index = th.multinomial(probs, num_samples=1)

            # Set the masked_actions
            masked_actions[i] = indices[action_index]

            # Convert arrays to Pillow Images
            screenshot_img = Image.fromarray(np.transpose(obs["screenshot"][i].detach().cpu().numpy(), (1, 2, 0)), 'RGB')
            mask_img = Image.fromarray(mask[i].squeeze().cpu().detach().numpy(), 'L')

            # Resize the screenshot to the mask size
            screenshot_img = screenshot_img.resize(mask_img.size)

            # Now apply the mask to the screenshot
            screenshot_img.putalpha(mask_img)

            # Now draw a red circle at the click location
            draw = ImageDraw.Draw(screenshot_img)

            # masked actions e.g. tensor([[-0.9213,  0.0236]], device='mps:0')
            # mask.shape = 
            # Scale the click location to the mask size
            x_scaled = int((masked_actions[i][0]+1) * 0.5 * mask[i].shape[0])
            y_scaled = int((masked_actions[i][1]+1) * 0.5 * mask[i].shape[1])

            # Check that the click location is within the screenshot image
            if(x_scaled < 0 or x_scaled >= screenshot_img.size[1] or y_scaled < 0 or y_scaled >= screenshot_img.size[0]):
                print("click location is out of bounds!")
                continue

            draw.ellipse((y_scaled-1, x_scaled-1, y_scaled+1, x_scaled+1), fill='red', outline='red')

            # Save the image with the env id as name
            screenshot_img.save(f"debug_{i}.png")

        return masked_actions



class MaskedSACPolicy(SACPolicy):

    actor: MaskedActor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomCNNExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        ss_obs_space = self.observation_space["screenshot"]
        return self.features_extractor_class(ss_obs_space, **self.features_extractor_kwargs)
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MaskedActor(**actor_kwargs).to(self.device)