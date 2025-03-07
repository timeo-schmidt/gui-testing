from tbparse import SummaryReader
import pandas as pd
import os

from matplotlib import rc
import matplotlib.pylab as plt

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

tb_paths = [
    "orch3_framestack_n_4_gray_Rv3_SAC_framestack_4_grayscale_unmasked_1",
    "orch3_framestack_n_4_SAC_framestack_4_grayscale_unmasked_1"
]

tb_labels = ["Reward v3 raw", "Reward v3 with log-scaling and negative default"]

# Create a data frame to which columns will be added
data = {}

for p in tb_paths:
    log_dir = "./tensorboard/" + p
    reader = SummaryReader(log_dir)
    df = reader.scalars
    df = df[df["tag"] == "rollout/ep_rew_mean"]
    # Sort the dataframe by step and limit to 100k steps
    df = df.sort_values(by="step")
    df = df[df["step"] <= 100000]
    # Get a new dataframe with step index and value column
    df = df[["step", "value"]]
    # reset the index
    df = df.reset_index(drop=True)
    # Add the data to the data dictionary
    data[p] = df



# Multiply the log scaled rewards by 0.1
data["orch3_framestack_n_4_SAC_framestack_4_grayscale_unmasked_1"]["value"] *= 0.1

# Create a 1x3 plot with the reward curves
# Set the figure size
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for i, (k, v) in enumerate(data.items()):
    ax[i].plot(v["step"], v["value"], label=k)

    # Set legend according to tb_labels
    ax[i].legend([tb_labels[i]])
    # Labels are x: steps, y: episode reward
    ax[i].set_xlabel("Steps", fontsize=14)
    ax[i].set_ylabel("Episode Reward", fontsize=14)
    # Set the x axis to 100k instead of 100000
    ax[i].set_xticks([0, 25000, 50000, 75000, 100000])
    ax[i].set_xticklabels(["0", "25k", "50k", "75k", "100k"])
    # Add padding to the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    ax[i].grid()

# Save the plot as a pdf in the directory this script is in __file__
current_dir = os.path.dirname(__file__)
print(current_dir)
plt.savefig(current_dir + "/ablation_rew_addon_new.pdf", bbox_inches="tight")

"""
Plot List:

Ablation Study:
-> ablation baseline
    - CNN feature extractor
        -> mlp policy
    - frame stacking
    - image preprocessing
        -> rgb 
    - reward function
        -> v1
        -> v2
        -> v3
        -> v3 + add ons

Generalisation Study
    -> Shopware from init
    -> RWA with shopware weight init

Baseline Experiments
    -> Random
    -> Q Table
    -> Human

"""