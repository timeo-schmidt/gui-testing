from tbparse import SummaryReader
import pandas as pd
import os

from matplotlib import rc
import matplotlib.pylab as plt

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

tb_paths = [
    "sac_graystack_5_500k_3",
    "rwa_init_shopware_SAC_framestack_4_grayscale_unmasked_1"
]

tb_labels = ["Random Initialisation", "Initialisation with Shopware Weights"]

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

# Create a plot with all the reward curves
# Set the figure size
fig, ax = plt.subplots(figsize=(10, 4))
for k, v in data.items():
    ax.plot(v["step"], v["value"], label=k)

# Set legend according to tb_labels
ax.legend(tb_labels)
# Labels are x: steps, y: episode reward
ax.set_xlabel("Steps", fontsize=14)
ax.set_ylabel("Episode Reward", fontsize=14)
# Set the x axis to 100k instead of 100000
ax.set_xticks([0, 25000, 50000, 75000, 100000])
ax.set_xticklabels(["0", "25k", "50k", "75k", "100k"])
# Add padding to the bottom of the plot
plt.subplots_adjust(bottom=0.15)
ax.grid()

# Save the plot as a pdf in the directory this script is in __file__
current_dir = os.path.dirname(__file__)
print(current_dir)
plt.savefig(current_dir + "/transfer_learning.pdf", bbox_inches="tight")

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