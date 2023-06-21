import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set the global font to be Times New Roman, size 10 (or any other size you want)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 13

groups = ['Cypress Real World App', 'Shopware 6']
methods = ['Random', 'Q-Learning', 'Unfamiliar Human Tester', 'Proposed Method', 'Expert Human Tester']
cypress_values = [50.75, 142.1, 170.0, 359, 363]
shopware_values = [15.6, 149.9, 169.0, 197, 291]

x = np.arange(len(methods))  # the label locations

width = 0.8  # the width of the bars

fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1x2 layout

G_A = "#95deb4"
G_B = "#51b079"

# Define color palettes
cypress_colors = [G_A, G_A, G_A, G_B, G_A]
shopware_colors = ['lightblue', 'lightblue', 'lightblue', 'steelblue', 'lightblue']

# Define alphas
cypress_alphas = [1, 1, 1, 1, 1]
shopware_alphas = [1, 1, 1, 1, 1]

zorders = [3, 3, 3, 3, 3]

# bar plot for Cypress
for i in range(len(x)):
    axs[0].bar(x[i], cypress_values[i], width, color=cypress_colors[i], alpha=cypress_alphas[i], zorder=zorders[i])
axs[0].set_ylabel('Discovered Elements')
axs[0].set_title(groups[0])
axs[0].set_xticks(x)
axs[0].set_xticklabels(methods, rotation=45, ha='right')  # Rotate labels 45 degrees
axs[0].set_axisbelow(True)
axs[0].yaxis.grid(color='gray', linestyle='dashed', alpha = 0.2)

# bar plot for Shopware
for i in range(len(x)):
    axs[1].bar(x[i], shopware_values[i], width, color=shopware_colors[i], alpha=shopware_alphas[i], zorder=zorders[i])
axs[1].set_ylabel('Discovered Elements')
axs[1].set_title(groups[1])
axs[1].set_xticks(x)
axs[1].set_xticklabels(methods, rotation=45, ha='right')  # Rotate labels 45 degrees
axs[1].set_axisbelow(True)
axs[1].yaxis.grid(color='gray', linestyle='dashed', alpha = 0.2)

# Make only the 'Proposed Method' label bold
for label in axs[0].xaxis.get_ticklabels():
    if label.get_text() == 'Proposed Method':
        label.set_weight('bold')

# Make only the 'Proposed Method' label bold
for label in axs[1].xaxis.get_ticklabels():
    if label.get_text() == 'Proposed Method':
        label.set_weight('bold')

# Draw a dashed horizonzal line where the proposed method is
axs[0].axhline(y=cypress_values[3], color='darkred', linestyle='dotted', linewidth=1)
axs[1].axhline(y=shopware_values[3], color='darkred', linestyle='dotted', linewidth=1)

fig.tight_layout()


current_dir = os.path.dirname(__file__)
print(current_dir)
plt.savefig(current_dir + "/baselines.pdf", bbox_inches="tight")
