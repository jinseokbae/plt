"""
For plain-nav
    '''shell
        python plot_return_curve.py --xlim 5000 --window_size 30 --file_path ~/Downloads/wandb_export_2025-01-15T19_14_05.988+09_00.csv --columns 7 10 4 1 --labels  'PLT-1' 'PLT-2' 'PLT-5' 'PLT-5-noR' --title 'plain_nav' --ylim 40 160 --drop True --num_envs 4096
    '''

For damaged-nav
    '''shell
    '''
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import numpy as np

# Parameters
parser = argparse.ArgumentParser(description='Plot data with smoothing')
parser.add_argument('--xlim', type=int, default=1000, help='Limit of x-axis, # of epoch')
parser.add_argument('--ylim', type=float, nargs='+', help='Limit of y-axis')
parser.add_argument('--num_envs', type=int, default=1536, help='Number of environments')
parser.add_argument('--horizon_length', type=int, default=1536, help='Horizon length')
parser.add_argument('--window_size', type=int, default=10, help='Size of the rolling window')
parser.add_argument('--file_path', type=str, default='data.csv', help='Path to the data file')
parser.add_argument('--columns', type=int, nargs='+', default=[1, 4], help='Columns to plot')
parser.add_argument('--labels', type=str, nargs='+', default=['PULSE', 'PLT'], help='Labels for the columns')
parser.add_argument('--title', type=str, default='speed', help='Title of the plot')
parser.add_argument('--drop', type=bool, default=False, help='Drop values')
args = parser.parse_args()

x_multiplier = args.num_envs * args.horizon_length
x_lim = args.xlim * 2 + 1
y_lim = args.ylim
window_size = args.window_size
file_path = args.file_path
columns = args.columns
labels = args.labels
title = args.title

colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Optional: set a Seaborn style/theme
sns.set_theme(style='darkgrid', palette='deep')

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_facecolor('#d3d3d3')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(15260))  # Set major ticks every 2 units
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Set major ticks every 0.5 units

# Read the data
df = pd.read_csv(file_path)
if args.drop:
    df.dropna(inplace=True)
else:
    df.interpolate(method='slinear', inplace=True)

x = df.iloc[:x_lim, 0]
x_lim_idx = np.argmin(np.abs(x - x_lim))
x = x[:x_lim_idx + 1]
# g_palette = sns.color_palette("dark")

# Define the custom color palette
custom_colors = ["#e9445e", "#9ba548", "#1eb3a6", "#a491ed", "#794a00"]

# Set the palette
g_palette = sns.color_palette(custom_colors)
for i, col in enumerate(columns):
    # breakpoint()
    y = df.iloc[:x_lim, col] / 150
    y_smoothed = y.rolling(window_size).mean()

    sns.lineplot(x=x, y=y, color=g_palette[i], alpha=0.2, legend=False)
    # sns.lineplot(x=x, y=y_smoothed, label=labels[i], color=g_palette[i], alpha=1)

for i, col in enumerate(columns):
    # breakpoint()
    y = df.iloc[:x_lim, col] / 150
    y_smoothed = y.rolling(window_size).mean()

    # sns.lineplot(x=x, y=y, color=g_palette[i], alpha=0.3)
    sns.lineplot(x=x, y=y_smoothed, label=labels[i], color=g_palette[i], alpha=1, linewidth=2.5, legend=False)

# Customize the axes and legend
ax.set_xlabel(f'Env{args.num_envs}_Hori{args.horizon_length}', fontsize=26)
ax.set_ylabel('mb rewards', fontsize=26)
ax.set_title(title, fontsize=30)
# legend = ax.legend(fontsize=32, loc='lower center', bbox_to_anchor=(0.75, 0.1))
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_alpha(1)
ax.grid(True, color='white')

x_min, x_max = x.min(), x.max()
x_padding = (x_max - x_min) * 0.05
ax.set_xlim(x_min, x_max + x_padding)
if y_lim is not None:
    ax.set_ylim([y / 150 for y in y_lim])

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Adjust subplot and save
# fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
# fig.savefig('/home/ham/SIGGRAPH2025/plot/{}_{}.png'.format(title, current_time), format='png', dpi=300)

# Display the plot
plt.show()

