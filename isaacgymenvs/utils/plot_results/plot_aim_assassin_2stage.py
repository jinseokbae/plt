import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    "model": ["LaFAN1 Aiming + Assasin", "LaFAN1 Aiming + Assasin", "LaFAN1 Aiming + Assasin", "LaFAN1 Aiming", "LaFAN1 Aiming", "LaFAN1 Aiming"],
    "value": [0.881, 0.843, 0.852, 0.436, 0.646, 0.67],
    "std_dev": [0.09434666667, 0.09989333333, 0.09736666667, 0.31794, 0.3231466667, 0.31556],
    "group": ["Scratch", "Adapt-5-noR", "Adapt-5", "Scratch", "Adapt-5-noR", "Adapt-5"]
}
df = pd.DataFrame(data)

sns.set_theme(style='darkgrid', palette='deep')

# Draw a neste barplot by model and group
g_palette = sns.color_palette("dark", n_colors=4)
fixed_palette = [g_palette[2], g_palette[1], g_palette[0]]
g = sns.catplot(
    data=df, kind="bar",
    x="model", y="value", hue="group",
    ci=None,  # Disable seaborn's default error bars
    palette=fixed_palette, alpha=.6, height=6, aspect=1.5
)

# Access the axes object from the FacetGrid
ax = g.ax

# Add error bars manually using matplotlib
for i, bar in enumerate(ax.patches):
    # Calculate the indices for the model and group
    model_index = i // len(df['group'].unique())
    group_index = i % len(df['group'].unique())
    
    model = df['model'].iloc[model_index]
    group = df['group'].unique()[group_index]
    std_dev = df[(df['model'] == model) & (df['group'] == group)]['std_dev'].values[0]
    
    ax.errorbar(
        bar.get_x() + bar.get_width() / 2, bar.get_height(),
        yerr=std_dev, fmt='none', c='black', capsize=5
    )

# Customize the plot
g.despine(left=True)
g.set_axis_labels("Imitation Targets", "Normalized Return")
g.legend.set_title("Stage 1")

plt.show()

