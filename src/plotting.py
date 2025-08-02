import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_radar_chart(df, title, save_path=None):
    """
    Generates a radar chart from a dataframe where rows are groups and columns are metrics.
    """
    categories = df.columns.tolist()
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=9)
    plt.ylim(0, 1)
    
    # Plot each individual
    for i in range(len(df)):
        values = df.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {df.index[i]}")
        ax.fill(angles, values, alpha=0.25)
    
    # Add a title and legend
    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
