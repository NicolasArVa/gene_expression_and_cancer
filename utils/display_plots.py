import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def plot(embeddings, labels=None, title=''):
    # Create color mapping
    if labels is not None:
        unique_labels = np.unique(labels)

        color_palette = sns.color_palette('deep', 8)
        colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in range(len(unique_labels))]

        # Plot each label group separately for better legend
        for i, label in enumerate(unique_labels):        
            # Get indices for this label
            indices = np.where(labels == label)
            
            # Get embedding points for this label
            points = embeddings[indices]
            
            # Plot with consistent color
            plt.scatter(
                points[:, 0], 
                points[:, 1], 
                color=colors[i],
                label=label,
                s=5
            )

        plt.legend(title='Categories')
        
    else:
        plt.scatter(
            embeddings[:, 0], 
            embeddings[:, 1],
            s=5
        )

    plt.title(title, fontsize=14)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)

    plt.show()