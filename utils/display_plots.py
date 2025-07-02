import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.preprocessing import LabelEncoder

def plot(embeddings, labels, title):
    le = LabelEncoder()
    le.fit(labels)
    # Create color mapping
    unique_labels = np.unique(labels)
    color_map = get_cmap('viridis', len(unique_labels))

    # Plot each label group separately for better legend
    for i, label in enumerate(unique_labels):
        # Get original label name
        label_name = le.inverse_transform([label])[0]
        
        # Get indices for this label
        indices = np.where(labels == label)
        
        # Get embedding points for this label
        points = embeddings[indices]
        
        # Plot with consistent color
        plt.scatter(
            points[:, 0], 
            points[:, 1], 
            color=color_map(i),
            label=label_name,
            s=5
        )

    plt.title(title, fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Categories')

    plt.show()