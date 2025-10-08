# Binary classification: users who removed bookmarks vs those who didn't
bookmark_removal_labels = df['bookmark_removed'].values  # Binary labels

plt.figure(figsize=(10, 8))

# Color scheme for binary classification
colors = ['#2E86AB', '#FF6B6B']  # Blue: kept bookmarks, Red: removed bookmarks
color_list = [colors[1] if removed else colors[0] 
             for removed in bookmark_removal_labels]

scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=color_list, alpha=0.6, s=25, 
                     edgecolor='white', linewidth=0.2)

plt.title('t-SNE Visualization: Bookmark Removal Behavior\n'
          '(Community Awareness Pattern)', fontsize=14, pad=20)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)

# Binary classification legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], 
           markersize=8, label='Maintained Bookmarks'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], 
           markersize=8, label='Removed Bookmarks')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('figure7_bookmark_removal_tsne.png', dpi=300, bbox_inches='tight')
plt.show()