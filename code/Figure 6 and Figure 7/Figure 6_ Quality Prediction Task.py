# Create figure with specific styling
plt.figure(figsize=(12, 8))

# Color mapping for quality levels
quality_colors = {
    0: '#FF6B6B',  # Light red - Low quality
    1: '#FFA726',  # Orange - Moderate quality  
    2: '#FFD93D',  # Yellow - Good quality
    3: '#4ECDC4',  # Teal - High quality
    4: '#2E86AB'   # Dark blue - Excellent quality
}

# Scatter plot with quality-based coloring
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=[quality_colors[l] for l in quality_labels],
                     alpha=0.7, s=30, edgecolor='white', linewidth=0.3)

# Styling and labels
plt.title('t-SNE Visualization of Live Streaming Quality Clusters\n'
          f'(perplexity={tsne_params["perplexity"]}, n_iter={tsne_params["n_iter"]})', 
          fontsize=14, pad=20)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True, alpha=0.2)

# Custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color, markersize=8, 
                            label=f'Quality Level {i}')
                 for i, color in quality_colors.items()]
plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('figure6_quality_tsne.png', dpi=300, bbox_inches='tight')
plt.show()