# Comprehensive t-SNE parameter settings
tsne_params = {
    'n_components': 2,                    # 2D visualization
    'random_state': RANDOM_SEED,          # Fixed random seed
    'perplexity': 30,                     # Optimized for our dataset size
    'n_iter': 1000,                       # Number of iterations
    'learning_rate': 200,                 # Default learning rate
    'early_exaggeration': 12.0,           # Early exaggeration factor
    'min_grad_norm': 1e-7,                # Minimum gradient norm
    'metric': 'euclidean',                # Distance metric
    'init': 'random',                     # Initialization method
    'verbose': 1,                         # Progress reporting
    'angle': 0.5                          # Balance between speed and accuracy
}

# Initialize and fit t-SNE
tsne = TSNE(**tsne_params)
X_embedded = tsne.fit_transform(X_scaled)

print(f"t-SNE embedding completed. Final KL divergence: {tsne.kl_divergence_:.4f}")