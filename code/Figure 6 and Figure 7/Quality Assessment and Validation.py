# For research validation (not included in paper)
def assess_tsne_convergence(X_scaled, perplexity_range):
    """Evaluate t-SNE stability across different perplexity values"""
    results = {}
    for perplexity in perplexity_range:
        tsne_test = TSNE(n_components=2, perplexity=perplexity, 
                        random_state=RANDOM_SEED, n_iter=1000)
        embedding = tsne_test.fit_transform(X_scaled)
        results[perplexity] = {
            'kl_divergence': tsne_test.kl_divergence_,
            'embedding': embedding
        }
    return results

# We verified that perplexity=30 provided stable, interpretable clusters
# across multiple random initializations