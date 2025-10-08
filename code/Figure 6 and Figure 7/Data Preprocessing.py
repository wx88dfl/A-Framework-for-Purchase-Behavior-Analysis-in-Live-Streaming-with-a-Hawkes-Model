import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Feature extraction from ODE-LSTM encoder outputs
# Shape: (n_samples, n_features) where n_features = 64 (for D=64)
feature_vectors = model.get_encoder_outputs()  # Shape: (12899, 64)

# Standardization (critical for t-SNE performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_vectors)

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Standardized features - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")