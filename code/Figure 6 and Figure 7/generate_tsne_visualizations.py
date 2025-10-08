import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import torch

def generate_tsne_visualizations(model, test_loader, device):
    """
    Generate t-SNE visualizations for the proposed ODE-LSTM Hawkes model
    
    Parameters:
    model: Trained ODE-LSTM Hawkes model
    test_loader: DataLoader containing test dataset
    device: Computation device (cpu/cuda)
    """
    
    # Set random seed for reproducibility (as mentioned in paper)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Extract features and labels
    room_level_features = []
    user_sequence_features = []
    quality_labels = []
    bookmark_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Assuming batch contains: 
            # behavioral_sequences, user_features, room_features, quality_labels, bookmark_labels
            behavioral_seq = batch['behavioral_sequences'].to(device)
            user_feat = batch['user_features'].to(device)
            room_feat = batch['room_features'].to(device)
            
            # Forward pass through encoder
            encoded_sequences = model.encoder(behavioral_seq, user_feat, room_feat)
            
            # Room-level features for Figure 6 (Quality Prediction)
            room_level_repr = model.room_level_decoder.aggregate(encoded_sequences)
            room_level_features.append(room_level_repr.cpu().numpy())
            quality_labels.append(batch['quality_labels'].numpy())
            
            # User-level features for Figure 7 (Bookmark Removal)
            user_sequence_repr = encoded_sequences[:, -1, :]  # Final hidden state
            user_sequence_features.append(user_sequence_repr.cpu().numpy())
            bookmark_labels.append(batch['bookmark_labels'].numpy())
    
    # Convert to numpy arrays
    room_level_features = np.vstack(room_level_features)
    user_sequence_features = np.vstack(user_sequence_features)
    quality_labels = np.concatenate(quality_labels)
    bookmark_labels = np.concatenate(bookmark_labels)
    
    # t-SNE parameters (as discussed)
    tsne_params = {
        'n_components': 2,
        'random_state': 42,
        'perplexity': 30,
        'n_iter': 1000,
        'init': 'pca',
        'learning_rate': 'auto'
    }
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Figure 6: Quality Prediction Task
    print("Generating Figure 6: Quality Prediction Visualization...")
    features_quality_scaled = StandardScaler().fit_transform(room_level_features)
    tsne_quality = TSNE(**tsne_params)
    features_2d_quality = tsne_quality.fit_transform(features_quality_scaled)
    
    plot_tsne(ax1, features_2d_quality, quality_labels, 
              'Figure 6: Quality Prediction Task', 'Quality Level')
    
    # Figure 7: Bookmark Removal Prediction
    print("Generating Figure 7: Bookmark Removal Visualization...")
    features_bookmark_scaled = StandardScaler().fit_transform(user_sequence_features)
    tsne_bookmark = TSNE(**tsne_params)
    features_2d_bookmark = tsne_bookmark.fit_transform(features_bookmark_scaled)
    
    plot_tsne(ax2, features_2d_bookmark, bookmark_labels,
              'Figure 7: Predicting Bookmark Removal in Live Streams', 
              'Bookmark Status')
    
    plt.tight_layout()
    plt.savefig('tsne_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return features_2d_quality, features_2d_bookmark

def plot_tsne(ax, features_2d, labels, title, legend_title):
    """
    Helper function to plot t-SNE results
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if legend_title == 'Quality Level':
            # For quality levels: 0=Low, 4=Excellent
            label_names = ['Low', 'Moderate', 'Good', 'High', 'Excellent']
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[colors[i]], label=label_names[label], alpha=0.7, s=30)
        else:
            # For bookmark status
            label_names = ['Non-bookmark', 'Bookmark']
            markers = ['o', 's']  # circle for non-bookmark, square for bookmark
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[colors[i]], marker=markers[label], 
                      label=label_names[label], alpha=0.7, s=30)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(title=legend_title)
    ax.grid(True, alpha=0.3)

# Alternative implementation for feature extraction only
def extract_features_for_tsne(model, dataloader, device):
    """
    Extract features for t-SNE visualization without plotting
    """
    model.eval()
    
    # Storage for features and labels
    all_features = {'quality': [], 'bookmark': []}
    all_labels = {'quality': [], 'bookmark': []}
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract features (implementation depends on your model architecture)
            quality_features = extract_quality_features(model, batch, device)
            bookmark_features = extract_bookmark_features(model, batch, device)
            
            all_features['quality'].append(quality_features.cpu().numpy())
            all_features['bookmark'].append(bookmark_features.cpu().numpy())
            all_labels['quality'].append(batch['quality_labels'].numpy())
            all_labels['bookmark'].append(batch['bookmark_labels'].numpy())
    
    # Concatenate all batches
    for key in all_features:
        all_features[key] = np.vstack(all_features[key])
        all_labels[key] = np.concatenate(all_labels[key])
    
    return all_features, all_labels

def extract_quality_features(model, batch, device):
    """
    Extract room-level features for quality prediction task
    """
    # Implementation depends on your specific model architecture
    # This is a placeholder - adapt to your actual model
    behavioral_seq = batch['behavioral_sequences'].to(device)
    encoded = model.encoder(behavioral_seq)
    room_level_features = torch.mean(encoded, dim=1)  # Average pooling
    return room_level_features

def extract_bookmark_features(model, batch, device):
    """
    Extract user-level features for bookmark prediction task  
    """
    # Implementation depends on your specific model architecture
    # This is a placeholder - adapt to your actual model
    behavioral_seq = batch['behavioral_sequences'].to(device)
    encoded = model.encoder(behavioral_seq)
    user_features = encoded[:, -1, :]  # Last hidden state
    return user_features

# Usage example:
if __name__ == "__main__":
    # Initialize your model and data loader
    # model = YourODELSTMHawkesModel(...)
    # test_loader = YourDataLoader(...)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate visualizations
    # features_2d_quality, features_2d_bookmark = generate_tsne_visualizations(
    #     model, test_loader, device
    # )
    pass