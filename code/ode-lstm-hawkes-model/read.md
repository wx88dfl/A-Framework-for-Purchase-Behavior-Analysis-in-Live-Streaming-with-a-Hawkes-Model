# Livestream User Behavior Analysis with ODE-LSTM Hawkes Model

This project provides a comprehensive framework for analyzing user behavior in livestreams through temporal sequence modeling and continuous-time dynamics.

## Project Structure

The codebase is organized into the following main scripts:

### Core Scripts

- **`ablation_study.py`**
  - Conducts ablation experiments to evaluate the contribution of different model components.
  - Compares performance of:
    - Baseline LSTM
    - ODE-LSTM variant
    - Event decoder variant
    - Room decoder variant
    - Full model
  - Generates comparative visualizations of accuracy, F1 scores, MCC, and training time.

- **`data_processing.py`**
  - Handles data loading, preprocessing, and sequence creation.
  - Key Functions:
    - `load_and_preprocess_data()`: Main pipeline for data preparation.
    - `create_sequences()`: Converts raw data into input sequences using a sliding window technique.
    - `prepare_data_for_model()`: Splits data into train/validation/test sets and converts to PyTorch tensors.
    - `assign_quality_level()`: Categorizes stream quality based on predefined thresholds.

- **`model.py`**
  - Contains the ODE-LSTM Hawkes model implementation.
  - Features:
    - ODE-LSTM cell implementation with continuous time evolution.
    - Forward pass handling event sequences and timestamps.
    - Prediction heads for event type, event timing, and quality level.

- **`train.py`**
  - Implements model training with early stopping.
  - Handles data loading, model initialization, and optimization.
  - Tracks training/validation loss for total loss, event-level loss, and room-level loss.
  - Generates training curve visualizations and saves model checkpoints.

- **`evaluate.py`**
  - Evaluates trained models on test data.
  - Computes performance metrics (accuracy, F1, MAE, RMSE, MCC).
  - Generates confusion matrices and prediction scatter plots.
  - Saves evaluation results in CSV format.

- **`visualization.py`**
  - Provides visualization utilities.
  - Compares model performance across different architectures.
  - Generates t-SNE visualizations of hidden states.
  - Creates comparative bar charts for model metrics.

- **`main.py`**
  - Main entry point for the application.
  - Parses command-line arguments.
  - Coordinates training, evaluation, ablation studies, and visualization.
  - Supports different run modes.

## Usage

Run the main script with desired parameters:

```bash
python main.py --mode all --hidden_dim 64 --batch_size 128 --lr 0.001 --epochs 100