#
## cnn.py
#
#  This is an implementation of a 1D Convolutional Neural Network (CNN) model for univariate time-series forecasting.

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def build_cnn_model(n_steps, n_features):
    """
    Builds and compiles a 1D Convolutional Neural Network (CNN) for time-series forecasting.
    
    Parameters:
    n_steps (int): The number of lag observations (lookback window size).
    n_features (int): The number of parallel series (1 for univariate data).
    
    Returns:
    keras.models.Sequential: A compiled Keras 1D CNN model.
    """
    # Initialize a linear stack of layers
    model = Sequential()
    
    # 1. 1D Convolutional Layer
    # Expects 3D input: (samples, n_steps, n_features).
    # filters=64: The layer learns 64 unique feature detectors.
    # kernel_size=2: Each filter shifts across 2 consecutive time steps at a time.
    # 'relu': Rectified Linear Unit activation maps non-linear relationship patterns.
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    
    # 2. 1D Max Pooling Layer
    # Downsamples the temporal features by taking the maximum value over a sliding window of size 2.
    # This reduces computational load and extracts the most distinct structural signals.
    model.add(MaxPooling1D(pool_size=2))
    
    # 3. Flatten Layer
    # Transmutes the extracted 3D feature maps into a 1D feature vector per sample.
    # This step bridges the convolutional feature extractor to the feedforward dense network.
    model.add(Flatten())
    
    # 4. Dense (Fully Connected) Hidden Layer
    # Interprets the flattened feature vectors using 50 fully connected neurons.
    model.add(Dense(50, activation='relu'))
    
    # 5. Output Layer
    # Maps the interpreted features to a single continuous numeric prediction.
    model.add(Dense(1))
    
    # 6. Model Compilation
    # Configures the training process using the Adam optimizer and Mean Squared Error loss.
    model.compile(optimizer='adam', loss='mse')
    
    return model