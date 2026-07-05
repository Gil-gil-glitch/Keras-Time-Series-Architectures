#
## mlp.py
#
#  This is an implementation of a Multilayer Perceptron (MLP) model for univariate time-series forecasting.
#

from keras.models import Sequential
from keras.layers import Dense, Flatten

def build_mlp_model(n_steps, n_features):
    """
    Builds and compiles a Multilayer Perceptron (MLP) for univariate time-series forecasting.
    
    Parameters:
    n_steps (int): The number of lag observations (lookback window size).
    n_features (int): The number of parallel series (1 for univariate data).
    
    Returns:
    keras.models.Sequential: A compiled Keras MLP model.
    """
    # Initialize a linear stack of layers
    model = Sequential()
    
    # 1. Flatten Layer
    # Time-series data is prepared as 3D: (samples, n_steps, n_features).
    # MLP requires a 2D matrix input: (samples, n_steps * n_features).
    # This Flatten layer collapses the temporal dimension into a 1D vector per sample.
    model.add(Flatten(input_shape=(n_steps, n_features)))
    
    # 2. Fully Connected (Dense) Hidden Layer
    # Uses 100 neurons to learn non-linear combinations of the flattened time steps.
    # 'relu' (Rectified Linear Unit) activation helps the network learn complex patterns.
    model.add(Dense(100, activation='relu'))
    
    # 3. Output Layer
    # A single neuron with no activation function (linear activation).
    # This outputs a single continuous numeric value, which is our forecast for time step (t).
    model.add(Dense(1))
    
    # 4. Model Compilation
    # 'adam' is an adaptive learning rate optimization algorithm.
    # 'mse' (Mean Squared Error) is the loss function optimized during training.
    model.compile(optimizer='adam', loss='mse')
    
    return model