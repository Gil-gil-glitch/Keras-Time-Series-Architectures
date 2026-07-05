#
## rnn.py
#
#  This is an implementation of a Recurrent Neural Network (RNN) model for univariate time-series forecasting.
#



from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

def build_rnn_model(n_steps, n_features, units=50):
    """
    Builds and compiles a Vanilla Recurrent Neural Network (SimpleRNN) for time-series forecasting.
    
    Parameters:
    n_steps (int): The number of lag observations (lookback window size).
    n_features (int): The number of parallel series (1 for univariate data).
    units (int): The dimensionality of the hidden state / working memory space.
    
    Returns:
    keras.models.Sequential: A compiled Keras SimpleRNN model.
    """
    # Initialize a linear stack of layers
    model = Sequential()
    
    # 1. SimpleRNN Layer
    # Expects a 3D input: (samples, n_steps, n_features).
    # It loops through the timeline sequentially, updating an internal state vector.
    # By default, it returns only the final hidden state vector of size `units`.
    model.add(SimpleRNN(units, activation='tanh', input_shape=(n_steps, n_features)))
    
    # 2. Output Layer
    # Takes the final recurrent summary vector and maps it to a single continuous prediction.
    model.add(Dense(1))
    
    # 3. Model Compilation
    # Optimizes the network parameters to minimize Mean Squared Error.
    model.compile(optimizer='adam', loss='mse')
    
    return model