#
## lstm.py
#
#  This is an implementation of various Long Short-Term Memory (LSTM) models for univariate time-series forecasting.
#

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

def build_vanilla_lstm(n_steps, n_features, units=50):
    """
    Builds a single-layer Vanilla LSTM for univariate sequence forecasting.
    """
    model = Sequential()
    
    # 1. Vanilla LSTM Layer
    # Processes the 3D shape sequentially, maintaining an internal long-term cell state.
    # Outputs a 2D tensor representing the hidden state at the final time step.
    model.add(LSTM(units, activation='relu', input_shape=(n_steps, n_features)))
    
    # 2. Dense Regressor Output
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model


def build_stacked_lstm(n_steps, n_features, units=50):
    """
    Builds a multi-layer Stacked LSTM for deep hierarchical sequence feature extraction.
    """
    model = Sequential()
    
    # 1. First Stacked LSTM Layer
    # return_sequences=True tells Keras to output the hidden states for ALL time steps.
    # This keeps the tensor 3D: (samples, n_steps, units), matching the input requirement for Layer 2.
    model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    
    # 2. Second Stacked LSTM Layer
    # Processes the sequence outputs from Layer 1 and returns only the final hidden state (2D).
    model.add(LSTM(units, activation='relu'))
    
    # 3. Dense Regressor Output
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model


def build_bidirectional_lstm(n_steps, n_features, units=50):
    """
    Builds a Bidirectional LSTM that processes lookback windows forward and backward.
    """
    model = Sequential()
    
    # 1. Bidirectional Wrapper
    # Clones the LSTM layer into two copies. One processes data forward; the other backward.
    # The final outputs of both paths are concatenated together, creating a 2D output vector of size (units * 2).
    model.add(Bidirectional(LSTM(units, activation='relu'), input_shape=(n_steps, n_features)))
    
    # 2. Dense Regressor Output
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model