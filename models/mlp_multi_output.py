#
## mlp_multi_output.py
#
#  Implementation of a Multi-Output Multilayer Perceptron (MLP)
#  for multi-step-ahead time-series forecasting.
#

from keras.models import Model
from keras.layers import Input, Dense

def build_multi_output_mlp(n_steps, n_features, n_outputs):
    """
    Builds an MLP capable of predicting multiple future time steps at once.
    
    Parameters:
    n_steps (int): The number of lag observations (lookback window).
    n_features (int): The number of parallel series (1 for univariate data).
    n_outputs (int): The number of future time steps to predict (forecast horizon).
    
    Returns:
    keras.models.Model: A compiled Keras Functional API Model instance.
    """
    # 1. Flattened Input Layer
    # For a classic MLP, we flatten the spatial/temporal dimension upfront.
    input_shape = n_steps * n_features
    input_layer = Input(shape=(input_shape,), name='input_layer')
    
    # 2. Hidden Processing Layers
    dense_1 = Dense(100, activation='relu', name='hidden_dense_1')(input_layer)
    dense_2 = Dense(50, activation='relu', name='hidden_dense_2')(dense_1)
    
    # 3. Multi-Output Layer
    # Instead of pointing to 1, the output vector matches your forecast horizon size.
    output_layer = Dense(n_outputs, name='multi_step_output')(dense_2)
    
    # 4. Construct and compile the model graph
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model