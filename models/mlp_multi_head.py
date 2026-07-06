#
## mlp_multi_head.py
#
#  Implementation of a Multi-Head Multilayer Perceptron (MLP)
#  using the Keras Functional API for multivariate input sequences.
#

from keras.models import Model
from keras.layers import Input, Dense, concatenate

def build_multi_head_mlp(n_steps, n_heads):
    """
    Builds a Multi-Head MLP where each input feature/series gets its own 
    dedicated input layer and hidden feature-extraction pipeline.
    
    Parameters:
    n_steps (int): The number of lag observations per head (lookback window).
    n_heads (int): The number of parallel input series (heads).
    
    Returns:
    keras.models.Model: A compiled Keras Functional API Model instance.
    """
    input_layers = []
    head_outputs = []
    
    # 1. Define separate heads for each input series
    for i in range(n_heads):
        # Each head expects a 1D vector of its own historical steps
        input_layer = Input(shape=(n_steps,), name=f'input_head_{i+1}')
        input_layers.append(input_layer)
        
        # Dedicated interpretation layer for this specific head
        head_dense = Dense(30, activation='relu', name=f'dense_head_{i+1}')(input_layer)
        head_outputs.append(head_dense)
        
    # 2. Merge features extracted from all heads
    merged = concatenate(head_outputs, name='merge_heads')
    
    # 3. Fully-connected interpretation layers after fusion
    dense_combined = Dense(50, activation='relu', name='dense_combined')(merged)
    
    # 4. Final single-value output (forecasting the next step of a target variable)
    output_layer = Dense(1, name='output')(dense_combined)
    
    # 5. Construct and compile the model graph
    model = Model(inputs=input_layers, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model