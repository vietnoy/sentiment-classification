import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CustomLSTMCell(tf.keras.layers.Layer):
    """
    Custom LSTM Cell implementation using TensorFlow operations
    This shows the internal mechanics while using TF for autodiff
    """
    
    def __init__(self, units, **kwargs):
        super(CustomLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]  # [hidden_state, cell_state]
        
    def build(self, input_shape):
        """Initialize weights and biases for LSTM gates"""
        input_dim = input_shape[-1]
        
        # Combined input dimension (input + hidden state)
        combined_dim = input_dim + self.units
        
        # Initialize weights for all 4 gates using Xavier initialization
        initializer = tf.keras.initializers.GlorotUniform()
        
        # Forget gate weights and bias
        self.W_f = self.add_weight(
            name='forget_gate_weights',
            shape=(combined_dim, self.units),
            initializer=initializer,
            trainable=True
        )
        self.b_f = self.add_weight(
            name='forget_gate_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Input gate weights and bias
        self.W_i = self.add_weight(
            name='input_gate_weights',
            shape=(combined_dim, self.units),
            initializer=initializer,
            trainable=True
        )
        self.b_i = self.add_weight(
            name='input_gate_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Candidate values weights and bias
        self.W_c = self.add_weight(
            name='candidate_weights',
            shape=(combined_dim, self.units),
            initializer=initializer,
            trainable=True
        )
        self.b_c = self.add_weight(
            name='candidate_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Output gate weights and bias
        self.W_o = self.add_weight(
            name='output_gate_weights',
            shape=(combined_dim, self.units),
            initializer=initializer,
            trainable=True
        )
        self.b_o = self.add_weight(
            name='output_gate_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs, states, training=None):
        """
        Forward pass through LSTM cell
        
        Args:
            inputs: Current input (batch_size, input_dim)
            states: Previous states [h_prev, c_prev]
            training: Training mode flag
            
        Returns:
            outputs: Current hidden state
            new_states: [new_hidden_state, new_cell_state]
        """
        # Unpack previous states
        h_prev, c_prev = states
        
        # Concatenate input and previous hidden state
        # Shape: (batch_size, input_dim + units)
        combined = tf.concat([inputs, h_prev], axis=-1)
        
        # FORGET GATE: Decide what to forget from cell state
        # f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
        forget_gate = tf.sigmoid(tf.matmul(combined, self.W_f) + self.b_f)
        
        # INPUT GATE: Decide what new information to store
        # i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
        input_gate = tf.sigmoid(tf.matmul(combined, self.W_i) + self.b_i)
        
        # CANDIDATE VALUES: Create new candidate values
        # C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
        candidate_values = tf.tanh(tf.matmul(combined, self.W_c) + self.b_c)
        
        # UPDATE CELL STATE: Combine forget and input gates
        # C_t = f_t * C_{t-1} + i_t * C̃_t
        new_cell_state = forget_gate * c_prev + input_gate * candidate_values
        
        # OUTPUT GATE: Decide what parts of cell state to output
        # o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
        output_gate = tf.sigmoid(tf.matmul(combined, self.W_o) + self.b_o)
        
        # HIDDEN STATE: Filter cell state through output gate
        # h_t = o_t * tanh(C_t)
        new_hidden_state = output_gate * tf.tanh(new_cell_state)
        
        return new_hidden_state, [new_hidden_state, new_cell_state]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialize hidden and cell states with zeros"""
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),  # hidden state
            tf.zeros((batch_size, self.units), dtype=dtype)   # cell state
        ]

class CustomBidirectionalRNN(tf.keras.layers.Layer):
    """
    Custom Bidirectional RNN layer that processes sequences in both directions
    
    BRNN processes the sequence twice:
    1. Forward: from first to last time step
    2. Backward: from last to first time step
    Then combines both outputs
    """
    
    def __init__(self, forward_cell, backward_cell, return_sequences=False, 
                 return_state=False, merge_mode='concat', **kwargs):
        super(CustomBidirectionalRNN, self).__init__(**kwargs)
        self.forward_cell = forward_cell
        self.backward_cell = backward_cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.merge_mode = merge_mode  # 'concat', 'sum', 'mul', 'ave'
        
    def call(self, inputs, initial_state=None, training=None):
        """
        Process sequence in both forward and backward directions
        
        Args:
            inputs: Input sequences (batch_size, sequence_length, input_dim)
            initial_state: Initial states for both directions
            training: Training mode flag
            
        Returns:
            outputs: Combined bidirectional outputs
            final_state: Final states if return_state=True
        """
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        # Initialize states for both directions
        if initial_state is None:
            forward_initial_state = self.forward_cell.get_initial_state(
                batch_size=batch_size, dtype=inputs.dtype
            )
            backward_initial_state = self.backward_cell.get_initial_state(
                batch_size=batch_size, dtype=inputs.dtype
            )
        else:
            forward_initial_state, backward_initial_state = initial_state
        
        # FORWARD PASS: Process sequence from start to end
        forward_outputs, forward_final_states = self._process_forward(
            inputs, forward_initial_state, training
        )
        
        # BACKWARD PASS: Process sequence from end to start
        backward_outputs, backward_final_states = self._process_backward(
            inputs, backward_initial_state, training
        )
        
        # COMBINE OUTPUTS based on merge_mode
        if self.return_sequences:
            # Combine sequence outputs
            combined_outputs = self._merge_outputs(forward_outputs, backward_outputs)
        else:
            # Combine only final outputs
            forward_final = forward_outputs[:, -1, :]  # Last time step
            backward_final = backward_outputs[:, 0, :]  # First time step (reversed)
            combined_outputs = self._merge_final_outputs(forward_final, backward_final)
        
        if self.return_state:
            return combined_outputs, [forward_final_states, backward_final_states]
        else:
            return combined_outputs
    
    def _process_forward(self, inputs, initial_state, training):
        """Process sequence in forward direction"""
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        # Use tf.TensorArray for collecting outputs
        outputs_ta = tf.TensorArray(
            dtype=inputs.dtype,
            size=sequence_length,
            dynamic_size=False
        )
        
        # Forward loop variables
        time = tf.constant(0)
        states = initial_state
        
        def forward_condition(time, states, outputs_ta_loop):
            return tf.less(time, sequence_length)
        
        def forward_body(time, states, outputs_ta_loop):
            # Get input at current time step (forward direction)
            input_t = inputs[:, time, :]
            
            # Process through forward cell
            output_t, new_states = self.forward_cell(input_t, states, training=training)
            
            # Store output
            outputs_ta_loop = outputs_ta_loop.write(time, output_t)
            
            return time + 1, new_states, outputs_ta_loop
        
        # Run forward loop
        final_time, final_states, final_outputs_ta = tf.while_loop(
            forward_condition,
            forward_body,
            [time, states, outputs_ta],
            parallel_iterations=32
        )
        
        # Stack outputs: (seq_len, batch_size, units)
        outputs = final_outputs_ta.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, seq_len, units)
        
        return outputs, final_states
    
    def _process_backward(self, inputs, initial_state, training):
        """Process sequence in backward direction"""
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        # Use tf.TensorArray for collecting outputs
        outputs_ta = tf.TensorArray(
            dtype=inputs.dtype,
            size=sequence_length,
            dynamic_size=False
        )
        
        # Backward loop variables (start from end)
        time = sequence_length - 1
        states = initial_state
        
        def backward_condition(time, states, outputs_ta_loop):
            return tf.greater_equal(time, 0)
        
        def backward_body(time, states, outputs_ta_loop):
            # Get input at current time step (backward direction)
            input_t = inputs[:, time, :]
            
            # Process through backward cell
            output_t, new_states = self.backward_cell(input_t, states, training=training)
            
            # Store output at original time position
            outputs_ta_loop = outputs_ta_loop.write(time, output_t)
            
            return time - 1, new_states, outputs_ta_loop
        
        # Run backward loop
        final_time, final_states, final_outputs_ta = tf.while_loop(
            backward_condition,
            backward_body,
            [time, states, outputs_ta],
            parallel_iterations=32
        )
        
        # Stack outputs: (seq_len, batch_size, units)
        outputs = final_outputs_ta.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, seq_len, units)
        
        return outputs, final_states
    
    def _merge_outputs(self, forward_outputs, backward_outputs):
        """Merge forward and backward sequence outputs"""
        if self.merge_mode == 'concat':
            # Concatenate along feature dimension
            return tf.concat([forward_outputs, backward_outputs], axis=-1)
        elif self.merge_mode == 'sum':
            return forward_outputs + backward_outputs
        elif self.merge_mode == 'mul':
            return forward_outputs * backward_outputs
        elif self.merge_mode == 'ave':
            return (forward_outputs + backward_outputs) / 2.0
        else:
            raise ValueError(f"Unknown merge_mode: {self.merge_mode}")
    
    def _merge_final_outputs(self, forward_final, backward_final):
        """Merge final outputs from both directions"""
        if self.merge_mode == 'concat':
            return tf.concat([forward_final, backward_final], axis=-1)
        elif self.merge_mode == 'sum':
            return forward_final + backward_final
        elif self.merge_mode == 'mul':
            return forward_final * backward_final
        elif self.merge_mode == 'ave':
            return (forward_final + backward_final) / 2.0
        else:
            raise ValueError(f"Unknown merge_mode: {self.merge_mode}")

class DeepBRNNModel(tf.keras.Model):
    """
    Deep Bidirectional RNN Model with 2 custom BRNN layers + Dense layers
    Using the same embedding matrix from Word2Vec
    """
    
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, 
                 brnn1_units=64, brnn2_units=32, dense_units=32, 
                 dropout_rate=0.3, merge_mode='concat', **kwargs):
        super(DeepBRNNModel, self).__init__(**kwargs)
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.brnn1_units = brnn1_units
        self.brnn2_units = brnn2_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.merge_mode = merge_mode
        
        # EMBEDDING LAYER: Use pre-trained Word2Vec embeddings
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,  # Keep Word2Vec embeddings frozen
            mask_zero=False,
            name='word2vec_embedding'
        )
        
        # FIRST BIDIRECTIONAL RNN LAYER
        self.brnn1_forward_cell = CustomLSTMCell(brnn1_units, name='brnn1_forward_cell')
        self.brnn1_backward_cell = CustomLSTMCell(brnn1_units, name='brnn1_backward_cell')
        self.brnn1_layer = CustomBidirectionalRNN(
            self.brnn1_forward_cell, 
            self.brnn1_backward_cell,
            return_sequences=True,  # Return all sequences for next BRNN
            merge_mode=merge_mode,
            name='custom_brnn1'
        )
        
        # DROPOUT AFTER FIRST BRNN
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        
        # SECOND BIDIRECTIONAL RNN LAYER
        # Input size is doubled due to bidirectional concatenation
        input_size_brnn2 = brnn1_units * 2 if merge_mode == 'concat' else brnn1_units
        self.brnn2_forward_cell = CustomLSTMCell(brnn2_units, name='brnn2_forward_cell')
        self.brnn2_backward_cell = CustomLSTMCell(brnn2_units, name='brnn2_backward_cell')
        self.brnn2_layer = CustomBidirectionalRNN(
            self.brnn2_forward_cell,
            self.brnn2_backward_cell,
            return_sequences=False,  # Return only final output
            merge_mode=merge_mode,
            name='custom_brnn2'
        )
        
        # DROPOUT AFTER SECOND BRNN
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name='dropout2')
        
        # DENSE LAYERS
        # Input size depends on merge_mode of final BRNN
        final_brnn_output_size = brnn2_units * 2 if merge_mode == 'concat' else brnn2_units
        self.dense1 = tf.keras.layers.Dense(
            dense_units, 
            activation='relu', 
            name='dense_hidden'
        )
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate, name='dropout3')
        
        # OUTPUT LAYER
        self.output_layer = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            name='output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through the entire BRNN model
        
        Args:
            inputs: Token sequences (batch_size, sequence_length)
            training: Training mode flag
            
        Returns:
            outputs: Predictions (batch_size, 1)
        """
        
        # 1. EMBEDDING LOOKUP
        # Convert token indices to dense vectors using Word2Vec embeddings
        # Shape: (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(inputs)
        
        # 2. FIRST BIDIRECTIONAL RNN LAYER
        # Process through first custom BRNN layer
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, brnn1_units*2)
        brnn1_out = self.brnn1_layer(embedded, training=training)
        brnn1_out = self.dropout1(brnn1_out, training=training)
        
        # 3. SECOND BIDIRECTIONAL RNN LAYER  
        # Process through second custom BRNN layer
        # Shape: (batch_size, seq_len, brnn1_units*2) -> (batch_size, brnn2_units*2)
        brnn2_out = self.brnn2_layer(brnn1_out, training=training)
        brnn2_out = self.dropout2(brnn2_out, training=training)
        
        # 4. DENSE LAYERS
        # Hidden dense layer with ReLU activation
        # Shape: (batch_size, brnn2_units*2) -> (batch_size, dense_units)
        dense_out = self.dense1(brnn2_out)
        dense_out = self.dropout3(dense_out, training=training)
        
        # 5. OUTPUT LAYER
        # Binary classification with sigmoid
        # Shape: (batch_size, dense_units) -> (batch_size, 1)
        outputs = self.output_layer(dense_out)
        
        return outputs
    
    def get_config(self):
        """Return model configuration for serialization"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'brnn1_units': self.brnn1_units,
            'brnn2_units': self.brnn2_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'merge_mode': self.merge_mode
        }

def create_and_train_custom_deep_brnn(data, epochs=10, learning_rate=0.001, 
                                     batch_size=32, merge_mode='concat'):
    """
    Create and train the custom Deep BRNN model
    
    Args:
        data: Data dictionary from gensim pipeline
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        merge_mode: How to combine forward and backward outputs
        
    Returns:
        model: Trained model
        history: Training history
    """
    
    print(" CREATING CUSTOM DEEP BIDIRECTIONAL RNN WITH TENSORFLOW")
    print("=" * 70)
    
    # Create model instance
    model = DeepBRNNModel(
        vocab_size=data['vocab_size'],
        embedding_dim=data['embedding_dim'],
        embedding_matrix=data['embedding_matrix'],
        brnn1_units=64,
        brnn2_units=32,
        dense_units=32,
        dropout_rate=0.3,
        merge_mode=merge_mode
    )
    
    # Compile model with optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Build model by calling it once with proper input shape
    sequence_length = data['X_train'].shape[1]
    dummy_input = tf.zeros((1, sequence_length), dtype=tf.int32)
    _ = model(dummy_input)
    
    # Print model architecture
    print("\n BIDIRECTIONAL RNN MODEL ARCHITECTURE:")
    print(f"Input: (batch_size, {sequence_length}) - Token sequences")
    print(f"Embedding: {data['vocab_size']} -> {data['embedding_dim']} (Word2Vec frozen)")
    print(f"BRNN 1: {data['embedding_dim']} -> 64*2 = 128 (Custom bidirectional)")
    print(f"BRNN 2: 128 -> 32*2 = 64 (Custom bidirectional)")
    print(f"Dense: 64 -> 32 (ReLU)")
    print(f"Output: 32 -> 1 (Sigmoid)")
    print(f"Merge mode: {merge_mode}")
    
    # Count parameters
    try:
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        print(f"Note: BRNN has ~2x parameters compared to unidirectional RNN")
    except ValueError:
        print("Parameters will be counted after first training batch")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_train'], data['y_train']
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_val'], data['y_val']
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\n TRAINING BIDIRECTIONAL RNN")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Sequence length: {sequence_length}")
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Count parameters after training
    try:
        total_params = model.count_params()
        print(f"\n Bidirectional RNN successfully trained!")
        print(f"Total parameters: {total_params:,}")
    except:
        print("\n Bidirectional RNN successfully trained!")
    
    return model, history

def analyze_brnn_internals(model, sample_input):
    """
    Analyze what happens inside the custom BRNN layers
    """
    print("\n ANALYZING BIDIRECTIONAL RNN INTERNALS")
    print("=" * 50)
    
    # Get intermediate outputs
    sample_batch = tf.expand_dims(sample_input, 0)
    
    # Embedding output
    embedded = model.embedding(sample_batch)
    print(f"Embedding output shape: {embedded.shape}")
    print(f"Sample embedding (first 5 dims): {embedded[0, 0, :5].numpy()}")
    
    # BRNN1 output
    brnn1_out = model.brnn1_layer(embedded)
    print(f"BRNN1 output shape: {brnn1_out.shape}")
    print(f"Note: Shape doubled due to bidirectional concatenation")
    print(f"BRNN1 output range: [{tf.reduce_min(brnn1_out):.4f}, {tf.reduce_max(brnn1_out):.4f}]")
    
    # BRNN2 output  
    brnn2_out = model.brnn2_layer(brnn1_out)
    print(f"BRNN2 output shape: {brnn2_out.shape}")
    print(f"BRNN2 output range: [{tf.reduce_min(brnn2_out):.4f}, {tf.reduce_max(brnn2_out):.4f}]")
    
    # Final prediction
    prediction = model(sample_batch)
    print(f"Final prediction: {prediction[0, 0].numpy():.4f}")
    
    print(f"\n BIDIRECTIONAL RNN CHARACTERISTICS:")
    print(f"• Processes sequence in BOTH directions")
    print(f"• Forward: past → future context")
    print(f"• Backward: future → past context") 
    print(f"• Combines both for richer representation")
    print(f"• Better for sentiment analysis (can see full context)")

def plot_brnn_training_history(history):
    """Plot training curves for BRNN model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Custom Deep BRNN - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Acc', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='red')
    ax2.set_title('Custom Deep BRNN - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_custom_brnn_model(model, data):
    """Evaluate the custom BRNN model on test set"""
    print("\n EVALUATING CUSTOM BRNN MODEL")
    print("=" * 40)
    
    # Evaluate on test set
    test_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_test'], data['y_test']
    )).batch(32)
    
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions for classification report
    predictions = model.predict(test_dataset, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = data['y_test'].astype(int)
    
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=['Negative', 'Positive']))
    
    return test_accuracy

# Main usage function
def main_custom_deep_brnn(data, merge_mode='concat'):
    """
    Main function to create, train, and evaluate custom deep BRNN
    """
    
    # Create and train model
    model, history = create_and_train_custom_deep_brnn(
        data, 
        epochs=10, 
        learning_rate=0.001, 
        batch_size=32,
        merge_mode=merge_mode
    )
    
    # Plot training history
    plot_brnn_training_history(history)
    
    # Analyze model internals
    sample_input = data['X_val'][0]
    analyze_brnn_internals(model, sample_input)
    
    # Evaluate model
    test_accuracy = evaluate_custom_brnn_model(model, data)
    
    print(f"\n CUSTOM DEEP BRNN TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Merge mode used: {merge_mode}")
    
    return model, history