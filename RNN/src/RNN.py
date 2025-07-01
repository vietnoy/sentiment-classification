import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CustomRNNCell(tf.keras.layers.Layer):
    """
    Custom Simple RNN Cell implementation using TensorFlow operations
    This shows the internal mechanics of a basic RNN cell
    
    Simple RNN Formula:
    h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
    """
    
    def __init__(self, units, **kwargs):
        super(CustomRNNCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  # Only hidden state (no cell state like LSTM)
        
    def build(self, input_shape):
        """Initialize weights and biases for Simple RNN"""
        input_dim = input_shape[-1]
        
        # Initialize weights using Xavier initialization
        initializer = tf.keras.initializers.GlorotUniform()
        
        # Input to hidden weights
        self.W_x = self.add_weight(
            name='input_weights',
            shape=(input_dim, self.units),
            initializer=initializer,
            trainable=True
        )
        
        # Hidden to hidden weights (recurrent weights)
        self.W_h = self.add_weight(
            name='recurrent_weights',
            shape=(self.units, self.units),
            initializer=initializer,
            trainable=True
        )
        
        # Bias
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs, states, training=None):
        """
        Forward pass through Simple RNN cell
        
        Args:
            inputs: Current input (batch_size, input_dim)
            states: Previous hidden state (batch_size, units)
            training: Training mode flag
            
        Returns:
            outputs: Current hidden state
            new_states: [new_hidden_state] (same as outputs for simple RNN)
        """
        # For simple RNN, states is just the hidden state (not a list)
        if isinstance(states, list):
            h_prev = states[0]
        else:
            h_prev = states
        
        # SIMPLE RNN COMPUTATION:
        # h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
        
        # Linear transformation of input
        input_part = tf.matmul(inputs, self.W_x)
        
        # Linear transformation of previous hidden state
        hidden_part = tf.matmul(h_prev, self.W_h)
        
        # Combine input, hidden, and bias, then apply tanh activation
        new_hidden_state = tf.tanh(input_part + hidden_part + self.b)
        
        # For simple RNN, output is the same as hidden state
        return new_hidden_state, [new_hidden_state]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialize hidden state with zeros"""
        return [tf.zeros((batch_size, self.units), dtype=dtype)]

class CustomRNN(tf.keras.layers.Layer):
    """
    Custom RNN layer that processes sequences using CustomRNNCell
    Using TensorFlow's dynamic_rnn equivalent for proper sequence processing
    """
    
    def __init__(self, cell, return_sequences=False, return_state=False, **kwargs):
        super(CustomRNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        
    def call(self, inputs, initial_state=None, training=None):
        """
        Process entire sequence through RNN using tf.while_loop
        
        Args:
            inputs: Input sequences (batch_size, sequence_length, input_dim)
            initial_state: Initial states for RNN
            training: Training mode flag
            
        Returns:
            outputs: Sequence outputs or final output
            final_state: Final states if return_state=True
        """
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        # Initialize states if not provided
        if initial_state is None:
            initial_state = self.cell.get_initial_state(
                batch_size=batch_size, 
                dtype=inputs.dtype
            )
        
        # Use a simpler approach: tf.TensorArray for collecting outputs
        if self.return_sequences:
            outputs_ta = tf.TensorArray(
                dtype=inputs.dtype,
                size=sequence_length,
                dynamic_size=False
            )
        
        # Initial loop variables
        time = tf.constant(0)
        states = initial_state
        
        def loop_condition(time, states, outputs_ta_loop):
            return tf.less(time, sequence_length)
        
        def loop_body(time, states, outputs_ta_loop):
            # Get input at current time step
            input_t = inputs[:, time, :]
            
            # Process through RNN cell
            output_t, new_states = self.cell(input_t, states, training=training)
            
            # Store output if needed
            if self.return_sequences:
                outputs_ta_loop = outputs_ta_loop.write(time, output_t)
            
            return time + 1, new_states, outputs_ta_loop
        
        if self.return_sequences:
            # Run the loop
            final_time, final_states, final_outputs_ta = tf.while_loop(
                loop_condition,
                loop_body,
                [time, states, outputs_ta],
                parallel_iterations=32
            )
            
            # Stack outputs
            outputs = final_outputs_ta.stack()  # (seq_len, batch_size, units)
            outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, seq_len, units)
        else:
            # Just need final output - simpler loop
            def simple_loop_body(time, states):
                input_t = inputs[:, time, :]
                output_t, new_states = self.cell(input_t, states, training=training)
                return time + 1, new_states
            
            def simple_loop_condition(time, states):
                return tf.less(time, sequence_length)
            
            final_time, final_states = tf.while_loop(
                simple_loop_condition,
                simple_loop_body,
                [time, states],
                parallel_iterations=32
            )
            
            outputs = final_states[0]  # Final hidden state
        
        if self.return_state:
            return outputs, final_states
        else:
            return outputs

class DeepRNNModel(tf.keras.Model):
    """
    Deep Simple RNN Model with 2 custom RNN layers + Dense layers
    Using the same embedding matrix from Word2Vec
    """
    
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, 
                 rnn1_units=64, rnn2_units=32, dense_units=32, 
                 dropout_rate=0.3, **kwargs):
        super(DeepRNNModel, self).__init__(**kwargs)
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn1_units = rnn1_units
        self.rnn2_units = rnn2_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        # EMBEDDING LAYER: Use pre-trained Word2Vec embeddings
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,  # Keep Word2Vec embeddings frozen
            mask_zero=False,  # Avoid the NaN issue we had before
            name='word2vec_embedding'
        )
        
        # FIRST RNN LAYER: Custom implementation
        self.rnn1_cell = CustomRNNCell(rnn1_units, name='rnn1_cell')
        self.rnn1_layer = CustomRNN(
            self.rnn1_cell, 
            return_sequences=True,  # Return all sequences for next RNN
            name='custom_rnn1'
        )
        
        # DROPOUT AFTER FIRST RNN
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        
        # SECOND RNN LAYER: Custom implementation  
        self.rnn2_cell = CustomRNNCell(rnn2_units, name='rnn2_cell')
        self.rnn2_layer = CustomRNN(
            self.rnn2_cell,
            return_sequences=False,  # Return only final output
            name='custom_rnn2'
        )
        
        # DROPOUT AFTER SECOND RNN
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name='dropout2')
        
        # DENSE LAYERS
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
        Forward pass through the entire model
        
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
        
        # 2. FIRST RNN LAYER
        # Process through first custom RNN layer
        # Shape: (batch_size, sequence_length, embedding_dim) -> (batch_size, sequence_length, rnn1_units)
        rnn1_out = self.rnn1_layer(embedded, training=training)
        rnn1_out = self.dropout1(rnn1_out, training=training)
        
        # 3. SECOND RNN LAYER  
        # Process through second custom RNN layer
        # Shape: (batch_size, sequence_length, rnn1_units) -> (batch_size, rnn2_units)
        rnn2_out = self.rnn2_layer(rnn1_out, training=training)
        rnn2_out = self.dropout2(rnn2_out, training=training)
        
        # 4. DENSE LAYERS
        # Hidden dense layer with ReLU activation
        # Shape: (batch_size, rnn2_units) -> (batch_size, dense_units)
        dense_out = self.dense1(rnn2_out)
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
            'rnn1_units': self.rnn1_units,
            'rnn2_units': self.rnn2_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate
        }
    
    def build_graph(self, input_shape):
        """Build model graph for summary"""
        self.build(input_shape)
        dummy_input = tf.keras.Input(shape=input_shape[1:], dtype=tf.int32)
        return tf.keras.Model(inputs=dummy_input, outputs=self.call(dummy_input))

def create_and_train_custom_deep_rnn(data, epochs=10, learning_rate=0.001, batch_size=32):
    """
    Create and train the custom Deep RNN model
    
    Args:
        data: Data dictionary from gensim pipeline
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        history: Training history
    """
    
    print(" CREATING CUSTOM DEEP RNN WITH TENSORFLOW")
    print("=" * 60)
    
    # Create model instance
    model = DeepRNNModel(
        vocab_size=data['vocab_size'],
        embedding_dim=data['embedding_dim'],
        embedding_matrix=data['embedding_matrix'],
        rnn1_units=64,
        rnn2_units=32,
        dense_units=32,
        dropout_rate=0.3
    )
    
    # Compile model with optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Build model by calling it once with proper input shape
    # Determine sequence length from data
    sequence_length = data['X_train'].shape[1]
    dummy_input = tf.zeros((1, sequence_length), dtype=tf.int32)
    _ = model(dummy_input)
    
    # Print model architecture
    print("\n MODEL ARCHITECTURE:")
    print(f"Input: (batch_size, {sequence_length}) - Token sequences")
    print(f"Embedding: {data['vocab_size']} -> {data['embedding_dim']} (Word2Vec frozen)")
    print(f"Simple RNN 1: {data['embedding_dim']} -> 64 (Custom implementation)")
    print(f"Simple RNN 2: 64 -> 32 (Custom implementation)")
    print(f"Dense: 32 -> 32 (ReLU)")
    print(f"Output: 32 -> 1 (Sigmoid)")
    
    # Count parameters (now the model is built)
    try:
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
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
    
    print(f"\n TRAINING ON {len(data['X_train'])} SAMPLES")
    print(f"Validation: {len(data['X_val'])} samples")
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
    
    # Now count parameters after training
    try:
        total_params = model.count_params()
        print(f"\n Model successfully trained!")
        print(f"Total parameters: {total_params:,}")
    except:
        print("\n Model successfully trained!")
    
    return model, history

def analyze_rnn_internals(model, sample_input):
    """
    Analyze what happens inside the custom RNN layers
    """
    print("\n ANALYZING RNN MODEL INTERNALS")
    print("=" * 40)
    
    # Get intermediate outputs
    sample_batch = tf.expand_dims(sample_input, 0)  # Add batch dimension
    
    # Embedding output
    embedded = model.embedding(sample_batch)
    print(f"Embedding output shape: {embedded.shape}")
    print(f"Sample embedding (first 5 dims): {embedded[0, 0, :5].numpy()}")
    
    # RNN1 output
    rnn1_out = model.rnn1_layer(embedded)
    print(f"RNN1 output shape: {rnn1_out.shape}")
    print(f"RNN1 output range: [{tf.reduce_min(rnn1_out):.4f}, {tf.reduce_max(rnn1_out):.4f}]")
    
    # RNN2 output  
    rnn2_out = model.rnn2_layer(rnn1_out)
    print(f"RNN2 output shape: {rnn2_out.shape}")
    print(f"RNN2 output range: [{tf.reduce_min(rnn2_out):.4f}, {tf.reduce_max(rnn2_out):.4f}]")
    
    # Final prediction
    prediction = model(sample_batch)
    print(f"Final prediction: {prediction[0, 0].numpy():.4f}")

def plot_rnn_training_history(history):
    """Plot training curves for custom RNN model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Custom Deep RNN - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Acc', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='red')
    ax2.set_title('Custom Deep RNN - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_custom_rnn_model(model, data):
    """Evaluate the custom RNN model on test set"""
    print("\n EVALUATING CUSTOM RNN MODEL")
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
def main_custom_deep_rnn(data):
    """
    Main function to create, train, and evaluate custom deep RNN
    """
    
    # Create and train model
    model, history = create_and_train_custom_deep_rnn(
        data, 
        epochs=10, 
        learning_rate=0.001, 
        batch_size=32
    )
    
    # Plot training history
    plot_rnn_training_history(history)
    
    # Analyze model internals
    sample_input = data['X_val'][0]
    analyze_rnn_internals(model, sample_input)
    
    # Evaluate model
    test_accuracy = evaluate_custom_rnn_model(model, data)
    
    print(f"\n CUSTOM DEEP RNN TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    return model, history
