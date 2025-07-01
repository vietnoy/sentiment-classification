import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self, layer_dims, activation='relu', output_activation='sigmoid',
                 dropout_rate=0.0, l2_lambda=0.0):
        """
        layer_dims: List like [input_dim, hidden1, ..., output_dim]
        dropout_rate: e.g., 0.5 for 50% dropout
        l2_lambda: strength of L2 regularization
        """
        super(NeuralNetwork, self).__init__()
        self.num_layers = len(layer_dims) - 1
        self.dropout_rate = dropout_rate
        self.hidden_activation = tf.nn.relu if activation == 'relu' else tf.nn.tanh
        self.output_activation = tf.nn.sigmoid if output_activation == 'sigmoid' else tf.nn.softmax
        self.dense_layers = []
        self.dropout_layers = []

        for i in range(self.num_layers):
            # L2 regularization
            l2 = tf.keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None

            # Dense layer with or without regularization
            dense = tf.keras.layers.Dense(
                units=layer_dims[i+1],
                activation=None,
                kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
                kernel_regularizer=l2
            )
            self.dense_layers.append(dense)

            # Add dropout (except after final output layer)
            if i < self.num_layers - 1:
                self.dropout_layers.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

    def call(self, x, training=False):
        a = x
        for i in range(self.num_layers - 1):
            z = self.dense_layers[i](a)
            a = self.hidden_activation(z)
            if self.dropout_rate > 0:
                a = self.dropout_layers[i](a, training=training)

        # Final output layer (no dropout)
        z = self.dense_layers[-1](a)
        return self.output_activation(z)