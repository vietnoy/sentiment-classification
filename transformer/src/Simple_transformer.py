import tensorflow as tf
import numpy as np
from keras.layers import Layer, Embedding, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

class PositionalEncoding(Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model
        
    def build(self, input_shape):
        # Create positional encoding matrix
        angle_rads = self.get_angles(
            np.arange(self.max_position)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
            self.d_model
        )
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        self.pos_encoding = self.add_weight(
            name='positional_encoding',
            shape=pos_encoding.shape,
            initializer='zeros',
            trainable=False
        )
        
        # Set the values
        self.pos_encoding.assign(tf.cast(pos_encoding, dtype=tf.float32))
        
        super().build(input_shape)
        
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class TransformerEncoderBlock(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        # Multi-Head Attention
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        
        # Feed Forward Network
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        
        # Layer Normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def point_wise_feed_forward_network(self, d_model, dff):
        return Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
    
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention with residual connection and layer norm
        attn_output = self.mha(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection
        
        # Feed forward network with residual connection and layer norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2

class TransformerSentimentClassifier(tf.keras.Model):
    def __init__(self, vocab_size, max_length=512, d_model=128, num_heads=8, 
                 num_layers=4, dff=512, rate=0.1, num_classes=1):
        super(TransformerSentimentClassifier, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Token and Position Embeddings
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_length, d_model)
        
        # Transformer Encoder Blocks
        self.enc_layers = [
            TransformerEncoderBlock(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        
        self.dropout = Dropout(rate)
        
        # Global Average Pooling to aggregate sequence representations
        self.global_avg_pool = GlobalAveragePooling1D()
        
        # Dense layers for classification (as you suggested!)
        self.dense1 = Dense(256, activation='relu')
        self.dropout_dense = Dropout(rate)
        self.dense2 = Dense(128, activation='relu')
        self.dropout_dense2 = Dropout(rate)
        
        # Output layer - sigmoid for binary classification
        if num_classes == 1:
            self.output_layer = Dense(1, activation='sigmoid')
        else:
            self.output_layer = Dense(num_classes, activation='softmax')
        
        # Initialize vocabulary mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def create_padding_mask(self, seq):
        """Create padding mask for attention mechanism"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # Add extra dimensions for attention mechanism
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def call(self, inputs, training=None):
        # Create padding mask
        mask = self.create_padding_mask(inputs)
        
        # Embedding layer
        x = self.embedding(inputs)  # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scale embeddings
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pass through all transformer encoder blocks
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        
        # This aggregates information from all time steps
        pooled_output = self.global_avg_pool(x)  # (batch_size, d_model)
        
        # Classification head
        x = self.dense1(pooled_output)
        x = self.dropout_dense(x, training=training)
        x = self.dense2(x)
        x = self.dropout_dense2(x, training=training)
        
        # Final classification
        output = self.output_layer(x)
        
        return output
    
    def build_vocabulary(self, texts, vocab_size=10000, max_length=512):
        """Build vocabulary from text data"""
        
        # Initialize special tokens
        self.word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }
        
        self.idx_to_word = {
            0: "<PAD>",
            1: "<UNK>", 
            2: "<START>",
            3: "<END>"
        }
        
        # Tokenize and count words
        all_words = []
        for text in texts:
            # Simple tokenization (you might want to use more sophisticated methods)
            words = text.lower().split()
            all_words.extend(words)
        
        # Get most common words
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(vocab_size - 4)
        
        # Build vocabulary
        for idx, (word, count) in enumerate(most_common):
            self.word_to_idx[word] = idx + 4
            self.idx_to_word[idx + 4] = word
        
        print(f"Built vocabulary with {len(self.word_to_idx)} words")
        
    def encode_texts(self, texts, max_length=None):
        """Convert texts to sequences of token indices"""
        if max_length is None:
            max_length = self.max_length
            
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = []
            for word in words:
                if word in self.word_to_idx:
                    sequence.append(self.word_to_idx[word])
                else:
                    sequence.append(self.word_to_idx["<UNK>"])
            sequences.append(sequence)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=max_length,
            padding='post',
            truncating='post',
            value=self.word_to_idx["<PAD>"]
        )
        
        return padded_sequences
