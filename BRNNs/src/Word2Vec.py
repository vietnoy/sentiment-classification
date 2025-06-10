import tensorflow as tf
import numpy as np
from collections import Counter
import random
from tqdm import tqdm
import os
import re

class TFWord2Vec:
    def __init__(self, embedding_dim=100, window_size=2, num_ns=5, vocab_size=10000, min_count=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_ns = num_ns
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.word_to_id = {}
        self.id_to_word = {}
        self.model = None
    
    def preprocess_text(self, text):
        """Clean and tokenize text"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    
    def build_vocab(self, sentences):
        """Build vocabulary from sentences"""
        # Count word frequencies
        all_words = []
        for sentence in sentences:
            if isinstance(sentence, str):
                # Process string sentences
                words = self.preprocess_text(sentence)
                all_words.extend(words)
            else:
                # Assume already tokenized
                all_words.extend(sentence)
        
        word_counts = Counter(all_words)
        
        # Filter by minimum count
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_count}
        
        # Sort by frequency
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        vocab = [word for word, _ in sorted_words[:self.vocab_size-1]]
        
        # Create mappings (reserve 0 for OOV)
        self.word_to_id = {w: i+1 for i, w in enumerate(vocab)}
        self.id_to_word = {i+1: w for i, w in enumerate(vocab)}
        self.id_to_word[0] = "<OOV>"
        
        print(f"Vocabulary built with {len(self.word_to_id)} words")
        return self.word_to_id
    
    def generate_skipgram_batch(self, sentences, batch_size=4096):
        """Generate skipgram training data in batches"""
        targets, contexts, labels = [], [], []
        
        # Process each sentence
        for sentence in tqdm(sentences, desc="Generating training data"):
            # Tokenize if needed
            if isinstance(sentence, str):
                words = self.preprocess_text(sentence)
            else:
                words = sentence
            
            # Convert to word IDs
            word_ids = [self.word_to_id.get(word, 0) for word in words 
                       if word in self.word_to_id or self.word_to_id.get(word, 0) > 0]
            
            # Generate skipgram pairs
            for i, target in enumerate(word_ids):
                # Define context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(word_ids), i + self.window_size + 1)
                
                # Get context words
                context_words = word_ids[window_start:i] + word_ids[i+1:window_end]
                
                for context in context_words:
                    # Add positive example
                    targets.append(target)
                    contexts.append(context)
                    labels.append(1)
                    
                    # Add negative samples
                    for _ in range(self.num_ns):
                        # Sample random word from vocabulary (excluding 0 - OOV)
                        neg = random.randint(1, len(self.word_to_id))
                        targets.append(target)
                        contexts.append(neg)
                        labels.append(0)
                
                # Yield batch when it reaches the desired size
                if len(targets) >= batch_size:
                    yield (
                        [np.array(targets).reshape(-1, 1), np.array(contexts).reshape(-1, 1)],
                        np.array(labels)
                    )
                    targets, contexts, labels = [], [], []
        
        # Yield any remaining examples
        if targets:
            yield (
                [np.array(targets).reshape(-1, 1), np.array(contexts).reshape(-1, 1)],
                np.array(labels)
            )
    
    def create_model(self):
        """Create Word2Vec model using TensorFlow Keras"""
        # Input layers
        input_target = tf.keras.layers.Input((1,))
        input_context = tf.keras.layers.Input((1,))
        
        # Embedding layer
        embedding = tf.keras.layers.Embedding(
            len(self.word_to_id) + 1,  # +1 for OOV
            self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            name="word_embedding"
        )
        
        # Get embeddings
        target_embedding = embedding(input_target)
        context_embedding = embedding(input_context)
        
        # Reshape to vectors
        target_vector = tf.keras.layers.Reshape((self.embedding_dim,))(target_embedding)
        context_vector = tf.keras.layers.Reshape((self.embedding_dim,))(context_embedding)
        
        # Dot product
        dot_product = tf.keras.layers.Dot(axes=1)([target_vector, context_vector])
        
        # Output prediction
        output = tf.keras.layers.Activation('sigmoid')(dot_product)
        
        # Create model
        self.model = tf.keras.Model(inputs=[input_target, input_context], outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, sentences, epochs=5, batch_size=4096, steps_per_epoch=None):
        """Train the Word2Vec model"""
        # Build vocabulary if not already done
        if not self.word_to_id:
            self.build_vocab(sentences)
        
        # Create model if not already done
        if not self.model:
            self.create_model()
        
        # For each epoch
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Track metrics
            epoch_loss = 0
            epoch_accuracy = 0
            batch_count = 0
            
            # Generate and train on batches
            for inputs, labels in self.generate_skipgram_batch(sentences, batch_size):
                # Train on batch
                history = self.model.train_on_batch(inputs, labels)
                
                # Update metrics
                epoch_loss += history[0]
                epoch_accuracy += history[1]
                batch_count += 1
                
                # Stop if steps_per_epoch is reached
                if steps_per_epoch and batch_count >= steps_per_epoch:
                    break
            
            # Print epoch summary
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            avg_accuracy = epoch_accuracy / batch_count if batch_count > 0 else 0
            print(f"  Epoch {epoch+1} summary - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}")
    
    def get_embeddings(self):
        """Get the trained word embeddings"""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        # Get weights from the embedding layer
        return self.model.get_layer("word_embedding").get_weights()[0]
    
    def get_vector(self, word):
        """Get embedding vector for a specific word"""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        # Check if word is in vocabulary
        if word not in self.word_to_id:
            print(f"Warning: '{word}' not in vocabulary")
            return np.zeros(self.embedding_dim)
        
        # Get word embedding
        word_id = self.word_to_id[word]
        embeddings = self.get_embeddings()
        return embeddings[word_id]
    
    def most_similar(self, word, top_n=10):
        """Find most similar words based on cosine similarity"""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        # Check if word is in vocabulary
        if word not in self.word_to_id:
            return f"Word '{word}' not in vocabulary"
        
        # Get word vector
        word_id = self.word_to_id[word]
        word_vector = self.get_embeddings()[word_id]
        
        # Compute similarities with all words
        embeddings = self.get_embeddings()
        
        # Normalize vectors for cosine similarity
        word_norm = np.linalg.norm(word_vector)
        all_norms = np.linalg.norm(embeddings, axis=1)
        
        # Avoid division by zero
        word_norm = max(word_norm, 1e-10)
        all_norms = np.maximum(all_norms, 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings, word_vector) / (all_norms * word_norm)
        
        # Find top similar words
        top_indices = np.argsort(similarities)[::-1][:top_n+1]  # +1 because the word itself will be included
        
        # Filter out the word itself
        results = []
        for idx in top_indices:
            if idx != word_id and idx in self.id_to_word:
                results.append((self.id_to_word[idx], float(similarities[idx])))
                if len(results) >= top_n:
                    break
        
        return results
    
    def save(self, filepath):
        """Save model and vocabulary"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model weights
        self.model.save_weights(filepath + "_weights.h5")
        
        # Save vocabulary and parameters
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'num_ns': self.num_ns,
            'vocab_size': self.vocab_size,
            'min_count': self.min_count
        }
        
        np.save(filepath + "_vocab.npy", vocab_data, allow_pickle=True)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model"""
        # Load vocabulary and parameters
        vocab_data = np.load(filepath + "_vocab.npy", allow_pickle=True).item()
        
        # Create model instance
        model = cls(
            embedding_dim=vocab_data['embedding_dim'],
            window_size=vocab_data['window_size'],
            num_ns=vocab_data['num_ns'],
            vocab_size=vocab_data['vocab_size'],
            min_count=vocab_data['min_count']
        )
        
        # Restore vocabulary
        model.word_to_id = vocab_data['word_to_id']
        model.id_to_word = vocab_data['id_to_word']
        
        # Create and restore model
        model.create_model()
        model.model.load_weights(filepath + "_weights.h5")
        
        return model

# Example usage
if __name__ == "__main__":
    # Example sentences (can be strings or already tokenized)
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "quick brown foxes jump over lazy dogs",
        "the brown dog chased the fox",
        "the fox outsmarted the hound"
    ]
    
    # Create and train model
    word2vec = TFWord2Vec(embedding_dim=100, window_size=2, num_ns=5, min_count=1)
    word2vec.train(sentences, epochs=5)
    
    # Find similar words
    similar_words = word2vec.most_similar("fox", top_n=3)
    print("Words similar to 'fox':", similar_words)
    
    # Save model
    word2vec.save("word2vec_model")
    
    # Load model
    loaded_model = TFWord2Vec.load("word2vec_model")
    
    # Verify loaded model
    similar_words = loaded_model.most_similar("fox", top_n=3)
    print("After loading - Words similar to 'fox':", similar_words)