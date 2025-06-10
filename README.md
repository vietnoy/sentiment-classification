# Sentiment Classification - Deep Learning from Scratch

## 🎯 Project Overview

This project demonstrates a comprehensive understanding of deep learning by implementing multiple neural network architectures **completely from scratch** for sentiment classification. Using Amazon product reviews, I've built and compared various models to classify text sentiment as Positive (1) or Negative (0).

**Key Achievement:** All neural network implementations are built from the ground up using TensorFlow operations, showing deep understanding of the mathematical foundations behind deep learning.

## 📊 Dataset

- **Source:** Amazon Product Reviews (amazon_polarity dataset)
- **Size:** 10,000+ training samples, 2,000+ validation samples
- **Labels:** Binary classification (0 = Negative, 1 = Positive)
- **Preprocessing:** Text cleaning, tokenization, and sequence padding

## 🧠 Implemented Architectures

### 1. Simple Neural Network (`Simple neural network/`)
- **Implementation:** Custom feedforward neural network with configurable layers
- **Features:** 
  - Multiple hidden layers with ReLU activation
  - Dropout regularization
  - L2 weight regularization
  - Custom forward/backward propagation

### 2. Recurrent Neural Network (`RNN/`)
- **Implementation:** Custom Simple RNN cells with manual state management
- **Architecture:** 
  - 2-layer deep RNN (64 → 32 units)
  - Custom Word2Vec embeddings (frozen)
  - Dropout between layers
- **Key Features:**
  - Manual implementation of RNN cell mechanics
  - Custom sequence processing with `tf.while_loop`
  - Tanh activation for hidden states

### 3. Long Short-Term Memory (`LSTM/`)
- **Implementation:** Custom LSTM cells with all 4 gates implemented manually
- **Architecture:**
  - 2-layer deep LSTM (64 → 32 units)
  - Forget, input, candidate, and output gates
  - Cell state and hidden state management
- **Mathematical Implementation:**
  ```
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # Cell state
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
  h_t = o_t ⊙ tanh(C_t)                  # Hidden state
  ```

### 4. Bidirectional RNN (`BRNNs/`)
- **Implementation:** Custom bidirectional LSTM processing sequences in both directions
- **Architecture:**
  - Forward and backward LSTM cells
  - 2-layer bidirectional processing
  - Multiple merge modes: concatenation, sum, multiply, average
- **Key Innovation:**
  - Separate forward (past→future) and backward (future→past) processing
  - Combined representations for richer context understanding

### 5. Custom Word2Vec (`Word2Vec.py`)
- **Implementation:** Skip-gram model with negative sampling
- **Features:**
  - Custom vocabulary building with frequency filtering
  - Negative sampling for efficient training
  - Cosine similarity for word relationships
  - Save/load functionality

## 🏗️ Project Structure

```
sentiment-classification/
├── BRNNs/
│   ├── src/
│   │   ├── BRNN.py          # Bidirectional RNN implementation
│   │   └── Word2Vec.py      # Word2Vec embeddings
│   └── notebook.ipynb       # BRNN experiments
├── LSTM/
│   ├── src/
│   │   ├── LSTM.py          # Custom LSTM implementation
│   │   └── Word2Vec.py      # Word2Vec embeddings
│   └── notebook.ipynb       # LSTM experiments
├── RNN/
│   ├── src/
│   │   ├── RNN.py           # Simple RNN implementation
│   │   └── Word2Vec.py      # Word2Vec embeddings
│   └── notebook.ipynb       # RNN experiments
├── Simple neural network/
│   ├── src/
│   │   ├── functions.py     # Utility functions
│   │   └── neural_network.py # Feedforward network
│   └── notebook.ipynb       # Neural network experiments
├── data/
│   ├── data_preprocessing.py # Data cleaning and tokenization
│   └── download_dataset.py  # Dataset download script
└── README.md
```

## 🚀 Key Technical Implementations

### Custom LSTM Cell Features:
- **4 Gates Implementation:** Forget, input, candidate, and output gates
- **State Management:** Proper handling of hidden and cell states
- **Gradient Flow:** Designed to handle vanishing gradient problem
- **Xavier Initialization:** Proper weight initialization for stable training

### Bidirectional Processing:
- **Forward Pass:** Processes sequence from start to end
- **Backward Pass:** Processes sequence from end to start
- **State Combination:** Multiple merge strategies for combining directions
- **Context Awareness:** Access to both past and future context

### Word2Vec Integration:
- **Skip-gram Architecture:** Predicts context words from target words
- **Negative Sampling:** Efficient training with random negative examples
- **Embedding Freezing:** Uses pre-trained embeddings without fine-tuning
- **Vocabulary Management:** Handles out-of-vocabulary words

## 📈 Performance Metrics

| Model | Test Accuracy | Architecture | Parameters |
|-------|--------------|--------------|------------|
| Simple NN | ~85% | 3-layer feedforward | ~50K |
| RNN | ~87% | 2-layer Simple RNN | ~75K |
| LSTM | ~91% | 2-layer LSTM | ~120K |
| Bidirectional RNN | ~93% | 2-layer BiLSTM | ~200K |

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install tensorflow pandas scikit-learn datasets matplotlib tqdm
```

### Quick Start
```bash
# Download and preprocess data
python data/download_dataset.py
python data/data_preprocessing.py

# Run experiments (choose one)
cd LSTM && jupyter notebook  # For LSTM experiments
cd BRNNs && jupyter notebook # For Bidirectional RNN experiments
cd RNN && jupyter notebook   # For Simple RNN experiments
```

### Training a Model
```python
from LSTM.src.LSTM import main_custom_deep_lstm
from data.data_preprocessing import load_and_preprocess

# Load and preprocess data
train_df, val_df, test_df = load_and_preprocess()

# Train custom LSTM model
model, history = main_custom_deep_lstm(data)
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Mathematical Understanding:** Implementation of complex neural network mathematics from scratch
2. **Deep Learning Fundamentals:** Forward/backward propagation, gradient descent, regularization
3. **Sequence Modeling:** Understanding of temporal dependencies and memory mechanisms
4. **Architecture Design:** Comparison of different neural network architectures
5. **Production Skills:** Data preprocessing, model evaluation, and performance optimization

## 🔬 Technical Highlights

### Custom Implementation Details:
- **No High-level APIs:** All models built using basic TensorFlow operations
- **Manual Gradient Computation:** Understanding of automatic differentiation
- **Memory Management:** Proper handling of sequence states and gradients
- **Numerical Stability:** Careful implementation to avoid numerical issues

### Advanced Features:
- **Early Stopping:** Prevents overfitting with validation monitoring
- **Learning Rate Scheduling:** Adaptive learning rate reduction
- **Dropout Regularization:** Multiple dropout layers for generalization
- **Batch Processing:** Efficient batch-wise training and inference

## 🚀 Future Enhancements

- [ ] Transformer architecture implementation from scratch
- [ ] Attention mechanism integration
- [ ] Multi-class sentiment classification (1-5 stars)
- [ ] Real-time inference API with FastAPI
- [ ] Model deployment with Docker containerization

## 📄 References

- **Amazon Polarity Dataset:** Large-scale sentiment classification dataset
- **LSTM Paper:** Hochreiter & Schmidhuber (1997)
- **Word2Vec:** Mikolov et al. (2013)
- **Bidirectional RNNs:** Schuster & Paliwal (1997)

---

**Note:** This project emphasizes learning through implementation. All neural networks are built from mathematical foundations to demonstrate deep understanding of the underlying algorithms.

## 🤝 Connect

**Author:** Đỗ Vĩnh Khang  
**GitHub:** [vietnoy](https://github.com/vietnoy)  
**LinkedIn:** [khang-do-vinh](https://www.linkedin.com/in/khang-do-vinh/)  
**CV:** [ai-engineer CV](https://docs.google.com/document/d/1nr4qqOwHjjCIROC8ITBWo3uRcAZSwnagb5MsKSw54kU/)
