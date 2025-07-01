# Sentiment Classification Project

A comprehensive sentiment analysis project implementing various neural network architectures on Amazon product reviews. This project compares different approaches from simple neural networks to modern transformers for binary sentiment classification.

## 📊 Dataset

- **Source**: Amazon Polarity Dataset
- **Task**: Binary sentiment classification (Positive/Negative)
- **Size**: ~3.6M training samples, ~400K test samples
- **Preprocessing**: Text cleaning, tokenization, and train/validation split

## 🏗️ Project Structure

```
sentiment-classification/
├── data/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   └── notebook.ipynb          # Data exploration
├── Simple neural network/
│   ├── notebook.ipynb          # Simple NN implementation
│   └── src/
│       ├── functions.py        # Utility functions
│       └── neural_network.py   # NN model definition
├── RNN/
│   ├── notebook.ipynb          # Vanilla RNN implementation
│   └── src/
│       ├── RNN.py             # RNN model
│       └── Word2Vec.py        # Word embeddings
├── BRNNs/
│   ├── notebook.ipynb          # Bidirectional RNN implementation
│   └── src/
│       ├── BRNN.py            # BiRNN model
│       └── Word2Vec.py        # Word embeddings
├── LSTM/
│   ├── notebook.ipynb          # LSTM implementation
│   └── src/
│       ├── LSTM.py            # LSTM model
│       └── Word2Vec.py        # Word embeddings
├── transformer/
│   ├── notebook.ipynb          # Transformer implementation
│   └── src/
│       └── Simple_transformer.py  # Transformer model
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Models Implemented

1. **Simple Neural Network**: Basic feedforward network with bag-of-words features
2. **RNN**: Vanilla Recurrent Neural Network for sequence modeling
3. **Bidirectional RNN**: Enhanced RNN processing sequences in both directions
4. **LSTM**: Long Short-Term Memory networks for better long-range dependencies
5. **Transformer**: Attention-based model for state-of-the-art performance

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/vietnoy/sentiment-classification.git
cd sentiment-classification
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Usage

### Data Preprocessing
```python
from data.data_preprocessing import load_and_preprocess

train_df, val_df, test_df = load_and_preprocess(
    split_ratio=0.1,
    train_sample_size=10000,
    val_sample_size=2000
)
```

### Running Models
Each model has its own Jupyter notebook in the respective folder:
- Open the desired notebook (e.g., `RNN/notebook.ipynb`)
- Run all cells to train and evaluate the model
- Compare results across different architectures

## 🎯 Features

- **Modular Design**: Each model in separate folder with clean separation
- **Consistent Interface**: Similar preprocessing and evaluation across models
- **Word Embeddings**: Word2Vec implementation for semantic representations
- **Performance Comparison**: Easy comparison between different architectures
- **Extensible**: Easy to add new models and techniques

## 📈 Expected Results

Models are evaluated on accuracy and loss metrics. Generally expected performance order:
1. Transformer (highest accuracy)
2. LSTM
3. Bidirectional RNN
4. Vanilla RNN
5. Simple Neural Network (baseline)

## 🔧 Requirements

- Python 3.7+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- datasets
- matplotlib
- seaborn

## 📚 Learning Objectives

This project demonstrates:
- Evolution of NLP architectures
- Sequence modeling techniques
- Attention mechanisms
- Text preprocessing pipelines
- Model comparison methodologies

## 🤝 Contributing

Feel free to contribute by:
- Adding new model architectures
- Improving preprocessing techniques
- Enhancing evaluation metrics
- Adding visualization tools

## 📄 License

This project is open source and available under the MIT License.