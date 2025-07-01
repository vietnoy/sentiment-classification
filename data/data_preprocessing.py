import re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9()!?\'`\"]", " ", text)  # remove unwanted chars
    text = re.sub(r"\s{2,}", " ", text)  # remove extra whitespace
    return text.strip().lower()

def load_and_preprocess(split_ratio=0.1, train_sample_size=10000, val_sample_size=2000):
    # Load train and test splits separately
    train_dataset = load_dataset("amazon_polarity", split="train")
    test_dataset = load_dataset("amazon_polarity", split="test")
    
    # Convert to DataFrames and clean
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    
    train_df = train_df.rename(columns={"content": "text", "label": "label"})
    test_df = test_df.rename(columns={"content": "text", "label": "label"})
    
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)
    
    # Split train into train/val
    train_df, val_df = train_test_split(train_df, test_size=split_ratio, stratify=train_df["label"], random_state=42)

    # Sample the datasets
    train_df = train_df.sample(n=train_sample_size, random_state=42)
    val_df = val_df.sample(n=val_sample_size, random_state=42)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
