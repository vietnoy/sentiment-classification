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
    dataset = load_dataset("amazon_polarity", split="train")
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"content": "text", "label": "label"})
    df["text"] = df["text"].apply(clean_text)
    
    # Split train/val/test
    train_df, val_df = train_test_split(df, test_size=split_ratio, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df["label"], random_state=42)

    # Downsample to safe sizes
    train_df = train_df.sample(n=train_sample_size, random_state=42)
    val_df = val_df.sample(n=val_sample_size, random_state=42)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    train_df, val_df, test_df = load_and_preprocess()
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))
