from datasets import load_dataset

def load_data():
    dataset = load_dataset("amazon_polarity")
    dataset['train'].train_test_split(test_size=0.1).save_to_disk("data/amazon_polarity")

if __name__ == "__main__":
    load_data()
