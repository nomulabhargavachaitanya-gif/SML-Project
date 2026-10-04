# I'm using the HuggingFace datasets library to get the AG News data as mentioned.
# I chose to limit it to 8000 samples as given in the project description.
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def get_balanced_ag_news(sample_size=8000):
    # Loading the 'train' split first to sample from it
    dataset = load_dataset("ag_news", split="train")
    df = pd.DataFrame(dataset)
    
    # The project needs exactly 8000 samples, so I'm taking 2000 from each of the 4 classes.
    # This keeps the classes balanced so the model doesn't get biased.
    balanced_df = df.groupby('label').apply(lambda x: x.sample(2000, random_state=42)).reset_index(drop=True)
    
    # I'm doing a 70/15/15 split as requested. 
    # Stratify is important here to keep the class balance in all sets.
    train_df, temp_df = train_test_split(balanced_df, test_size=0.30, stratify=balanced_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
    
    return train_df, val_df, test_df