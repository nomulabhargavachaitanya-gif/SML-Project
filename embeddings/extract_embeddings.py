import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import pickle
import os

# Using DistilBERT because it's faster than regular BERT but still very accurate.
# I'm checking if a GPU is available because doing this on a CPU takes a long time.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(text_list, cache_path):
    # I implemented a cache check so we don't have to re-run the heavy model every time.
    if os.path.exists(cache_path):
        print("Loading embeddings from cache: " + cache_path)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval() # Making sure the model is in evaluation mode
    
    cls_embs, mean_embs = [], []
    
    batch_size = 32
    # Process texts in batches for faster extraction
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Extracting DistilBERT features"):
            batch_texts = text_list[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            
            # (a) I take the first token [CLS] as a summary of the whole sentence.
            cls_embs.append(last_hidden[:, 0, :].cpu().numpy())
            
            # (b) I calculate the average of all tokens, but I mask out the padding.
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embs.append((sum_embeddings / sum_mask).cpu().numpy())
            
    res = (np.vstack(cls_embs), np.vstack(mean_embs))
    with open(cache_path, 'wb') as f:
        pickle.dump(res, f)
    return res