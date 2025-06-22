import re
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader
import json


text_path = r".\the_verdict.txt"
vocab_path = r".\vocab.json"
with open(text_path, "r", encoding="utf-8") as f,\
     open(vocab_path, 'w', encoding='utf-8') as f1:
    raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token:integer for integer,token in enumerate(all_words)}
    json.dump(vocab, f1)