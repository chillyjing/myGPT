import re
from importlib.metadata import version
import tiktoken
import torch
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


'''class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str2int = vocab
        self.int2str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        precessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        precessed = [item.strip() for item in precessed if item.strip()]
        precessed = [item if item in self.str2int else "<|unk|>" for item in precessed]
        ids = [self.str2int[s] for s in precessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int2str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text'''

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)

class GPTDatasetV1(Dataset):
    def __init__(self, tokenizer, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length , stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_works=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(tokenizer, txt, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)
    return dataloader

max_length=4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)
data_iter = iter(dataloader) #创建迭代器
inputs, target = next(data_iter)

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape) #8*4*256,相当于为每一个id映射到256维度，嵌入层相当于一个查找表
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings

if "__name__" == "__main__":
    print(input_embeddings)