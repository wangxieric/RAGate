import math
import torch
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Iterable
from data_processing_ketod import ketod_data_processing
from tqdm import tqdm

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # dimension of each head

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # output layer
        self.W_o = nn.Linear(embed_dim, embed_dim)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        b = query.shape[0] # batch size
        
        q = self.W_q(query).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # (b, num_heads, seq_len, head_dim)
        k = self.W_k(key).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        dot_product_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            dot_product_score = dot_product_score.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(dot_product_score, dim=-1)
        out = torch.matmul(attention_scores, v)

        out = out.transpose(1, 2).contiguous().view(b, -1, self.embed_dim)
        out = self.W_o(out)

        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super(PositionalEncoding, self).__init__()

        self.embed_dim = embed_dim
        self.max_len = max_len

        self.pe = self.compute_positional_encoding()
        
    def forward(self, x):
        seq_len = x.shape[1]
        input_dim = x.shape[2]

        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x
    
    def compute_positional_encoding(self):
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(self.max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class TransformerEncoderCell(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderCell, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward_network = FeedForwardNetwork(embed_dim, hidden_dim, dropout)

        self.norm_attention = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # multi-head attention
        attention_output = self.multi_head_attention(x, x, x, mask)
        
        x = x + self.dropout(attention_output)
        x = self.norm_attention(x)

        # feed forward network
        ffn_output = self.feed_forward_network(x)
        y = x + self.dropout(ffn_output)
        y = self.norm_ffn(y)

        return y
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(embed_dim)

        self.encoder_cells = nn.ModuleList([TransformerEncoderCell(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for encoder_cell in self.encoder_cells:
            x = encoder_cell(x, mask)
        x = self.norm(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, hidden_dim, num_classes, dropout=0.1, pad_token:int=0):
        super(TransformerClassifier, self).__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, mask):
        embedded = self.embedding(text) * math.sqrt(self.embed_dim)
        position_encoded = self.positional_encoding(embedded)
        transformer_output = self.encoder(position_encoded, mask)
        
        # Average Pool
        x = torch.mean(transformer_output, dim=1)
        logits = self.fc(x)
        logits = F.log_softmax(logits, dim=1)
        return logits
    

class TextClassificationDataset(Dataset):
    def __init__(self, data_dir):
        self.data = load_dataset('csv', data_files=data_dir, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]['output'], self.data[idx]['input']
        return label, text


def text_preprocessing(tokenizer, vocab, text):
    text_pipeline = lambda x: vocab(tokenizer(x))
    processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
    return processed_text

def yield_tokens(data_iter: Iterable, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

if __name__ == "__main__":
    # Initiate vocabulary
    tokenizer = get_tokenizer("basic_english")
    dataset_name = "../../data/lm_finetune_data/cxt-only_train.csv"
    train_iter = TextClassificationDataset(dataset_name)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(vocab_size=len(vocab), num_layers=5, embed_dim=64, num_heads=4, hidden_dim=64, num_classes=2)
    model_path = "../../outputs/MHA-system_context/model_e_50_lr_0.001_b_32_heads_4_layers_5_emb_64.pt"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    test_examples = ['hello', 'Hi']
    test_examples = torch.stack([text_preprocessing(tokenizer, vocab, text) for text in test_examples], dim=0).to(device)

    predictions = model(test_examples, mask=None)
    print(predictions.argmax(dim=1))

    # Load the dataset
    root_dir = "root dir/"
    output_dir = "output dir/"
    dataset_dir = root_dir + "test_data_with_snippets.json"
    ketod_data = ketod_data_processing(dataset_dir)
    dialogue_ids, turn_idx, contexts, contexts_and_system_responses, _, _ = ketod_data.process_data()
    predictions = {}
    for idx in tqdm(range(len(contexts))):
        input = contexts[idx]
        prediction_idx = dialogue_ids[idx] + '_' + turn_idx[idx]
        prediction = model(torch.stack([text_preprocessing(tokenizer, vocab, input)], dim=0).to(device), mask=None).argmax(dim=1).item()
        predictions[prediction_idx] = prediction
        # break
    ketod_data.add_prediction(predictions)
    ketod_data.save_data(output_dir + "test_data_with_snippets_enrich_mha_pred.json")