import os
import re
from pathlib import Path

from datasets import load_dataset, list_datasets, DatasetDict
import torch
from transformers import GPT2Tokenizer, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


DATA_PATH = Path(os.path.expandvars('$HOME')) / 'data'

ds = load_dataset('wikipedia', '20200501.en',  beam_runner='DirectRunner', cache_dir=str(DATA_PATH))
ds_train = ds['train']

out_path = DATA_PATH / 'wiki_20200501_en'
embs_path = out_path / 'gpt2_embs'
embs_path.mkdir(parents=True, exist_ok=True)
print(f'Output directory: {embs_path}')

max_tokens = 1024
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model: GPT2Model = GPT2Model.from_pretrained('gpt2')
for i in range(len(ds_train)):
    item = ds_train[i]
    title, text = item['title'], item['text']
    tokens = tokenizer(text, return_tensors='pt')
    n_tokens = tokens['input_ids'].shape[1]
    print(f'{i:03d}. {title[:80]}. Words: {len(text.split())}. Tokens: {n_tokens}')
    n_parts = n_tokens // max_tokens + min(n_tokens % max_tokens, 1)
    for i_part in range(n_parts):
        ts_part = tokens.copy()
        inds = slice(i_part * max_tokens, (i_part + 1) * max_tokens)
        ts_part['input_ids'] = ts_part['input_ids'][:, inds]
        ts_part['attention_mask'] = ts_part['attention_mask'][:, inds]
        out: BaseModelOutputWithPastAndCrossAttentions = model(**ts_part)
        title_part = title.lower().replace(' ', '_')[:40]
        fname = f'{i:07d}_{i_part:03d}_{title_part}.pt'
        fpath = embs_path / fname
        emb = out.last_hidden_state
        print(f'Saving tensor {list(emb.shape)} to file {fname}')
        torch.save(emb, fpath)
    if i == 20:
        break

