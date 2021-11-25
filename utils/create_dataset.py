import os
import re
import glob
import numpy as np
import pandas as pd

def clean(text):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text)
    text = re.sub(r'#[\S]+\b', '', text)
    text = re.sub(r'@[\S]+\b', '', text)
    text = re.sub(r'https?\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def join_sources(split='train'):
    datasets = []
    for i, source in enumerate(glob.glob(f'{os.environ.get("IN_PATH")}/*')):
        data = pd.read_csv(f'{source}/{split}.csv')
        num_source = int(source[-1])
        print(f'Processing source #{num_source}: {source}')
        if num_source == 1:
            data['toxicity'] = data.toxic + data.severe_toxic + \
                                data.obscene + data.threat + \
                                data.insult + data.identity_hate
            data['target'] = (data.toxicity > 0).astype(int)
        elif num_source == 2:
            data['target'] = (data.target >= 0.5).astype(int)
            data['comment_text'] = data.comment_text.map(clean)
            data = data[['comment_text', 'target']]
            datasets.append(data)
    return pd.concat(datasets).drop_duplicates()

def get_toxicity_stats(df):
    positive_samples = len(df.query('target==1'))
    negative_samples = len(df.query('target==0'))
    total_samples = positive_samples + negative_samples
    
    print(f'Number of positive samples (toxic): {positive_samples} ({np.round(positive_samples/total_samples * 100, 2)}%)')
    print(f'Number of negative samples (non-toxic): {negative_samples} ({np.round(negative_samples/total_samples * 100, 2)}%)')
    
    return positive_samples, negative_samples

def balance_classes(df, bias=0):
    print(f'BEFORE BALANCING')
    positive_samples, negative_samples = get_toxicity_stats(df)
    
    dff = pd.concat([
        df.query('target==1'),
        df.query('target==0').sample(n = int(positive_samples * (1+bias)), random_state = 123)
    ]).sample(frac=1).reset_index(drop=True)
    
    print(f'\nAFTER BALANCING')
    positive_samples, negative_samples = get_toxicity_stats(dff)
    print(f'\nTOTAL SAMPLES: {positive_samples + negative_samples}')
    return dff
    
def preprocess_driver(split='train', bias=0.5):
    opath = f'{os.environ.get("IN_PATH")}/preprocessed_{split}.csv'
    df = join_sources(split=split)
    df = balance_classes(df, bias=bias)
    df.to_csv(opath, index=False)
    
if  __name__ ==  '__main__':
    preprocess_driver(split='train', bias=0.5)
    
    