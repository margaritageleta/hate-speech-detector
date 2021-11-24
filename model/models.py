import os
import torch
import numpy as np
from torch import nn
from tokenizers import BertWordPieceTokenizer
from transformers import DistilBertModel, DistilBertConfig
from skorch import NeuralNetClassifier


class DistilBert(nn.Module):
    def __init__(self):
        super(DistilBert, self).__init__()

        self.tokenizer = BertWordPieceTokenizer(os.path.join(os.environ.get("MODEL_PATH"),'distilbert/vocab.txt'), lowercase=False)
        self.transformer = DistilBertModel.from_pretrained(os.path.join(os.environ.get("MODEL_PATH"),'distilbert/model'))

    def encode(self, x, maxlen = 512):
        all_ids = []   
        for i in range(0, x.shape[0]):
            text = x[i,:].astype(str).tolist()
            # Tokenize current text chunk
            encs = self.tokenizer.encode_batch(text, add_special_tokens=True)
            # Extending the list = squeezing the list
            all_ids.extend([enc.ids for enc in encs])
            # Adding padding
            all_ids = [i + [tokenizer.token_to_id('[PAD]')] * (maxlen - len(i))
                                          for i in all_ids]
        return torch.LongTensor(all_ids)
            
    def forward(self, x):
        x = self.encode(x)
        o = self.transformer(input_ids=x)
        # Return only [CLS] transformed token
        return o.last_hidden_state[:,0,:]