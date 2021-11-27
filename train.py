import os
import gc
import sys
import glob
import yaml
import time
import torch
import numba
import wandb
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch_optimizer as torchoptim
from transformers import DistilBertTokenizer, DistilBertTokenizerFast
from transformers import DistilBertModel, DistilBertConfig

from model.models import FineTuneNet

## Seed for reproducibility.
seed = 2021 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True

@numba.jit(forceobj=True)
def batch_encode(tokenizer, texts, batch_size=256, max_length=512):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.
    
    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""
    
    input_ids = []
    attention_mask = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                                  max_length=max_length,
                                                  padding='max_length', 
                                                  truncation=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False)             
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    
    
    return torch.LongTensor(input_ids), torch.Tensor(attention_mask)

if __name__ == '__main__':
    ## Load params
    with open(os.path.join(os.environ.get("IN_PATH"), '../params.yaml'), 'r') as f:
        params = yaml.safe_load(f)

    ## Load data
    data = pd.read_csv(os.path.join(os.environ.get("IN_PATH"), 'preprocessed_train.csv'))
    data = data.dropna()
    batch_size = params['batch_size']

    N_Batches = data.comment_text.values.shape[0] // batch_size

    train_size = int(N_Batches * 0.9) * batch_size + batch_size
    validation_size = int(N_Batches * 0.05) * batch_size
    test_size = int(N_Batches * 0.05) * batch_size

    a = train_size
    b = a + validation_size
    c = b + test_size

    X_train, Y_train = data.comment_text.values[0:a], data.target.values[0:a]
    X_valid, Y_valid = data.comment_text.values[a:b], data.target.values[a:b]
    X_test, Y_test = data.comment_text.values[b:c], data.target.values[b:c]

    ## Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(os.environ.get("MODEL_PATH"),'distilbert/vocab.txt'),lowercase=False)
    
    # Encode X_train, X_valid, X_test
    X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())
    X_valid_ids, X_valid_attention = batch_encode(tokenizer, X_valid.tolist())
    X_test_ids, X_test_attention = batch_encode(tokenizer, X_test.tolist())
    
    # Load distilBert model
    distilBert = DistilBertModel.from_pretrained(os.path.join(os.environ.get("MODEL_PATH"),'distilbert/model2'))

    # Make DistilBERT layers trainable
    for param in distilBert.parameters():
        param.requires_grad = True

    model = FineTuneNet(distilBert, params)
    model = model.cuda()
    
    criterion = nn.BCELoss()
    optimizer = torchoptim.QHAdam(
        params=model.parameters(), 
        lr=params['lr'],
        betas=(0.9, 0.999),
        nus=(1.0, 1.0),
        weight_decay=params['weight_decay'],
        decouple_weight_decay=False,
        eps=1e-8,
    )
    
    experiment = params['experiment']
    epochs = params['epochs']
    num_steps = len(X_train) // batch_size
    num_steps_vd = len(X_valid) // batch_size
    
    ## Initialize wandb
    os.environ["WANDB_START_METHOD"] = "thread"
    ## Automate tag creation on run launch:
    wandb_tags = []
    wandb_tags.append(f"lr {params['lr']}")
    wandb_tags.append(f"dropout {params['dropout']}")
    wandb.init(
        project='nlptoxic',
        dir=os.environ.get('OUT_PATH'),
        tags=wandb_tags,
        # resume='allow',
    )
    wandb.run.name = f'Experiment #{experiment}'
    wandb.run.save()
    print('Wandb ready.')
    wandb.watch(model)

    best_loss = np.inf
    for epoch in tqdm(range(epochs), file=sys.stdout): 

        ## TRAINING LOOP
        model = model.train()
        running_loss = 0.0
        for k, i in enumerate(range(0, X_train_ids.shape[0], batch_size)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            input_tokens = X_train_ids[i:i+batch_size,:].long()
            input_masks = X_train_attention[i:i+batch_size,:]
            input_tokens = input_tokens.cuda()
            input_masks = input_masks.cuda()

            outputs = model(input_tokens, input_masks).cpu()
            loss = criterion(outputs, torch.Tensor(Y_train[i:i+batch_size].reshape(-1,1)))

            wandb.log({ 'BCE train': loss })
            
            if k % 10 == 0:
                print(f'{np.round(k / num_steps * 100,3)}% | TR Loss: {loss}')
            loss.backward()
            optimizer.step()

            input_tokens = input_tokens.cpu()
            input_masks = input_masks.cpu()
            outputs = outputs.detach().cpu()
            del input_tokens, input_masks, outputs
            gc.collect()
            torch.cuda.empty_cache()

            # print statistics
            running_loss += loss.item()
        tqdm.write('[%d, %5d] TR loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps)) 
        
        ## VALIDATION LOOP
        model = model.eval()
        running_loss = 0.0
        for k, i in enumerate(range(0, X_valid_ids.shape[0], batch_size)):

            input_tokens = X_valid_ids[i:i+batch_size,:].long()
            input_masks = X_valid_attention[i:i+batch_size,:]
            input_tokens = input_tokens.cuda()
            input_masks = input_masks.cuda()
            
            with torch.no_grad():
                outputs = model(input_tokens, input_masks).cpu()
                loss = criterion(outputs, torch.Tensor(Y_valid[i:i+batch_size].reshape(-1,1)))
            
            wandb.log({ 'BCE valid': loss })
            
            if k % 10 == 0:
                print(f'{np.round(k / num_steps_vd * 100,3)}% | VD Loss: {loss}')

            input_tokens = input_tokens.cpu()
            input_masks = input_masks.cpu()
            outputs = outputs.detach().cpu()
            del input_tokens, input_masks, outputs
            gc.collect()
            torch.cuda.empty_cache()

            # print statistics
            running_loss += loss.item()
        tqdm.write('[%d, %5d] VD loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps_vd)) 
        if bool(running_loss < best_loss):
            print('Storing a new best model...')
            torch.save(model.state_dict(), os.path.join(os.environ.get('OUT_PATH'), f'distil_weights_{experiment}.pt'))
            
    print('Finished Training!')
    