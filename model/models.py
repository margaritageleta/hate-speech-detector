import gc
import numba
import warnings
from torch import nn
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from skorch import NeuralNetClassifier
warnings.filterwarnings("ignore")


class DistilBert(nn.Module):
    def __init__(self, bsize = 256):
        super(DistilBert, self).__init__()
        self.bsize = bsize
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(os.environ.get("MODEL_PATH"),'distilbert/vocab.txt'), lowercase=False)
        self.transformer = DistilBertModel.from_pretrained(os.path.join(os.environ.get("MODEL_PATH"),'distilbert/model')).cuda()

    # @numba.jit(forceobj=True)
    def encode(self, x, maxlen = 512):
        all_ids = np.empty((0, maxlen), np.int_)
        print(f'Tokenizing...')
        encs = distil.tokenizer.batch_encode_plus(x.squeeze(), add_special_tokens=True)['input_ids']
        # Extract IDS and add padding
        padded_encs = [enc[:maxlen] + [self.tokenizer.pad_token_id] * (maxlen - len(enc)) for enc in encs]
        all_ids = np.vstack((all_ids, np.array(padded_encs)))
        return torch.LongTensor(all_ids)
            
    def forward(self, x):
        x = self.encode(x)
        o = []
        print('Transforming...')
        for i in tqdm(range(0, x.shape[0], self.bsize)):
            xi = x[i:i + self.bsize,:]
            xi = xi.cuda()
            oi = self.transformer(input_ids=xi).last_hidden_state[:,0,:].detach().cpu()
            o.append(oi)
            xi = xi.cpu()
            del xi, oi
            gc.collect()
            torch.cuda.empty_cache()
        o = torch.cat(o, axis=0)
        # Return only [CLS] transformed token
        return o.detach().cpu()