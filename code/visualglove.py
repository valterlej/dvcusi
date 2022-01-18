'''
Implementation adapted from https://nlpython.com/implementing-glove-model-with-pytorch/
'''
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from code.clustering import predict
from tqdm import tqdm
import random
import os

class VisualGloveDataset:

    def __init__(self, tokens, window_size=25, vocab_size=1000):
        self._window_size = window_size
        self.tokens = tokens              
        self._word2id = {w:i for i, w in enumerate(range(0,vocab_size))}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self._ids_tokens = []
        for sent_tokens in self.tokens:
                self._ids_tokens.append([self._word2id[w] for w in sent_tokens])
        self._create_coocurrence_matrix()       

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)        
        
        for _id_tokens in self._ids_tokens:        
            for i, w in enumerate(_id_tokens):
                start_i = max(i - self._window_size, 0)
                end_i = min(i + self._window_size + 1, len(_id_tokens))
                for j in range(start_i, end_i):
                    if i != j:
                        c = _id_tokens[j]
                        cooc_mat[w][c] += 1 / max(abs(j-1), 1)                
        
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        # create indexes and x value tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)
        
        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()
    
    def get_batches(self, batch_size):
        # generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


def predicting_video_clusters(cluster, data, split_rate=0.5):
    """Predict video clusters and store in individual lists
    """
    all_tokens = []
    for d in tqdm(data):
        preds = predict(d, cluster, None, False, True)
        all_tokens.append(preds)    
    random.shuffle(all_tokens)
    train = all_tokens[:int(len(all_tokens)*split_rate)]
    val = all_tokens[int(len(all_tokens)*split_rate):]

    return train, val


class VisualGloveModel(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim):
        super(VisualGloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
    
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x


class VisualGloveEmbedder():
    def __init__(self, glove_model):
        self.model = glove_model
        emb_i = self.model.wi.weight.cpu().data.numpy()
        emb_j = self.model.wj.weight.cpu().data.numpy()
        self.emb = emb_i + emb_j

    """x is a list of tokens"""
    def get_embedding(self, x):
        return self.emb[x]
    
    def get_embedding_from_video_stack(self, stack, cluster):        
        predictions = list(cluster.predict(stack))
        return self.emb[predictions]

def training(cluster, data, n_epochs=1500, batch_size=2048, 
             window_size=25, x_max=20, alpha=0.75, embed_dim=128, 
             max_epochs_lower=80, model_path="./data/visualglove.pt", 
             plot_loss=False, plot_vocabulary=False, vocabulary_size=1000):       

    train, _ = predicting_video_clusters(cluster, data, split_rate=1.0)

    dataset = VisualGloveDataset(train, window_size=window_size, vocab_size=vocabulary_size)
    glove = VisualGloveModel(dataset._vocab_len, embed_dim)
    glove.cuda()

    def weight_func(x, x_max, alpha):
        wx = (x/x_max)**alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx.cuda()

    def wmse_loss(weights, inputs, targets):
        loss = weights * F.mse_loss(inputs, targets, reduction='none')
        return torch.mean(loss).cuda()


    optimizer = optim.Adagrad(glove.parameters(), lr=0.05)

    n_batches = int(len(dataset._xij) / batch_size)
    loss_values = list()
    epochs_loss_values = list()

    lower_loss = 1
    epochs_lower = 0    

    for e in range(1, n_epochs+1):
        batch_i = 0

        for x_ij, i_idx, j_idx in dataset.get_batches(batch_size):

            batch_i += 1

            optimizer.zero_grad()

            outputs = glove(i_idx, j_idx)
            weights_x = weight_func(x_ij, x_max, alpha)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))

            loss.backward()

            optimizer.step()

            loss_values.append(loss.item())

            if batch_i % 100 == 0:
                print(f"Epoch: {e}/{n_epochs} \t Batch: {batch_i}/{n_batches} \t Loss: {np.mean(loss_values[-100:])}")

        epochs_loss_values.append(loss_values[-1])
        if lower_loss > epochs_loss_values[-1]:
            lower_loss = epochs_loss_values[-1]
            epochs_lower = 0
        else:
            if epochs_lower == 0:
                print(f"saving model in epoch {e}")
                torch.save(glove.state_dict(), model_path)            
            epochs_lower += 1
            if epochs_lower == max_epochs_lower:
                break            
    
    for e, l in enumerate(epochs_loss_values):
        if e % 20 == 0:
            print(f"Epoch: {e}\tLoss: {l}")

    print(f"Lower loss: {lower_loss}")
    return glove


def predict_visual_embedding(glove_model, cluster, stacks_rgb, files_rgb, stacks_flow, files_flow, output_emb_dir, emb_file_extension, output_concat_dir, concat_file_extension, use_flow=True):
    
    if not os.path.isdir(output_emb_dir):
        os.makedirs(output_emb_dir, exist_ok=True)
    if not os.path.isdir(output_concat_dir):
        os.makedirs(output_concat_dir, exist_ok=True)

    embedder = VisualGloveEmbedder(glove_model)

    for i, stack_rgb in enumerate(tqdm(stacks_rgb)):
        try:
            file_rgb = files_rgb[i]
            if use_flow:
                stack_flow = stacks_flow[i]
                file_flow = files_flow[i]
            
            x = embedder.get_embedding_from_video_stack(stack_rgb, cluster)

            x_ = x / 2
            
            if not use_flow:
                emb_rgb = np.concatenate((stack_rgb, x), axis=1)
            else:
                emb_rgb = np.concatenate((stack_rgb, x_), axis=1)
                emb_flow = np.concatenate((stack_flow, x_), axis=1) ### emb_rgb and emb_flow are summed in the BMT method

            f_emb = os.path.join(output_emb_dir, file_rgb.replace("_rgb","").replace("_flow","").replace(".npy","")+emb_file_extension)                                   
            try:
                f_concat_rgb = os.path.join(output_concat_dir, file_rgb.replace("_rgb","").replace("_flow","").replace(".npy","")+"_rgb"+concat_file_extension)                            
                with open(f_concat_rgb, 'wb') as f:            
                    np.save(f_concat_rgb, emb_rgb)
                if use_flow:
                    f_concat_flow = os.path.join(output_concat_dir, file_flow.replace("_rgb","").replace("_flow","").replace(".npy","")+"_flow"+concat_file_extension)                
                    with open(f_concat_flow, 'wb') as f:            
                        np.save(f_concat_flow, emb_flow)            
                with open(f_emb, 'wb') as f:
                    np.save(f, x)
            except:
                print(f"{f} was ignored due to problems with files")
        except Exception as e:
            print(e)
            continue