import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import random
from tqdm.auto import tqdm
from data_util import TrafficDataset, BatchSampler
from model import MultitaskSequenceModel


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, training_data, optimizer, criterion, validation_data=None, batch_size=64, epochs=20, verbose=True, device=None):
    
    if device is None:
        device = get_device()
    
    model.to(device)
    S = list(range(len(training_data)))
    n_batch = math.ceil(len(training_data)/batch_size)
    loss_history = dict(train=[], val=[])
    torch.manual_seed(4)
    random.seed(4)
    for epoch in tqdm(range(epochs)):
        alpha = 2 - ((epoch+1)/epochs)**2; beta = ((epoch+1)/epochs)**2
        random.shuffle(S)
        batches = [S[i*batch_size:min(i*batch_size+batch_size, len(S))] for i in range(n_batch)]
        training_loader = DataLoader(training_data, batch_sampler=BatchSampler(batches))
        train_losses = []
        model.train()
        for i, (seq, seq_mask, y, y_mask) in enumerate(training_loader):
            seq, seq_mask, y, y_mask = seq.to(device), seq_mask.to(device), y.to(device), y_mask.to(device)
            seq_recon, yhat = model(seq)
            loss_recon = criterion(alpha*(~seq_mask)*seq_recon, alpha*(~seq_mask)*seq) + criterion(beta*seq_mask*seq_recon, beta*seq_mask*seq)
            loss_pred = alpha * criterion(alpha*(~y_mask)*yhat, alpha*(~y_mask)*y)
            loss = loss_recon + loss_pred
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_data.seqs[batches[i]] = (~training_data.seq_masks[batches[i]] * training_data.seqs[batches[i]] + 
                                              training_data.seq_masks[batches[i]] * .5 * (training_data.seqs[batches[i]] + seq_recon.detach()))
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        loss_history['train'].append(train_loss)

        if validation_data:
            validation_loader = DataLoader(validation_data, batch_size=batch_size)
            val_losses = []
            model = model.eval()
            with torch.inference_mode():
                for seq, seq_mask, y, y_mask in validation_loader:
                    seq, seq_mask, y, y_mask = seq.to(device), seq_mask.to(device), y.to(device), y_mask.to(device)
                    seq_recon, yhat = model(seq)
                    loss = criterion(seq_recon, seq) + criterion(yhat, y)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            loss_history['val'].append(val_loss)

        if verbose:
            print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

    return loss_history


root_dir = '../data/training/'
csv_files = sorted([fn for fn in os.listdir(root_dir) if os.path.splitext(fn)[1] == '.csv'])
seq_len = 24; y_len = 1; feature_size = 46
training_data = TrafficDataset(root_dir, csv_files, seq_len, y_len, feature_size, mode='train')
validation_data = TrafficDataset(root_dir, csv_files, seq_len, y_len, feature_size, mode='val')

torch.manual_seed(1)
model = MultitaskSequenceModel(feature_size=feature_size, embedding_size=64, num_layers=2, dropout=0.2)

optimizer = optim.Adam(model.parameters(), lr=.01)
criterion = nn.MSELoss(reduction='sum')

loss_history = train(model, training_data, optimizer, criterion, validation_data, batch_size=64, epochs=20)

