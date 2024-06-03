import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import pandas as pd
import os


class TrafficDataset(Dataset):
    
    def __init__(self, root_dir, csv_files, seq_len, y_len, feature_size, mode='train', rho=0.25, train_size=0.8):
        self.seqs = torch.empty((0, seq_len, feature_size))
        self.targets = torch.empty((0, y_len, feature_size))
        self.rho = rho if mode == 'train' else .0
        torch.manual_seed(0)
        for idx, csv_file in enumerate(csv_files):
            df = pd.read_csv(os.path.join(root_dir, csv_file), dtype='float32')
            hourlyPace_comp = torch.tensor(df.values)
            hourlyPace_incomp = F.dropout(hourlyPace_comp, p=self.rho) * (1 - self.rho)
            n = hourlyPace_comp.shape[0] - seq_len - y_len + 1
            for i in range(n):
                self.seqs = torch.cat((self.seqs, hourlyPace_incomp[i:i+seq_len].unsqueeze(0)), dim=0)
                self.targets = torch.cat((self.targets, hourlyPace_incomp[i+seq_len:i+seq_len+y_len].unsqueeze(0)), dim=0)
        self.seq_masks = (self.seqs==0)
        mean = (self.seqs.sum(dim=2)/(~self.seq_masks).sum(dim=2)).unsqueeze(2).repeat(1, 1, feature_size)
        self.seqs += self.seq_masks * mean
        self.seq_len = seq_len
        N = int(train_size * len(self.seqs))
        if mode == 'train':
            self.seqs = self.seqs[:N]
            self.seq_masks = self.seq_masks[:N]
            self.targets = self.targets[:N]
            self.target_masks = (self.targets==0)
        elif mode == 'val':
            self.seqs = self.seqs[N:]
            self.seq_masks = self.seq_masks[N:]
            self.targets = self.targets[N:]
            self.target_masks = (self.targets==0)
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        seq_mask = self.seq_masks[idx]
        y = self.targets[idx]
        y_mask = self.target_masks[idx]
        return seq, seq_mask, y, y_mask
    

class BatchSampler(Sampler):
    
    def __init__(self, batches):
        self.batches = batches
    
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch

