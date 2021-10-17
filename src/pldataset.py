import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset import VbdDataset, VbdTestDataset


class VbdDataModule(LightningDataModule):
    
    def __init__(self, base_dir, siglen=None, batch_size=1, batch_size_valid=None, stft_hparams=None, pin_memory=True, num_workers=4):
        super().__init__()
        self.base_dir = base_dir
        self.siglen = siglen
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        
        if stft_hparams is None:
            self.nfft = 1024
            self.nhop = 256
        else:
            self.nfft = stft_hparams.nfft
            self.nhop = stft_hparams.nhop
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = VbdDataset(self.base_dir, mode='train')
            self.valid_set = VbdDataset(self.base_dir, mode='valid')
            
        if stage == 'validate' or stage is None:
            self.valid_set = VbdDataset(self.base_dir, mode='valid')
            
        if stage == 'test' or stage is None:
            # Testing with a "real" test set should only be run once. Here, we used the validation set.
            self.test_set = VbdDataset(self.base_dir, mode='valid')

    def train_dataloader(self):
        loader = DataLoader(self.train_set,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=self.pin_memory,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn_train
                           )
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_set,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=self.pin_memory,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn_valid
                           )
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_set,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=self.pin_memory,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn_test
                           )
        return loader
    
    def collate_fn_train(self, batch):
        siglen = self.siglen
        noisy_list = []
        clean_list = []
        for noisy, clean in batch:
            if len(clean) > siglen:
                start_idx = np.random.randint(0, len(clean)-siglen)
                noisy_list += [noisy[start_idx:start_idx+siglen]]
                clean_list += [clean[start_idx:start_idx+siglen]]

            else:
                noisy_list += [F.pad(noisy, (0, siglen-len(clean)))]
                clean_list += [F.pad(clean, (0, siglen-len(clean)))]
        
        return torch.stack(noisy_list), torch.stack(clean_list)
    
    def collate_fn_valid(self, batch):
        siglen = self.siglen
        
        noisy_list = []
        clean_list = []
        for noisy, clean in batch:
            if len(clean) > siglen:
                start_idx = 0
                noisy_list += [noisy[start_idx:start_idx+siglen]]
                clean_list += [clean[start_idx:start_idx+siglen]]

            else:
                noisy_list += [F.pad(noisy, (0, siglen-len(clean)))]
                clean_list += [F.pad(clean, (0, siglen-len(clean)))]
        
        return torch.stack(noisy_list), torch.stack(clean_list)

    def collate_fn_test(self, batch):
        # Batch size is assumed to be 1
        noisy, clean = batch[0]

        length = len(clean)
        npad = (length - self.nfft) // self.nhop * self.nhop + self.nfft - length
        noisy = F.pad(noisy, (0, npad))
        clean = F.pad(clean, (0, npad))
        return noisy[None, :], clean[None, :], length