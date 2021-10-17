import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

class VbdDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dir, mode='train'):
        path = pathlib.Path(base_dir).joinpath(mode)
        self.npy_names = np.sort(list(path.glob('*.npy')))
       
    def __len__(self):
        return len(self.npy_names)
        
    def __getitem__(self, idx):
        with open(self.npy_names[idx], "rb") as f:
            noisy = np.load(f)
            clean = np.load(f)
        
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)
        return noisy, clean
    
class VbdTestDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dir):
        self.base_dir = pathlib.Path(base_dir)
        path = self.base_dir.joinpath('clean_testset_wav2')
        self.fnames = np.sort([fname.parts[-1] for fname in  path.glob('*.wav')])
       
    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        noisy, _ = torchaudio.load(self.base_dir.joinpath('noisy_testset_wav2', fname))
        clean, _ = torchaudio.load(self.base_dir.joinpath('clean_testset_wav2', fname))
        return noisy[0, :], clean[0, :]