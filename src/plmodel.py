import functools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorch_lightning import LightningModule

from src.model import UNet
from src.utils.scheduler import CosineWarmupScheduler
from src.metrics import psa, sisdr


class VbdLitModel(LightningModule):
    def __init__(self, model_hparams, optimizer_hparams, stft_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet()
        self.stft = functools.partial(torch.stft, n_fft=stft_hparams.nfft, hop_length=stft_hparams.nhop, return_complex=True)
        self.istft = functools.partial(torch.istft, n_fft=stft_hparams.nfft, hop_length=stft_hparams.nhop, return_complex=False)

    def forward(self, wave):
        complex_spec, normalized_magnitude = self.pre_process_on_gpu(wave)
        mask = self.model(normalized_magnitude[:, None, :, :])
        return mask * complex_spec

    def training_step(self, batch, batch_idx):
        noisy_wave, clean_wave = batch  # Tensors are on GPU.
        complex_spec_estimate = self.forward(noisy_wave)
        
        complex_spec_clean = self.stft(clean_wave)
        loss = psa(complex_spec_clean, complex_spec_estimate)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_wave, clean_wave = batch  # Tensors are on GPU.
        complex_spec_estimate = self.forward(noisy_wave)
        
        complex_spec_clean = self.stft(clean_wave)
        valid_loss = psa(complex_spec_clean, complex_spec_estimate)
        
        self.log('valid_loss', valid_loss)
        return valid_loss.item()
    
    def validation_epoch_end(self, outputs):
        self.log('valid_avg_loss', np.mean(outputs), prog_bar=True)


    def test_step(self, batch, batch_idx):
        noisy_wave, clean_wave, length = batch  # Tensors are on GPU. Batch size is assumed to be 1
        complex_spec_estimate = self.forward(noisy_wave)
        estimate_wave = self.istft(complex_spec_estimate)

        clean_wave = clean_wave[0, :length]
        noisy_wave = noisy_wave[0, :length]
        estimate_wave = estimate_wave[0, :length]
        sisdri = ((sisdr(clean_wave, estimate_wave) - sisdr(clean_wave, noisy_wave)).item())
            
        self.log('test_sisdri', sisdri, prog_bar=True)
        return sisdri

    def configure_optimizers(self):
        params = self.hparams.optimizer_hparams
        optimizer = optim.AdamW(self.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=params.warmup, max_iters=params.nepoch)
        scheduler = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
        return [optimizer], [scheduler]

    def pre_process_on_gpu(self, wave, flooring=1e-4):
        complex_spec = self.stft(wave)
        magnitude = torch.abs(complex_spec)
        maxval = magnitude.reshape(-1).max(-1)[0]
        log_magnitude = torch.log10(torch.clamp(magnitude, min=flooring*maxval))
        normalized_magnitude = log_magnitude - log_magnitude.mean(-1, keepdim=True)
        return complex_spec, normalized_magnitude