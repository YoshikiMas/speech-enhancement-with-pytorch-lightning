import functools
import argparse
import pathlib
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.pldataset import VbdDataModule
from src.plmodel import VbdLitModel
from src.dataset import VbdTestDataset


def collate_fn(batch, stft_hparams):
    # Batch size is assumed to be 1
    noisy, clean = batch[0]

    length = len(clean)
    npad = (length - stft_hparams.nfft) // stft_hparams.nhop * stft_hparams.nhop + stft_hparams.nfft - length
    noisy = F.pad(noisy, (0, npad))
    clean = F.pad(clean, (0, npad))
    return noisy[None, :], clean[None, :], length


def test(dataset, plmodel, cfg):
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    teset_data_loader = DataLoader(dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   drop_last=False,
                                   pin_memory=True,
                                   num_workers=1,
                                   collate_fn=functools.partial(collate_fn, stft_hparams=cfg.stft)
                                  )       
    
    trainer = Trainer(gpus=1)
    result = trainer.test(model=plmodel, dataloaders=[teset_data_loader])
    print(result[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('best_epoch', type=int)
    args = parser.parse_args()
    
    cfg_path = pathlib.Path(args.cfg_path)
    cfg = OmegaConf.load(cfg_path.joinpath('config.yaml'))
    
    dataset = VbdTestDataset(args.data_dir)
    plmodel = VbdLitModel.load_from_checkpoint(cfg_path.joinpath(f'model_epoch={args.best_epoch:04}.ckpt'))
    test(dataset, plmodel, cfg)

if __name__ == '__main__':
    main()