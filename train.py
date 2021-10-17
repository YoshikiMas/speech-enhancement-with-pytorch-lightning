import argparse
import pathlib
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.pldataset import VbdDataModule
from src.plmodel import VbdLitModel
from src.utils.callbacks import PerEpochProgressBar


def train(dm, plmodel, cfg, save_path):
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    early_stop_callback = EarlyStopping(monitor='valid_loss',
                                        min_delta=0.00,
                                        patience=cfg.optim.patience
                                       )
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          mode='min',
                                          dirpath=save_path,
                                          filename='model_{epoch:04d}',
                                          save_top_k=1
                                         )
    pbar_callback = PerEpochProgressBar()
    
    trainer = Trainer(max_epochs=cfg.optim.nepoch,
                      gpus=1,
                      precision=16,
                      num_sanity_val_steps=-1,
                      callbacks=[checkpoint_callback, early_stop_callback, pbar_callback],
                     )
    
    trainer.fit(plmodel, dm)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path', type=str)
    args = parser.parse_args()
    
    cfg_path = args.cfg_path
    cfg = OmegaConf.load(pathlib.Path(cfg_path).joinpath('config.yaml'))
    
    dm = VbdDataModule(base_dir=cfg.data.base_dir,
                       siglen=cfg.data.siglen,
                       batch_size=cfg.optim.batch_size,
                       stft_hparams=cfg.stft
                      )

    plmodel = VbdLitModel(cfg.model,
                          cfg.optim,
                          cfg.stft
                         )

    trainer = train(dm, plmodel, cfg, save_path=cfg_path)
    result = trainer.test()
    print(result)


if __name__ == '__main__':
    main()