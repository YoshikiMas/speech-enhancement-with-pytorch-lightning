from pytorch_lightning.callbacks import ProgressBar

class PerEpochProgressBar(ProgressBar):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch:
            print()
        super().on_train_epoch_start(trainer, pl_module)