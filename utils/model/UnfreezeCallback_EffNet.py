import torch
import pytorch_lightning as pl

class UnfreezeCallback_EffNet(pl.Callback):
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """
        Args:
            patience (int): Number of epochs to wait before unfreezing layers if validation loss stalls.
            min_delta (float): Minimum change in the monitored quantity to qualify as improvement.
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = None
        self.wait = 0
        self.unfrozen_once = False
        self.unfrozen_twice = False

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss")

        if current_val_loss is None:
            return

        if self.best_val_loss is None:
            self.best_val_loss = current_val_loss

        elif current_val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = current_val_loss
            self.wait = 0

        else:
            self.wait += 1

            if self.wait >= self.patience and not self.unfrozen_once:
                print("Unfreezing blocks 7, 6, and 5 due to plateau in validation loss.")
                self.unfreeze_layers(pl_module, num_layers=3)
                self.unfrozen_once = True
                self.wait = 0

            elif self.wait >= self.patience and self.unfrozen_once and not self.unfrozen_twice:
                print("Unfreezing all layers due to plateau in validation loss.")
                self.unfreeze_layers(pl_module, num_layers=None)
                self.unfrozen_twice = True

    def unfreeze_layers(self, pl_module, num_layers=None):
        """
        Unfreezes layers in the EfficientNet model. If num_layers is provided, it only unfreezes that number of layers.
        If num_layers is None, it unfreezes all layers.
        """
        layers_to_unfreeze = [pl_module.efficientnet.features[6],
                              pl_module.efficientnet.features[5],
                              pl_module.efficientnet.features[4]]
        
        if num_layers is not None:
            layers_to_unfreeze = layers_to_unfreeze[:num_layers]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        current_lr = pl_module.hparams.lr
        new_lr = current_lr * 0.3
        print(f"Learning rate changed from {current_lr} to {new_lr}")
        pl_module.hparams.lr = new_lr

        new_optimizer = torch.optim.AdamW([
            {'params': pl_module.efficientnet.parameters()},
            {'params': pl_module.gender_classifier.parameters()},
            {'params': pl_module.race_classifier.parameters()},
            {'params': pl_module.age_regression.parameters()},
        ], lr=new_lr, weight_decay=1e-5)

        pl_module.trainer.optimizers = [new_optimizer]

        torch.cuda.empty_cache()

        print(f"Layers unfrozen: {num_layers if num_layers else 'all'}, optimizer reinitialized, learning rate decreased.")
