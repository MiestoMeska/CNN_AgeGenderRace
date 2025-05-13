import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics import Accuracy, Precision, F1Score, MeanAbsoluteError

class EfficientNetMultiOutput_v4(pl.LightningModule):
    def __init__(self, class_weights, lr=1e-3, n_classes_gender=2):
        super(EfficientNetMultiOutput_v4, self).__init__()
        self.save_hyperparameters()

        # ✅ Rename to backbone for BackboneFinetuning compatibility
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        self.gender_classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes_gender)
        )

        self.age_regressor = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.criterion_gender = nn.CrossEntropyLoss(weight=class_weights['gender'])
        self.criterion_age = nn.MSELoss()

        self.gender_accuracy = Accuracy(task='multiclass', num_classes=n_classes_gender)
        self.gender_precision = Precision(task='multiclass', num_classes=n_classes_gender, average='macro')
        self.gender_f1 = F1Score(task='multiclass', num_classes=n_classes_gender, average='macro')
        self.age_mae = MeanAbsoluteError()

    def forward(self, x):
        features = self.backbone(x)
        gender_output = self.gender_classifier(features)
        age_output = self.age_regressor(features).squeeze(1)
        return gender_output, age_output

    def compute_metrics(self, gender_output, age_output, labels_gender, labels_age, stage):
        gender_acc = self.gender_accuracy(gender_output, labels_gender)
        gender_precision = self.gender_precision(gender_output, labels_gender)
        gender_f1 = self.gender_f1(gender_output, labels_gender)
        age_mae = self.age_mae(age_output, labels_age)

        self.log(f'{stage}_gender_acc', gender_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_precision', gender_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_f1', gender_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_age_mae', age_mae, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        images, labels_gender, labels_age = batch

        gender_output, age_output = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_age = self.criterion_age(age_output, labels_age.float())

        loss = loss_gender + loss_age

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.compute_metrics(gender_output, age_output, labels_gender, labels_age, stage='train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_gender, labels_age = batch

        gender_output, age_output = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_age = self.criterion_age(age_output, labels_age.float())

        loss = loss_gender + loss_age

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.compute_metrics(gender_output, age_output, labels_gender, labels_age, stage='val')

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.backbone.parameters()},  # ✅ include backbone in optimizer
            {'params': self.gender_classifier.parameters()},
            {'params': self.age_regressor.parameters()},
        ], lr=self.hparams.lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
