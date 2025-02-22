import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from torchmetrics import Accuracy, Precision, F1Score
import torch.optim as optim

class EfficientNetMultiOutput_class(pl.LightningModule):
    def __init__(self, class_weights, lr=1e-3, n_classes_gender=2, n_classes_race=5, n_classes_age=9):
        super(EfficientNetMultiOutput_class, self).__init__()
        self.save_hyperparameters()

        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        num_features = self.efficientnet.classifier[1].in_features
        
        self.efficientnet.classifier = nn.Identity() 

        self.gender_classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes_gender)
        )
        
        self.race_classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes_race)
        )
        
        self.age_group_classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes_age)
        )
        
        self.criterion_gender = nn.CrossEntropyLoss(weight=class_weights['gender'])
        self.criterion_race = nn.CrossEntropyLoss(weight=class_weights['race'])
        self.criterion_age_group = nn.BCEWithLogitsLoss(pos_weight=class_weights['age'])  # Binary Cross-Entropy for multi-label classification

        self.gender_accuracy = Accuracy(task='multiclass', num_classes=n_classes_gender)
        self.gender_precision = Precision(task='multiclass', num_classes=n_classes_gender, average='macro')
        self.gender_f1 = F1Score(task='multiclass', num_classes=n_classes_gender, average='macro')
        
        self.race_accuracy = Accuracy(task='multiclass', num_classes=n_classes_race)
        self.race_precision = Precision(task='multiclass', num_classes=n_classes_race, average='macro')
        self.race_f1 = F1Score(task='multiclass', num_classes=n_classes_race, average='macro')
        
        self.age_group_f1 = F1Score(task='multilabel', num_labels=n_classes_age, average='macro')

    def forward(self, x):
        features = self.efficientnet(x)

        gender_output = self.gender_classifier(features)
        race_output = self.race_classifier(features)
        age_group_output = self.age_group_classifier(features) 

        return gender_output, race_output, age_group_output

    def compute_metrics(self, gender_output, race_output, age_group_output, labels_gender, labels_race, labels_age, stage):
        device = self.device

        gender_acc = self.gender_accuracy.to(device)(gender_output, labels_gender)
        gender_precision = self.gender_precision.to(device)(gender_output, labels_gender)
        gender_f1 = self.gender_f1.to(device)(gender_output, labels_gender)

        race_acc = self.race_accuracy.to(device)(race_output, labels_race)
        race_precision = self.race_precision.to(device)(race_output, labels_race)
        race_f1 = self.race_f1.to(device)(race_output, labels_race)

        age_group_f1 = self.age_group_f1.to(device)(age_group_output, labels_age)

        self.log(f'{stage}_gender_acc', gender_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_precision', gender_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_f1', gender_f1, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}_race_acc', race_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_precision', race_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_f1', race_f1, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{stage}_age_group_f1', age_group_f1, on_step=True, on_epoch=True, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age_group = batch  

        labels_age_group = F.one_hot(labels_age_group, num_classes=self.hparams.n_classes_age).float()

        gender_output, race_output, age_group_output = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age_group = self.criterion_age_group(age_group_output, labels_age_group)
        
        loss = loss_gender + loss_race + loss_age_group

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        self.compute_metrics(gender_output, race_output, age_group_output, labels_gender, labels_race, labels_age_group, stage='train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age_group = batch

        labels_age_group = F.one_hot(labels_age_group, num_classes=self.hparams.n_classes_age).float()

        gender_output, race_output, age_group_output = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age_group = self.criterion_age_group(age_group_output, labels_age_group)
        
        loss = loss_gender + loss_race + loss_age_group
        
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        self.compute_metrics(gender_output, race_output, age_group_output, labels_gender, labels_race, labels_age_group, stage='val')

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.gender_classifier.parameters()},
            {'params': self.race_classifier.parameters()},
            {'params': self.age_group_classifier.parameters()},
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
