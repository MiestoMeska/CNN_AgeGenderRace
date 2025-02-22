import pytorch_lightning as pl
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchmetrics
from torchmetrics import Accuracy, Precision, F1Score, MeanAbsoluteError, MeanSquaredError
import torch.optim as optim

class ResNetMultiOutputCNN_mean(pl.LightningModule):
    def __init__(self, class_weights, lr=1e-3, n_classes_gender=2, n_classes_race=5, n_classes_age=9):
        super(ResNetMultiOutputCNN_mean, self).__init__()
        self.save_hyperparameters()
        
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        num_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity()
        
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
        
        self.age_regression = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.criterion_gender = nn.CrossEntropyLoss(weight=class_weights['gender'])
        self.criterion_race = nn.CrossEntropyLoss(weight=class_weights['race'])
        self.criterion_age_regression = nn.MSELoss()

        self.gender_accuracy = Accuracy(task='multiclass', num_classes=n_classes_gender)
        self.gender_precision = Precision(task='multiclass', num_classes=n_classes_gender, average='macro')
        self.gender_f1 = F1Score(task='multiclass', num_classes=n_classes_gender, average='macro')
        
        self.race_accuracy = Accuracy(task='multiclass', num_classes=n_classes_race)
        self.race_precision = Precision(task='multiclass', num_classes=n_classes_race, average='macro')
        self.race_f1 = F1Score(task='multiclass', num_classes=n_classes_race, average='macro')
        
        self.age_mae = MeanAbsoluteError()
        self.age_mse = MeanSquaredError()

    def forward(self, x):
        features = self.resnet(x)

        gender_output = self.gender_classifier(features)
        race_output = self.race_classifier(features)
        age_output_regression = self.age_regression(features).squeeze(1)
        age_output_regression = age_output_regression * 8

        return gender_output, race_output, age_output_regression

    def predict_age_group(self, age_output_regression):
        predicted_age_group = torch.round(age_output_regression).long()

        predicted_age_group = torch.clamp(predicted_age_group, min=0, max=8)
        
        return predicted_age_group  

    def compute_metrics(self, gender_output, race_output, age_output_regression, labels_gender, labels_race, labels_age, stage):
        device = self.device

        gender_acc = self.gender_accuracy.to(device)(gender_output, labels_gender)
        gender_precision = self.gender_precision.to(device)(gender_output, labels_gender)
        gender_f1 = self.gender_f1.to(device)(gender_output, labels_gender)

        race_acc = self.race_accuracy.to(device)(race_output, labels_race)
        race_precision = self.race_precision.to(device)(race_output, labels_race)
        race_f1 = self.race_f1.to(device)(race_output, labels_race)

        predicted_age_groups = self.predict_age_group(age_output_regression)

        age_group_acc = Accuracy(task='multiclass', num_classes=9).to(device)(predicted_age_groups, labels_age)
        age_group_f1 = F1Score(task='multiclass', num_classes=9, average='macro').to(device)(predicted_age_groups, labels_age)

        age_mae = self.age_mae.to(device)(age_output_regression, labels_age.float())
        age_mse = self.age_mse.to(device)(age_output_regression, labels_age.float())

        self.log(f'{stage}_gender_acc', gender_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_precision', gender_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_f1', gender_f1, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}_race_acc', race_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_precision', race_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_f1', race_f1, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{stage}_age_mae', age_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_age_mse', age_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{stage}_age_group_acc', age_group_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_age_group_f1', age_group_f1, on_step=True, on_epoch=True, prog_bar=True)


    def training_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age_group = batch

        gender_output, race_output, age_output_regression = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)

        loss_age_group_regression = self.criterion_age_regression(age_output_regression, labels_age_group.float())  
        
        loss = loss_gender + loss_race + loss_age_group_regression

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        self.compute_metrics(gender_output, race_output, age_output_regression, labels_gender, labels_race, labels_age_group, stage='train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age_group = batch
        gender_output, race_output, age_output_regression = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age_regression = self.criterion_age_regression(age_output_regression, labels_age_group.float())  # Loss for continuous age regression
        
        loss = loss_gender + loss_race + loss_age_regression
        
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        self.compute_metrics(gender_output, race_output, age_output_regression, labels_gender, labels_race, labels_age_group, stage='val')

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.gender_classifier.parameters()},
            {'params': self.race_classifier.parameters()},
            {'params': self.age_regression.parameters()},
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
