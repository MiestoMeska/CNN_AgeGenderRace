import pytorch_lightning as pl
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchmetrics
from torchmetrics import Accuracy, Precision, F1Score
import torch.optim as optim


class ResNetMultiOutputCNN(pl.LightningModule):
    def __init__(self, class_weights, lr=1e-3, n_classes_gender=2, n_classes_race=5, n_classes_age=9):
        super(ResNetMultiOutputCNN, self).__init__()
        self.save_hyperparameters()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        num_features = self.resnet.fc.in_features
        

        self.resnet.fc = nn.Identity()
        
        self.fc_gender = nn.Linear(num_features, n_classes_gender)
        self.fc_race = nn.Linear(num_features, n_classes_race)
        self.fc_age = nn.Linear(num_features, n_classes_age)
        
        self.criterion_gender = nn.CrossEntropyLoss(weight=class_weights['gender'])
        self.criterion_race = nn.CrossEntropyLoss(weight=class_weights['race'])
        self.criterion_age = nn.CrossEntropyLoss(weight=class_weights['age'])


        self.gender_accuracy = Accuracy(task='multiclass', num_classes=n_classes_gender)
        self.gender_precision = Precision(task='multiclass', num_classes=n_classes_gender, average='macro')
        self.gender_f1 = F1Score(task='multiclass', num_classes=n_classes_gender, average='macro')
        
        self.race_accuracy = Accuracy(task='multiclass', num_classes=n_classes_race)
        self.race_precision = Precision(task='multiclass', num_classes=n_classes_race, average='macro')
        self.race_f1 = F1Score(task='multiclass', num_classes=n_classes_race, average='macro')
        
        self.age_accuracy = Accuracy(task='multiclass', num_classes=n_classes_age)
        self.age_precision = Precision(task='multiclass', num_classes=n_classes_age, average='macro')
        self.age_f1 = F1Score(task='multiclass', num_classes=n_classes_age, average='macro')



    def forward(self, x):
        x = x.to(self.device)
        x = self.resnet(x)
        gender_output = self.fc_gender(x)
        race_output = self.fc_race(x)
        age_output = self.fc_age(x)
        return gender_output, race_output, age_output

    def _compute_metrics(self, gender_output, race_output, age_output, labels_gender, labels_race, labels_age, stage):
        gender_acc = self.gender_accuracy(gender_output, labels_gender)
        gender_precision = self.gender_precision(gender_output, labels_gender)
        gender_f1 = self.gender_f1(gender_output, labels_gender)

        race_acc = self.race_accuracy(race_output, labels_race)
        race_precision = self.race_precision(race_output, labels_race)
        race_f1 = self.race_f1(race_output, labels_race)

        age_acc = self.age_accuracy(age_output, labels_age)
        age_precision = self.age_precision(age_output, labels_age)
        age_f1 = self.age_f1(age_output, labels_age)

        self.log(f'{stage}_gender_acc', gender_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_precision', gender_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_f1', gender_f1, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}_race_acc', race_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_precision', race_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_f1', race_f1, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{stage}_age_acc', age_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_age_precision', age_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_age_f1', age_f1, on_step=True, on_epoch=True, prog_bar=True)


    def training_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age = batch

        gender_output, race_output, age_output = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age = self.criterion_age(age_output, labels_age)
        
        loss = loss_gender + loss_race + loss_age
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        self._compute_metrics(gender_output, race_output, age_output, labels_gender, labels_race, labels_age, stage='train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age = batch
        gender_output, race_output, age_output = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age = self.criterion_age(age_output, labels_age)
        loss = loss_gender + loss_race + loss_age
        
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        self._compute_metrics(gender_output, race_output, age_output, labels_gender, labels_race, labels_age, stage='val')

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.fc_gender.parameters()},
            {'params': self.fc_race.parameters()},
            {'params': self.fc_age.parameters()},
        ], lr=self.hparams.lr)
        return optimizer
