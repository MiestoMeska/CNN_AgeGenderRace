import pytorch_lightning as pl
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torchmetrics
from torchmetrics import Accuracy, Precision, F1Score, MeanAbsoluteError, MeanSquaredError
import torch.optim as optim

class EfficientNetMultiOutput_ordinalReg(pl.LightningModule):
    def __init__(self, class_weights, lr=1e-3, n_classes_gender=2, n_classes_race=5, n_classes_age=9):
        super(EfficientNetMultiOutput_ordinalReg, self).__init__()
        self.save_hyperparameters()

        self.age_weights = class_weights['age']

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
        
        self.age_classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_classes_age - 1)
        )

        self.learnable_thresholds = nn.Parameter(torch.full((n_classes_age - 1,), 0.0), requires_grad=False)

        self.criterion_gender = nn.CrossEntropyLoss(weight=class_weights['gender'])
        self.criterion_race = nn.CrossEntropyLoss(weight=class_weights['race'])
        self.criterion_ordinal = self.orn_loss

        self.gender_accuracy = Accuracy(task='multiclass', num_classes=n_classes_gender)
        self.gender_precision = Precision(task='multiclass', num_classes=n_classes_gender, average='macro')
        self.gender_f1 = F1Score(task='multiclass', num_classes=n_classes_gender, average='macro')

        self.race_accuracy = Accuracy(task='multiclass', num_classes=n_classes_race)
        self.race_precision = Precision(task='multiclass', num_classes=n_classes_race, average='macro')
        self.race_f1 = F1Score(task='multiclass', num_classes=n_classes_race, average='macro')

        self.age_group_f1 = F1Score(task='multiclass', num_classes=n_classes_age, average='macro')

    def on_train_start(self):
        print(f"Initial Thresholds: {torch.sigmoid(self.learnable_thresholds).cpu().detach().numpy()}")

    def forward(self, x):
        features = self.efficientnet(x)

        gender_output = self.gender_classifier(features)
        race_output = self.race_classifier(features)
        age_output_logits = self.age_classifier(features)

        return gender_output, race_output, age_output_logits

    def predict_age_group(self, age_output_logits):
        probs = torch.sigmoid(age_output_logits)

        thresholds = self.learnable_thresholds
        
        predicted_age_group = (probs > thresholds).sum(dim=1)
        
        return predicted_age_group

    def orn_loss(self, logits, target, num_classes=9):
        logits_cumsum = torch.sigmoid(logits)
        thresholds = torch.sigmoid(self.learnable_thresholds)

        mask = torch.arange(num_classes - 1).unsqueeze(0).repeat(logits.size(0), 1).to(target.device)
        mask_i = (target > mask).float()
        
        loss = ((logits_cumsum - thresholds.unsqueeze(0)) ** 2).mean() * mask_i.mean()
        return loss


    def compute_metrics(self, gender_output, race_output, age_output_logits, labels_gender, labels_race, labels_age, stage):
        device = self.device

        gender_acc = self.gender_accuracy.to(device)(gender_output, labels_gender)
        gender_precision = self.gender_precision.to(device)(gender_output, labels_gender)
        gender_f1 = self.gender_f1.to(device)(gender_output, labels_gender)

        race_acc = self.race_accuracy.to(device)(race_output, labels_race)
        race_precision = self.race_precision.to(device)(race_output, labels_race)
        race_f1 = self.race_f1.to(device)(race_output, labels_race)

        predicted_age_groups = self.predict_age_group(age_output_logits)

        age_group_f1 = self.age_group_f1.to(device)(predicted_age_groups, labels_age)

        self.log(f'{stage}_gender_acc', gender_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_precision', gender_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_gender_f1', gender_f1, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log(f'{stage}_race_acc', race_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_precision', race_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_race_f1', race_f1, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f'{stage}_age_group_f1', age_group_f1, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age_group = batch

        gender_output, race_output, age_output_logits = self(images)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age_ordinal = self.criterion_ordinal(age_output_logits, labels_age_group)
        
        loss = loss_gender + loss_race + loss_age_ordinal

        self.log('train_loss', loss, on_step=True, on_epoch=True)

        self.compute_metrics(gender_output, race_output, age_output_logits, labels_gender, labels_race, labels_age_group, stage='train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_gender, labels_race, labels_age_group = batch
        gender_output, race_output, age_output_logits = self(images)

        predicted_age_groups = self.predict_age_group(age_output_logits)

        loss_gender = self.criterion_gender(gender_output, labels_gender)
        loss_race = self.criterion_race(race_output, labels_race)
        loss_age_ordinal = self.criterion_ordinal(age_output_logits, labels_age_group)
        
        loss = loss_gender + loss_race + loss_age_ordinal
        
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        self.compute_metrics(gender_output, race_output, age_output_logits, labels_gender, labels_race, labels_age_group, stage='val')

        if batch_idx == 0:
            print(f"True Age Groups: {labels_age_group.cpu().numpy()}")
            print(f"Pred Age Groups: {predicted_age_groups.cpu().numpy()}")
            print(f"Age Output Logits: {age_output_logits.cpu().detach().numpy()[:5]}")
            print(f"Sigmoid Probabilities: {torch.sigmoid(age_output_logits).cpu().detach().numpy()[:5]}")
            print(f"Learned Thresholds: {torch.sigmoid(self.learnable_thresholds).cpu().detach().numpy()}")

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.gender_classifier.parameters(), 'lr': self.hparams.lr},
            {'params': self.race_classifier.parameters(), 'lr': self.hparams.lr},
            {'params': self.age_classifier.parameters(), 'lr': self.hparams.lr},
            {'params': self.learnable_thresholds, 'lr': self.hparams.lr}
        ], weight_decay=1e-5)

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
