import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np

class HandwrittingRecognitionNet(pl.LightningModule):
  def __init__(self, hparams, **kwargs):
    """
    Initialize the model from a given dict containing all your hparams
    """
    super(HandwrittingRecognitionNet, self).__init__()
    for key in hparams.keys():
      self.hparams[key]=hparams[key]

    self.model = self.build_model(hparams)
    self.loss_func = nn.CrossEntropyLoss()

    def init_weights(layer):
      if type(layer) in [nn.Conv2d]:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.constant_(layer.bias, 0.01)

      # initialize layer weights
      self.model.apply(init_weights)

  def build_model(self, hparams):
    model = nn.Sequential(
      nn.Conv2d(1, 512, 1),
      nn.BatchNorm2d(512),
      nn.ReLU(512),
      nn.Dropout(0.2),
      nn.Conv2d(512, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(256),
      nn.Dropout(0.2),
      nn.Conv2d(256, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(128),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Conv2d(128, 78, 1),
      nn.Flatten(),
    )

    return model

  def forward(self, x):
    letter = self.model(x)

    return letter

  def general_end(self, loss, mode):
    avg_loss = torch.stack([x for x in loss]).mean()
    return avg_loss

  def general_step(self, batch, batch_idx, mode="train"):
    inputs = batch
    images, labels = inputs["image"], inputs["label"]
    
    predicted_letter = self.forward(images)

    loss = self.loss_func(predicted_letter, labels)
    return loss

  def training_step(self, batch, batch_idx):
    loss = self.general_step(batch, batch_idx)
    self.log('train_loss', loss)

    return loss
  
  def validation_step(self, batch, batch_idx):
    loss = self.general_step(batch, batch_idx, mode="val")
    self.log('val_loss', loss)

    return loss
  
  def validation_epoch_end(self, outputs):
    avg_loss = self.general_end(outputs, "val")
    print("Avg-Loss={}".format(avg_loss))
    tensorboard_logs = {'val/loss': avg_loss}

    return {'val_loss': avg_loss, 'log': tensorboard_logs}

  def configure_optimizers(self):
    optimizer = None

    if self.hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])
    if self.hparams["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])
    if self.hparams["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams["learning_rate"], weight_decay = self.hparams["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-8, verbose=True, factor=0.5)

    return {"optimizer": optimizer, "scheduler": scheduler}

