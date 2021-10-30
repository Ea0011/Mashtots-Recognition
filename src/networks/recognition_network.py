from typing import List, Tuple
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
from .blocks.res_block import ResBlock
from .utils.module_builder import ModuleBuilder

class HandwrittingRecognitionNet(pl.LightningModule):
  def __init__(self, hparams, **kwargs):
    """
    Initialize the model from a given dict containing all your hparams
    """
    super(HandwrittingRecognitionNet, self).__init__()
    for key in hparams.keys():
      self.hparams[key]=hparams[key]

    self.feature_extractor, self.classifier = self.build_model(hparams)
    self.loss_func = nn.CrossEntropyLoss()

  def build_model(self, hparams) -> Tuple[nn.ModuleList, nn.Sequential]:
    feature_extractor = ModuleBuilder(ResBlock, hparams["res_block_params"])()
    output_dim = hparams["res_block_params"][-1]["out_channels"]

    classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Conv2d(output_dim, 78, 1),
      nn.Flatten(),
    )

    return feature_extractor, classifier

  def forward(self, x):
    for _, l in enumerate(self.feature_extractor):
      x = l(x)

    x = self.classifier(x)

    return x

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
      optimizer = torch.optim.SGD(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        momentum=0.9,
        lr=self.hparams["learning_rate"],
        weight_decay = self.hparams["weight_decay"],
      )
    if self.hparams["optimizer"] == "Adam":
      optimizer = torch.optim.Adam(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        lr=self.hparams["learning_rate"],
        weight_decay = self.hparams["weight_decay"],
      )
    if self.hparams["optimizer"] == "AdamW":
      optimizer = torch.optim.AdamW(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        lr=self.hparams["learning_rate"],
        weight_decay = self.hparams["weight_decay"],
      )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-8, verbose=True, factor=0.5)

    return {"optimizer": optimizer, "scheduler": scheduler}

