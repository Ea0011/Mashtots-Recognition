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
from .blocks.res_block import ResBlock, ResBlockParams
from .utils.module_builder import ModuleBuilder
from .utils.visualisations import visualise_predictions
from data.image_folder_dataset import LABELS
import sys

if sys.version_info >= (3, 8):
  from typing import TypedDict  # pylint: disable=no-name-in-module
else:
  from typing_extensions import TypedDict

HandwrittingRecognitionNetParams = TypedDict(
  'HandwrittingRecognitionNetParams',
  learning_rate=float,
  weight_decay=float,
  optimizer=str,
  res_block_params=List[ResBlockParams],
)
class HandwrittingRecognitionNet(pl.LightningModule):
  def __init__(self, **hparams):
    """
    Initialize the model from a given dict containing all your hparams
    """
    super(HandwrittingRecognitionNet, self).__init__()

    self.save_hyperparameters()

    # to log compute graph to tBoard
    self.example_input_array = torch.zeros((1, 1, 64, 64)).float()

    self.feature_extractor, self.classifier = self.build_model()
    self.loss_func = nn.CrossEntropyLoss()

  def build_model(self) -> Tuple[nn.ModuleList, nn.Sequential]:
    feature_extractor = ModuleBuilder(ResBlock, self.hparams.res_block_params)()
    output_dim = self.hparams.res_block_params[-1]["out_channels"]

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

    if mode == "val":
      fig = visualise_predictions(images, labels, predicted_letter, LABELS)
      self.logger.experiment.add_figure("Pred vs Actual", fig)

    correct_items = torch.sum(torch.argmax(predicted_letter, dim=1) == labels)
    return loss, correct_items / len(labels)

  def training_step(self, batch, batch_idx):
    loss, accuracy = self.general_step(batch, batch_idx)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log('train_accuracy', accuracy * 100, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss
  
  def validation_step(self, batch, batch_idx):
    loss, accuracy = self.general_step(batch, batch_idx, mode="val")
    self.log('val_loss', loss, prog_bar=True)
    self.log('val_accuracy', accuracy * 100, prog_bar=True)

    return loss
  
  def validation_epoch_end(self, outputs):
    avg_loss = self.general_end(outputs, "val")
    print("Avg-Loss={}".format(avg_loss))
    tensorboard_logs = {'val/loss': avg_loss}

    return {'val_loss': avg_loss, 'log': tensorboard_logs}

  def configure_optimizers(self):
    optimizer = None

    if self.hparams.optimizer == "SGD":
      optimizer = torch.optim.SGD(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        momentum=0.9,
        lr=self.hparams.learning_rate,
        weight_decay = self.hparams.weight_decay,
      )
    if self.hparams.optimizer == "Adam":
      optimizer = torch.optim.Adam(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        lr=self.hparams.learning_rate,
        weight_decay = self.hparams.weight_decay,
      )
    if self.hparams.optimizer == "AdamW":
      optimizer = torch.optim.AdamW(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        lr=self.hparams.learning_rate,
        weight_decay = self.hparams.weight_decay,
      )
    if self.hparams.optimizer == "RMSProps":
      optimizer = torch.optim.AdamW(
        [
          { "params": self.feature_extractor.parameters() },
          { "params": self.classifier.parameters() },
        ],
        lr=self.hparams.learning_rate,
        weight_decay = self.hparams.weight_decay,
      )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-10, verbose=True, factor=0.1)

    return {"optimizer": optimizer, "scheduler": scheduler}

