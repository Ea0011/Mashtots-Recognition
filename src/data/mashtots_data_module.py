import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class MashtotsDataModule(pl.LightningDataModule):
  def __init__(self, train_set, val_set, test_set, batch_size: 32, **kwargs) -> None:
    super().__init__()
    self.train_data = train_set
    self.val_data = val_set
    self.test_data = test_set
    self.batch_size = batch_size

  def setup(self, stage = None):
    self.training_set = self.train_data
    self.validation_set = self.val_data
    self.test_set = self.test_data

  def train_dataloader(self):
    return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

  def val_dataloader(self):
    return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)