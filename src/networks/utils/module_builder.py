from typing import Dict, List
import torch
import torch.nn as nn

class ModuleBuilder():
  def __init__(self, Block, block_params: List[Dict]) -> None:
    self.Block = Block
    self.ModuleList = nn.ModuleList()
    self.block_params = block_params

  def __call__(self) -> nn.ModuleList:
    for param in self.block_params:
      self.ModuleList.append(self.Block(**param))

    return self.ModuleList