from typing import Dict, List
import torch
import torch.nn as nn

class ModuleBuilder():
  """
  Builds a Module List with given blocks and params for each block
  Args
    Block(nn.Module):                 A module to include in each layer
    block_params(List[BlockParams]):  A list of params for each block   
  """
  def __init__(self, Block, block_params: List[Dict]) -> None:
    self.Block = Block
    self.ModuleList = nn.ModuleList()
    self.block_params = block_params

  def __call__(self) -> nn.ModuleList:
    for param in self.block_params:
      self.ModuleList.append(self.Block(**param))

    return self.ModuleList