import torch.nn as nn
import torch.nn.functional as F
import sys

if sys.version_info >= (3, 8):
  from typing import TypedDict  # pylint: disable=no-name-in-module
else:
  from typing_extensions import TypedDict

ResBlockParams = TypedDict(
  'ResBlockParams', 
  input_channels=int,
  out_channels=int,
  stride=int,
  kernel_size=int,
  dropout_p=float,
  padding=int,
  use_pool=bool,
)

class ResBlock(nn.Module):
  def __init__(self, input_channels, out_channels, stride=1, kernel_size=3, dropout_p=0.5, padding=1, use_pool=False) -> None:
    """
    Args:
      in_channels (int):  Number of input channels.
      out_channels (int): Number of output channels.
      stride (int):       Controls the stride.
      kernel_size(int):   Size of the Conv layer.
      dorpout_p(probability): The ratio of droupout feature maps
      use_pool(bool):     Whether to perform max pool at the end
    """
    super(ResBlock, self).__init__()

    self.skip = nn.Sequential()
    self.dropout = nn.Dropout(p=dropout_p)
    self.pool = nn.MaxPool2d(2) if use_pool == True else None
    self.relu = nn.ReLU()

    # Adapt num of channels if input or output do not match in shapes
    if stride != 1 or input_channels != out_channels:
      self.skip = nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.skip = None

    self.block = nn.Sequential(
      nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Dropout(p=dropout_p),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
      nn.BatchNorm2d(out_channels),
    )

  def forward(self, x):
    out = self.block(x)
    out += (x if self.skip is None else self.skip(x))
    out = self.relu(out)
    out = out if self.pool is None else self.pool(out) 
    return self.dropout(out)
