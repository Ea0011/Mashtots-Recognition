import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

def images_to_probs(outputs):
  '''
  Generates predictions and corresponding probabilities from a trained
  network and a list of images
  '''
  # convert output probabilities to predicted class
  _, preds_tensor = torch.max(outputs, 1)
  preds = np.squeeze(preds_tensor.numpy())
  return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]

def visualise_predictions(images, labels, model_outputs, classes):
  '''
  Generates matplotlib Figure using a trained network, along with images
  and labels from a batch, that shows the network's top prediction along
  with its probability, alongside the actual label, coloring this
  information based on whether the prediction was correct or not.
  Uses the "images_to_probs" function.
  '''
  preds, probs = images_to_probs(model_outputs)
  # plot the images in the batch, along with predicted and true labels
  fig = plt.figure(figsize=(12, 48))
  sample = np.random.permutation(np.arange(len(labels)))[:8]
  for count, idx in enumerate(sample):
    ax = fig.add_subplot(4, 4, count+1, xticks=[], yticks=[])
    plt.imshow(images[idx].squeeze(0))
    ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
      classes[preds[idx]],
      probs[idx] * 100.0,
      classes[labels[idx]]),
        color=("green" if preds[idx]==labels[idx].item() else "red")
      )

  return fig