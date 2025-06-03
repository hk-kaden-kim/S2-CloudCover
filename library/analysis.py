from torch import Tensor, nn

from terratorch.tasks import SemanticSegmentationTask
from terratorch.models.model import AuxiliaryHead
from terratorch.tasks.tiled_inference import TiledInferenceParameters

from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

import matplotlib.pyplot as plt

class TensorboardLogReader():

    def __init__(self, event_file: str):
        self.event_file = event_file
        self.acc = EventAccumulator(event_file).Reload()
        self.tags = self.acc.Tags()['scalars']

    def get_values(self, name):

        values = self.acc.Scalars(name)
        values = np.array([v.value for v in values])

        return values

def loss_plot(plots, tran_n_step=51):

    nrows, ncols = 1, len(plots)
    fig = plt.figure(figsize=(ncols*4, nrows*4))

    for i, p in enumerate(plots):

        tr_loss = p['train'].get_values('train/loss')[::tran_n_step]
        val_loss = p['train'].get_values('val/loss')

        ax = fig.add_subplot(nrows, ncols, i+1)

        ax.plot(tr_loss, label='train')
        ax.plot(val_loss, label='val')

        ax.set_ylim(0.0, 0.8)
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.set_title(p['name'])
        ax.legend()

    fig.tight_layout()

def perf_plot(plots, marker):

    nrows, ncols = 1, 1 
    fig = plt.figure(figsize=(ncols*6, nrows*4))

    for i, p in enumerate(plots):
        
        mIoU = p['test'].get_values('test/Multiclass_Jaccard_Index')
        param = p['param_M']

        plt.scatter([param], mIoU, label=p['name'], s=marker[i][3], 
                    marker=marker[i][0], edgecolors=marker[i][1], facecolors=marker[i][2])

    plt.ylabel('mIoU')
    plt.xlabel('params (M)')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    fig.tight_layout()