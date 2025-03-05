from torch import Tensor, nn

from terratorch.tasks import SemanticSegmentationTask
from terratorch.models.model import AuxiliaryHead
from terratorch.tasks.tiled_inference import TiledInferenceParameters

from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex, Dice

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

import matplotlib.pyplot as plt

def TensorboardPlot(models: list, metrics: str):

    nrows, ncols = 1, 3
    fig = plt.figure(figsize=(ncols*4, nrows*4))

    for i, t in enumerate(['train', 'val', 'test']):
        ax = fig.add_subplot(nrows, ncols, i+1)
        for k, m in models.items():
            m_scores = m[0] if t != 'test' else m[1]
            score = m_scores.get_values(f"{t}/{metrics}")
            ax.plot(score, label=f"{k}: {score.max():.4f}")
        ax.set_title(t)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f"{metrics}")
        ax.legend()

    fig.tight_layout()

class TensorboardLogReader():

    def __init__(self, event_file: str):
        self.event_file = event_file
        self.acc = EventAccumulator(event_file).Reload()
        self.tags = self.acc.Tags()['scalars']

    def get_values(self, name):

        values = self.acc.Scalars(name)
        values = np.array([v.value for v in values])

        return values

class CustomSemanticSegmentationTask(SemanticSegmentationTask):

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "Multiclass_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                "Multiclass_Accuracy_Class": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        multidim_average="global",
                        average=None,
                    ),
                    labels=class_names,
                ),
                "Multiclass_Jaccard_Index_Micro": MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
                "Multiclass_Jaccard_Index_Macro": MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index,
                ),
                "Multiclass_Jaccard_Index_Class": ClasswiseWrapper(
                    MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),
                "Dice_Micro": Dice(
                    num_classes=num_classes, ignore_index=ignore_index,
                ),
                "Dice_Macro": MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "Multiclass_F1_Score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [metrics.clone(prefix=f"test/{dl_name}/") for dl_name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])
