import os
import torch
import rasterio
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from torch import Tensor
from torchgeo.datasets import CloudCoverDetection
from torchgeo.datasets.errors import RGBBandsMissingError
from torchgeo.datasets.utils import Path
from typing import ClassVar
from collections.abc import Callable, Sequence
from matplotlib.figure import Figure

class _CustomTransforms(object):
    """Customized Tranform Function."""

    def __call__(self, sample):

        image, mask = sample['image'], sample['mask']

        MIN_MAX_SCALE = A.Normalize(normalization='min_max_per_channel')

        image = image.cpu().detach().numpy() # Convert to the numpy array
        image = np.transpose(image, (1,2,0)) # (B,H,W) to (H,W,B)
        image = MIN_MAX_SCALE(image=image)['image'] # Min-Max Scaling by each Band
        image = torch.from_numpy(np.transpose(image, (2,0,1))) # (H,W,B) to (B,H,W)

        return {'image': image, 'mask': mask}

class CustomCloudCoverDetection(CloudCoverDetection):
    """
    Customized Tranform Function.
    - Compatible with Albumentation
    - Plot RGB and NIR together
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/ref_cloud_cover_detection_challenge_v1/final'
    all_bands = ('B02', 'B03', 'B04', 'B08')
    rgb_bands = ('B04', 'B03', 'B02')
    splits: ClassVar[dict[str, str]] = {'train': 'public', 'test': 'private'}

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initiatlize a CloudCoverDetection instance.

        Args:
            root: root directory where dataset can be found
            split: 'train' or 'test'
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: If *split* or *bands* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self.csv = os.path.join(self.root, self.split, f'{self.split}_metadata.csv')
        self._verify()

        self.metadata = pd.read_csv(self.csv)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Returns a sample from dataset.

        Args:
            index: index to return

        Returns:
            data and label at given index
        """
        chip_id = self.metadata.iat[index, 0]
        image = self._load_image(chip_id)
        label = self._load_target(chip_id)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        sample = {'image': image, 'mask': label}

        return sample

    def _load_image(self, chip_id: str) -> Tensor:
        """Load all source images for a chip.

        Args:
            chip_id: ID of the chip.

        Returns:
            a tensor of stacked source image data
        """
        path = os.path.join(self.root, self.split, f'{self.split}_features', chip_id)
        images = []
        for band in self.bands:
            with rasterio.open(os.path.join(path, f'{band}.tif')) as src:
                images.append(src.read(1).astype(np.float32))
        return np.stack(images, axis=2)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            time_step: time step at which to access image, beginning with 0
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        if 'prediction' in sample:
            prediction = sample['prediction']
            n_cols = 4
        else:
            n_cols = 3

        image, mask = sample['image'], sample['mask']

        R,G,B = image[rgb_indices[0],:,:], image[rgb_indices[1],:,:], image[rgb_indices[2],:,:]
        NIR = image[-1,:,:]
        
        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))
        axs[0].imshow(np.stack((R, G, B), axis=2))
        axs[0].axis('off')
        axs[1].imshow(NIR)
        axs[1].axis('off')
        axs[2].imshow(mask)
        axs[2].axis('off')

        if 'prediction' in sample:
            axs[3].imshow(prediction)
            axs[3].axis('off')
            if show_titles:
                axs[3].set_title('Prediction')

        if show_titles:
            axs[0].set_title('RGB\n(B4,B3,B2)')
            axs[1].set_title('NIR\n(B8)')
            axs[2].set_title('Mask\n(0:No, 1:Cloud)')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
