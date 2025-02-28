import os
import torch
import rasterio
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from typing import Any

from torch import Tensor
from torchgeo.datasets import CloudCoverDetection, NonGeoDataset
from torchgeo.datasets.errors import RGBBandsMissingError
from torchgeo.datasets.utils import Path
from torchgeo.datamodules import NonGeoDataModule

from typing import ClassVar
from collections.abc import Callable, Sequence
from matplotlib.figure import Figure

from terratorch.datasets.transforms import albumentations_to_callable_with_dict
from terratorch.datamodules.torchgeo_data_module import build_callable_transform_from_torch_tensor

class CustomNonGeoDataModule(NonGeoDataModule):

    def __init__(
        self,
        dataset_class: type[NonGeoDataset],
        batch_size: int = 1,
        num_workers: int = 0,
        train_aug: A.Compose | None | list[A.BasicTransform] = None,
        val_aug: A.Compose | None | list[A.BasicTransform] = None,
        test_aug: A.Compose | None | list[A.BasicTransform] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new CustomNonGeoDataModule instance.

        Args:
            dataset_class: Class used to instantiate a new dataset.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            train_aug:
            val_aug:      ... NonGeoDataModule does not distinguish transform functions ...
            test_aug:
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        self.train_aug = train_aug
        self.val_aug = val_aug
        self.test_aug = test_aug

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        # print(self.kwargs)
        if stage in ['fit']:
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='train', transforms=self.train_aug, **self.kwargs
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='val', transforms=self.val_aug, **self.kwargs
            )
        if stage in ['test']:
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='test', transforms=self.test_aug, **self.kwargs
            )

class CustomCloudCoverDetection(CloudCoverDetection):
    """
    Customized Tranform Function.
    - Compatible with Albumentation
    - Plot RGB and NIR together
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/ref_cloud_cover_detection_challenge_v1/final'
    all_bands = ('B02', 'B03', 'B04', 'B08')
    rgb_bands = ('B04', 'B03', 'B02')
    splits: ClassVar[dict[str, str]] = {'train': 'public', 'test': 'private', 'val': ''}

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: A.Compose | None | list[A.BasicTransform] = None,
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

        if isinstance(transforms, list):
            transforms_as_callable = albumentations_to_callable_with_dict(transforms)
            self.transforms = build_callable_transform_from_torch_tensor(transforms_as_callable)
        else:
            self.transforms = transforms

        self.download = download

        self.sub_root = 'train' if self.split == 'val' else self.split

        self.csv = os.path.join(self.root, self.sub_root, f'{self.split}_metadata.csv')
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

        if self.transforms is not None:
            sample = self.transforms(
                tensor_dict={
                    "image": self._load_image(chip_id), 
                    "mask":self._load_target(chip_id)}
                )

        # return sample
        return sample

    def _load_image(self, chip_id: str) -> Tensor:
        """Load all source images for a chip.

        Args:
            chip_id: ID of the chip.

        Returns:
            a tensor of stacked source image data
        """
        path = os.path.join(self.root, self.sub_root, f'{self.sub_root}_features', chip_id)
        images = []
        for band in self.bands:
            with rasterio.open(os.path.join(path, f'{band}.tif')) as src:
                images.append(src.read(1).astype(np.float32))

        return torch.from_numpy(np.array(images))

    def _load_target(self, chip_id: str) -> Tensor:
        """Load label image for a chip.

        Args:
            chip_id: ID of the chip.

        Returns:
            a tensor of the label image data
        """
        path = os.path.join(self.root, self.sub_root, f'{self.sub_root}_labels')
        with rasterio.open(os.path.join(path, f'{chip_id}.tif')) as src:
            return torch.from_numpy(src.read(1).astype(np.int64))

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
