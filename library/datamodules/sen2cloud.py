# Copyright contributors to the Terratorch project

import albumentations as A

from typing import Any
from collections.abc import Sequence

from torch import Tensor
from torch.utils.data import DataLoader

from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.io.file import load_from_file_or_attribute

from library.datasets.sen2cloud import Sen2Cloud

MEANS = {
    "B02": 0.0,
    "B03": 0.0,
    "B04": 0.0,
    "B08": 0.0,
}

STDS = {
    "B02": 1.0,
    "B03": 1.0,
    "B04": 1.0,
    "B08": 1.0,
}

class Sen2CloudDataModule(NonGeoDataModule):
    """NonGeo Sentinel-2 Cloud Cover data module implementation"""

    def __init__(
        self,
        data_root: str,
        means: list[float] | str = None,
        stds: list[float] | str = None,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = Sen2Cloud.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        constant_scale: float = 1,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        verbose: bool = False,
        train_shuffle: bool = True,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(Sen2Cloud, batch_size, num_workers, **kwargs)

        self.data_root = data_root
        self.bands = bands

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.verbose = verbose
        self.train_shuffle = train_shuffle

        if means is None: means = [MEANS[b] for b in bands]
        if stds is None: stds = [STDS[b] for b in bands]
        if self.verbose: print(f"Input standardization :\nmean - {means}\nstd - {stds}")
        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        self.aug = Normalize(means, stds)       
        self.drop_last = drop_last
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata
        
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train" and not self.train_shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )
