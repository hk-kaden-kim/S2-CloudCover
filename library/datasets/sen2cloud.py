import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import torch
import rasterio
import xarray
import xrspatial.multispectral as ms

from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray, concat
from torchgeo.datasets.errors import RGBBandsMissingError

from terratorch.datasets.utils import default_transform, validate_bands

class Sen2Cloud(NonGeoDataset):
    """NonGeo dataset implementation for Sentinel 2 Cloud Cover Dataset."""
    all_band_names = (
        "B02",     #  BLUE
        "B03",    #  GREEN
        "B04",      #  RED
        "B08",      #  NIR
    )

    rgb_bands = ("B04", "B03", "B02")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 2
    splits = {"train": "public", "val": "public", "test": "private"}

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        constant_scale: float = 1,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = 0,
        use_metadata: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            split (str): one of 'train', 'val' or 'test'.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the corresponding data module,
                should not include normalization. Defaults to None, which applies ToTensorV2().
            constant_scale (float): Factor to multiply image values by. Defaults to 0.0001.
            no_data_replace (float | None): Replace nan values in input images with this value.
                If None, does no replacement. Defaults to 0.
            no_label_replace (int | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            use_metadata (bool): whether to return metadata info (time and location).
        """
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {self.splits}."
            raise ValueError(msg)
        self.sub_dir = self.splits[split]
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)

        self.image_files = pd.read_csv(os.path.join(self.data_root, self.sub_dir,
                                                    f"{self.split}_metadata.csv"))
        
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata
        self.metadata = None

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index: int) -> torch.Tensor:
        date = pd.to_datetime(self.image_files.iat[index, 2], 
                              format='%Y-%m-%dT%H:%M:%SZ')
        return torch.tensor([[date.year, date.dayofyear - 1]], dtype=torch.float32)  # (n_timesteps, coords)

    def _get_coords(self, index: int) -> torch.Tensor:
        chip_id = self.image_files.iat[index, 0]
        file_path = os.path.join(self.data_root, self.sub_dir, 
                                 f"{self.split if self.split!='val' else 'train'}_labels", 
                                 f"{chip_id}.tif")
        with rasterio.open(file_path) as data:
            lon, lat = data.lnglat()
        lat_lon = np.asarray([lat, lon])
        return torch.tensor(lat_lon, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_sample(index, nan_replace=self.no_data_replace)

        location_coords, temporal_coords = None, None
        if self.use_metadata:
            location_coords = self._get_coords(index)
            temporal_coords = self._get_date(index)

        # to channels last
        image = image.to_numpy()
        image = np.moveaxis(image, 0, -1)

        # filter bands
        image = image[..., self.band_indices]

        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "mask": self._load_mask(index, nan_replace=self.no_label_replace).to_numpy(),
        }
        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()

        if self.use_metadata:
            output["location_coords"] = location_coords
            output["temporal_coords"] = temporal_coords

        return output

    def _load_sample(self, index: int, nan_replace: int | float | None = None) -> DataArray:
        """Load all samples.
        Returns:
            a tensor of stacked source image data
        """
        chip_id = self.image_files.iat[index, 0]
        file_path = os.path.join(self.data_root, self.sub_dir, 
                                 f"{self.split if self.split!='val' else 'train'}_features", 
                                 f"{chip_id}")
        samples = []
        for band in self.bands:
            data_path = os.path.join(file_path, f"{band}.tif")
            data = rioxarray.open_rasterio(data_path).fillna(nan_replace)
            samples.append(data[0,:,:])

        return concat(samples,dim="new_dim")
    
    def _load_mask(self, index: int, nan_replace: int | float | None = None) -> DataArray:
        """Load mask.
        Returns:
            a tensor of the label image data
        """
        chip_id = self.image_files.iat[index, 0]
        file_path = os.path.join(self.data_root, self.sub_dir, 
                            f"{self.split if self.split!='val' else 'train'}_labels", 
                            f"{chip_id}.tif")
        data = rioxarray.open_rasterio(file_path).fillna(nan_replace)
        return data[0,:,:]

    def plot(self, sample: dict[str, Tensor], show_titles: bool = True, 
             suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        n_cols = 3      # Train / val / Test

        image, mask = sample['image'], sample['mask']

        RGB = ms.true_color(r=xarray.DataArray(image[rgb_indices[0],:,:].numpy(), dims=["y", "x"]), 
                            g=xarray.DataArray(image[rgb_indices[1],:,:].numpy(), dims=["y", "x"]), 
                            b=xarray.DataArray(image[rgb_indices[2],:,:].numpy(), dims=["y", "x"]))
        NIR = image[-1,:,:]
        
        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 10))
        
        axs[0].imshow(RGB.data[:,:,:-1])
        axs[0].axis('off')
        axs[1].imshow(NIR)
        axs[1].axis('off')
        axs[2].imshow(mask, vmin=0, vmax=1)
        axs[2].axis('off')

        if show_titles:
            axs[0].set_title('RGB\n(B4,B3,B2)')
            axs[1].set_title('NIR\n(B8)')
            axs[2].set_title('Mask\n(0:No, 1:Cloud)')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

