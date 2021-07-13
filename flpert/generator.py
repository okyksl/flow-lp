import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader


class Generator(ABC):
    """Generates data and saves to disk"""

    def generate(self) -> None:
        """Generates samples with model"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.logger.info("Beginning generation")
        start_time = time.time()

        self._gen_loop()

        gen_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished generation! Total time: {gen_time_h:.2f}h")

    @abstractmethod
    def _gen_loop(self) -> None:
        raise NotImplementedError


class Perturber(Generator):
    """Records a data loader to disk
        seed: seed for reproducibility
        device: device to run the model on
        num_classes: # of different classes in dataset
        data_loader: data_loader
        pert_fn: perturbation function to apply
        save_path: folder in which to save samples
        make_grid: whether to output in batches or seperately
    """

    def __init__(
        self,
        seed: int,
        device: torch.device,
        num_classes: int,
        data_loader: DataLoader,
        pert_fn: Optional[Callable],
        save_path: str,
        make_grid: bool = False,
    ):
        self.seed = seed
        self.device = device
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.pert_fn = pert_fn
        self.save_path = save_path
        self.make_grid = make_grid
        self.logger = logging.getLogger()

    def _gen_loop(self) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.data_loader), leave=False)
        pbar.set_description(f"Perturber")

        #  create folders if saving seperately
        if not self.make_grid:
            for i in range(self.num_classes):
                os.makedirs(self.save_path + f"/{i}", exist_ok=True)

        # loop
        idx = 0
        for data, target in self.data_loader:
            data = data.to(self.device)

            # apply perturbation
            if self.pert_fn is not None:
                data = self.pert_fn(data)

            if self.make_grid:
                images_concat = torchvision.utils.make_grid(
                    data, nrow=int(data.shape[0] ** 0.5), padding=2, pad_value=255,
                )
                torchvision.utils.save_image(
                    images_concat, self.save_path + f"/{idx}.png"
                )
            else:
                for i in range(data.shape[0]):
                    image, label = data[i], target[i]
                    torchvision.utils.save_image(
                        image, self.save_path + f"/{label}/{idx+i}.png"
                    )
            idx += data.shape[0]
            pbar.update()

        pbar.close()


class FlowGenerator(Generator):
    """Flow sample generator

    Args:
        seed: seed for reproducibility
        device: device to run the model on
        model: model to generate samples
        dim: image dimensions
        save_path: folder in which to save samples
        checkpoint_path: path to model checkpoint, to load model,
        batch_size: batch size to use with flow
        num_samples: total number of samples to generate
        sample_temperature: temperatures to sample from flow
        make_grid: whether to output in batches or seperately
    """

    def __init__(
        self,
        seed: int,
        device: torch.device,
        model: torch.nn.Module,
        dim: List[int],
        save_path: str,
        checkpoint_path: str,
        batch_size: int = 64,
        num_samples: int = 10000,
        sample_temperature: Union[float, List[float]] = [1.0],
        make_grid: bool = False,
    ) -> None:
        self.seed = seed
        self.device = device
        self.model = model
        self.dim = dim
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.make_grid = make_grid

        if type(sample_temperature) == float:
            self.sample_temperature = [sample_temperature]
        else:
            self.sample_temperature = sample_temperature

        self.logger = logging.getLogger()
        self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(self, path: str,) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.logger.info(f"Loaded checkpoint {path} for generation!")

    def _gen_loop(self) -> None:
        self.model.eval()

        # progress bar
        len_batches = (
            self.num_samples // self.batch_size + self.num_samples % self.batch_size > 0
        )
        pbar = tqdm.tqdm(total=len_batches, leave=False)
        pbar.set_description(f"Generator")

        #  create folder
        os.makedirs(self.save_path, exist_ok=True)

        # loop
        idx = 0
        for _ in range(len_batches):
            # determine batch size
            if idx + self.batch_size > self.num_samples:
                batch_size = self.num_samples - idx
            else:
                batch_size = self.batch_size

            # random sample
            z = torch.randn(
                (batch_size, *self.dim), dtype=torch.float32, device=self.device
            )

            # loop through temperatures
            for temperature in self.sample_temperature:
                with torch.no_grad():
                    x, _ = self.model(z * temperature, reverse=True)

                    # save images
                    if self.make_grid:
                        images_concat = torchvision.utils.make_grid(
                            x, nrow=int(batch_size ** 0.5), padding=2, pad_value=255,
                        )
                        torchvision.utils.save_image(
                            images_concat,
                            self.save_path + f"/{idx}_temperature{temperature}.png",
                        )
                    else:
                        for j in range(len(images)):
                            torchvision.utils.save_image(
                                x[j],
                                self.save_path
                                + f"/{idx + j}_temperature_{temperature}.png",
                            )
            idx += batch_size

            pbar.update()

        pbar.close()
