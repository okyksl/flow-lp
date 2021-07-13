import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from flpert.metrics import AccuracyMetric, LossMetric, NLLLoss
from flpert.utils import bits_per_dims


class Evaluator(ABC):
    """Abstract model evaluator

    Args:
        seed: seed for reproducibility
        device: device to train the model on
        model: model to train
        data_loader: evaluation data loader
        checkpoint_path: path to model checkpoint
        pert_fn: perturbation function to apply data
    """

    def __init__(
        self,
        seed: int,
        device: torch.device,
        model: torch.nn.Module,
        data_loader: DataLoader,
        checkpoint_path: str,
        pert_fn: Optional[Callable] = None,
    ) -> None:
        self.seed = seed
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.pert_fn = pert_fn

        self.logger = logging.getLogger()
        self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(
        self,
        path: str,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.logger.info(f"Loaded checkpoint {path} for evaluation!")

    def evaluate(self) -> None:
        """Evaluate the model"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self._init_loop()

        self.logger.info("Beginning evaluation")
        start_time = time.time()

        self._eval_loop()

        eval_time_h = (time.time() - start_time) / 3600
        self._end_loop()
        self.logger.info(f"Finished evaluation! Total time: {eval_time_h:.2f}h")

    @abstractmethod
    def _init_loop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _eval_loop(self, epoch: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _end_loop(self, epoch: int, epoch_time: float) -> None:
        raise NotImplementedError


class FlowEvaluator(Evaluator):
    """Evaluates a flow model"""

    def _init_loop(self) -> None:
        # set loss_fn
        self.loss_fn = NLLLoss().to(self.device)

        #  set dim
        for x, _ in self.data_loader:
            self.dim = x.shape[1:]
            self.batch_size = x.shape[0]
            break

        # loss metrics
        self.eval_loss = LossMetric()

    def _eval_loop(self) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.data_loader), leave=False)
        pbar.set_description(f"Evaluation")

        # set to eval
        self.model.eval()

        # loop
        for x, _ in self.data_loader:
            with torch.no_grad():
                # to device
                x = x.to(self.device)

                # apply perturbation
                if self.pert_fn is not None:
                    x = self.pert_fn(x)

                # forward
                z, sldj = self.model(x, reverse=False)
                loss = self.loss_fn(z, sldj)

                # update metrics
                self.eval_loss.update(loss.item(), x.shape[0])
                loss_avg = self.eval_loss.compute()

                # update progress bar
                pbar.set_postfix(nll=loss_avg, bpd=bits_per_dims(loss_avg, self.dim))
                pbar.update()

        pbar.close()

    def _end_loop(self):
        # print results
        self.logger.info(self._eval_str())

        # clear metrics
        self.eval_loss.reset()

    def _eval_str(self):
        loss_avg = self.eval_loss.compute()
        s = f"Evaluation "
        s += f"| Loss: {loss_avg:.3f} "
        s += f"| Bits: {bits_per_dims(loss_avg, self.dim):.3f} "
        return s


class ConvEvaluator(Evaluator):
    """Evaluates a conv model"""

    def __init__(
        self,
        seed: int,
        device: torch.device,
        model: torch.nn.Module,
        data_loader: DataLoader,
        checkpoint_path: str,
        pert_fn: Optional[Callable] = None,
        eval_top_acc: Optional[List[int]] = [1],
    ) -> None:
        super().__init__(
            seed=seed,
            device=device,
            model=model,
            data_loader=data_loader,
            checkpoint_path=checkpoint_path,
            pert_fn=pert_fn,
        )
        self.eval_top_acc = eval_top_acc

    def _init_loop(self) -> None:
        # set loss_fn
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        #  set metrics
        self.eval_loss = LossMetric()
        self.eval_accs = []
        for k in self.eval_top_acc:
            self.eval_accs.append(AccuracyMetric(k=k))

    def _eval_loop(self) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.data_loader), leave=False)
        pbar.set_description(f"Evaluation")

        # set to eval
        self.model.eval()

        # loop
        for data, target in self.data_loader:
            with torch.no_grad():
                # to device
                data, target = data.to(self.device), target.to(self.device)

                # apply perturbation
                if self.pert_fn is not None:
                    data = self.pert_fn(data)

                # forward
                out = self.model(data)
                loss = self.loss_fn(out, target)

                # update metrics
                self.eval_loss.update(loss.item(), data.shape[0])
                for eval_acc in self.eval_accs:
                    eval_acc.update(out, target)

                # update progress bar
                pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)
                pbar.update()

        pbar.close()

    def _end_loop(self):
        # print results
        self.logger.info(self._eval_str())

        # clear metrics
        self.eval_loss.reset()
        for eval_acc in self.eval_accs:
            eval_acc.reset()

    def _eval_str(self):
        s = f"Evaluation "
        s += f"| Loss: {self.eval_loss.compute():.3f} "
        for eval_acc in self.eval_accs:
            s += f"| Top-{eval_acc.k} acc: {eval_acc.compute():.3f} "
        return s
