# ref: https://github.com/dmizr/phuber/blob/master/phuber/trainer.py

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
from torch.utils.tensorboard import SummaryWriter

from flpert.metrics import AccuracyMetric, LossMetric, NLLLoss
from flpert.utils import bits_per_dims


class Trainer(ABC):
    """Abstract model trainer

    Args:
        seed: seed for reproducibility
        epochs: number of epochs
        device: device to train the model on
        model: model to train
        train_loader: training dataloader
        val_loader: validation dataloader
        optimizer: model optimizer
        scheduler: learning rate scheduler
        train_pert_fn: perturbation function to apply train data
        val_pert_fn: perturbation function to apply val data
        update_sched_on_iter: whether to call the scheduler every iter or every epoch
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        eval_freq: frequency (in epochs) of evaluation
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training
    """

    def __init__(
        self,
        seed: int,
        epochs: int,
        device: torch.device,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_pert_fn: Optional[Callable] = None,
        val_pert_fn: Optional[Callable] = None,
        update_sched_on_iter: bool = False,
        max_grad_norm: Optional[float] = None,
        eval_freq: Optional[int] = 1,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_pert_fn = train_pert_fn
        self.val_pert_fn = val_pert_fn
        self.update_sched_on_iter = update_sched_on_iter
        self.max_grad_norm = max_grad_norm
        self.eval_freq = eval_freq
        self.writer = writer
        self.save_path = save_path

        if self.eval_freq > 0:
            assert (self.val_loader is not None, "Provide val loader if eval_freq > 0!")

        self.start_epoch = 0
        self.logger = logging.getLogger()
        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(
        self,
        path: str,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"]

        if self.start_epoch > self.epochs:
            raise ValueError("Starting epoch is larger than train epochs")

        self.logger.info(f"Loaded checkpoint {path} for training!")

    def _save_model(self, path, epoch):
        obj = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(obj, os.path.join(self.save_path, path))

    def train(self) -> None:
        """Trains the model"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self._init_loop()

        self.logger.info("Beginning training")
        start_time = time.time()

        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            start_epoch_time = time.time()
            self._train_loop(epoch)

            if self.eval_freq > 0 and epoch % self.eval_freq == 0:
                self._val_loop(epoch)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time)

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(os.path.join(self.save_path, "final_model.pt"), self.epochs)

    @abstractmethod
    def _init_loop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_loop(self, epoch: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _val_loop(self, epoch: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _end_loop(self, epoch: int, epoch_time: float) -> None:
        raise NotImplementedError


class FlowTrainer(Trainer):
    """Trains a flow model"""

    def __init__(
        self,
        seed: int,
        epochs: int,
        device: torch.device,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_pert_fn: Optional[Callable],
        val_pert_fn: Optional[Callable],
        update_sched_on_iter: bool = False,
        max_grad_norm: Optional[float] = None,
        eval_freq: Optional[int] = 1,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        sample_epoch_frequency: Optional[int] = 5,
        sample_temperature: Optional[Union[float, List[float]]] = [1.0],
    ) -> None:
        super().__init__(
            seed=seed,
            epochs=epochs,
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train_pert_fn=train_pert_fn,
            val_pert_fn=val_pert_fn,
            update_sched_on_iter=update_sched_on_iter,
            max_grad_norm=max_grad_norm,
            eval_freq=eval_freq,
            writer=writer,
            save_path=save_path,
            checkpoint_path=checkpoint_path,
        )
        self.sample_epoch_frequency = sample_epoch_frequency

        if type(sample_temperature) == float:
            self.sample_temperature = [sample_temperature]
        else:
            self.sample_temperature = sample_temperature

    def _init_loop(self) -> None:
        # set loss_fn
        self.loss_fn = NLLLoss().to(self.device)

        #  set dim
        for x, _ in self.train_loader:
            self.dim = x.shape[1:]
            self.batch_size = x.shape[0]
            break

        # loss metrics
        self.train_loss = LossMetric()
        self.val_loss = LossMetric()

        self.val_best = float("inf")

    def _train_loop(self, epoch: int) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # set to train
        self.model.train()

        # start loop
        for x, _ in self.train_loader:
            # to device
            x = x.to(self.device)

            # apply perturbation
            if self.train_pert_fn is not None:
                x = self.train_pert_fn(x)

            # forward + backward
            self.optimizer.zero_grad()
            z, sldj = self.model(x, reverse=False)
            loss = self.loss_fn(z, sldj)
            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            #  steps
            self.optimizer.step()

            if self.update_sched_on_iter:
                self.scheduler.step()

            # update metrics
            self.train_loss.update(loss.item(), x.shape[0])
            loss_avg = self.train_loss.compute()

            # update progress bar
            pbar.set_postfix(
                nll=loss_avg,
                bpd=bits_per_dims(loss_avg, self.dim),
                lr=self.optimizer.param_groups[0]["lr"],
            )
            pbar.update()

        # update scheduler if it is epoch-based
        if not self.update_sched_on_iter:
            self.scheduler.step()

        pbar.close()

    def _val_loop(self, epoch: int) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.val_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # set to eval
        self.model.eval()

        # loop
        for x, _ in self.val_loader:
            with torch.no_grad():
                # to device
                x = x.to(self.device)

                # apply perturbation
                if self.val_pert_fn is not None:
                    x = self.val_pert_fn(x)

                # forward
                z, sldj = self.model(x, reverse=False)
                loss = self.loss_fn(z, sldj)

                # update metrics
                self.val_loss.update(loss.item(), x.shape[0])
                loss_avg = self.val_loss.compute()

                # update progress bar
                pbar.set_postfix(nll=loss_avg, bpd=bits_per_dims(loss_avg, self.dim))
                pbar.update()

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float):
        # print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # save model
        if self.save_path is not None:
            self._save_model(os.path.join(self.save_path, "most_recent.pt"), epoch)

        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            loss_avg = self.val_loss.compute()
            if loss_avg < self.val_best:
                self.val_best = loss_avg
                self._save_model(os.path.join(self.save_path, "best.pt"), epoch)

        # clear metrics
        self.train_loss.reset()
        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            self.val_loss.reset()

        # generate images
        if epoch % self.sample_epoch_frequency == 0:
            z = torch.randn(
                (self.batch_size, *self.dim), dtype=torch.float32, device=self.device
            )

            samples_path = os.path.join(self.save_path, "samples")
            os.makedirs(samples_path, exist_ok=True)

            for temperature in self.sample_temperature:
                with torch.no_grad():
                    x, _ = self.model(z * temperature, reverse=True)

                    images_concat = torchvision.utils.make_grid(
                        x,
                        nrow=int(self.batch_size ** 0.5),
                        padding=2,
                        pad_value=255,
                    )
                    torchvision.utils.save_image(
                        images_concat,
                        samples_path
                        + "/epoch_{}_temperature_{}.png".format(epoch, temperature),
                    )

    def _epoch_str(self, epoch: int, epoch_time: float):
        loss_avg = self.train_loss.compute()
        s = f"Epoch {epoch} "
        s += f"| Train loss: {loss_avg:.3f} "
        s += f"| Train bits: {bits_per_dims(loss_avg, self.dim):.3f} "

        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            loss_avg = self.val_loss.compute()
            s += f"| Val loss: {loss_avg:.3f} "
            s += f"| Val bits: {bits_per_dims(loss_avg, self.dim):.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"
        return s

    def _write_to_tb(self, epoch):
        loss_avg = self.train_loss.compute()
        self.writer.add_scalar("Loss/train", loss_avg, epoch)
        self.writer.add_scalar("Bits/train", bits_per_dims(loss_avg, self.dim), epoch)

        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            loss_avg = self.val_loss.compute()
            self.writer.add_scalar("Loss/val", loss_avg, epoch)
            self.writer.add_scalar("Bits/val", bits_per_dims(loss_avg, self.dim), epoch)


class ConvTrainer(Trainer):
    """Trains a convolution model"""

    def __init__(
        self,
        seed: int,
        epochs: int,
        device: torch.device,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_pert_fn: Optional[Callable] = None,
        val_pert_fn: Optional[Callable] = None,
        update_sched_on_iter: bool = False,
        max_grad_norm: Optional[float] = None,
        eval_freq: Optional[int] = 1,
        eval_top_acc: Optional[List[int]] = [1],
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            seed=seed,
            epochs=epochs,
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train_pert_fn=train_pert_fn,
            val_pert_fn=val_pert_fn,
            update_sched_on_iter=update_sched_on_iter,
            max_grad_norm=max_grad_norm,
            eval_freq=eval_freq,
            writer=writer,
            save_path=save_path,
            checkpoint_path=checkpoint_path,
        )
        self.eval_top_acc = eval_top_acc

    def _init_loop(self) -> None:
        # set loss_fn
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        #  set metrics
        self.train_loss = LossMetric()
        self.train_acc = AccuracyMetric(k=1)

        self.val_loss = LossMetric()

        self.val_accs = []
        for k in self.eval_top_acc:
            self.val_accs.append(AccuracyMetric(k=k))

        self.val_best = float("inf")

    def _train_loop(self, epoch: int) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # set to train
        self.model.train()

        # start loop
        for data, target in self.train_loader:
            # to device
            data, target = data.to(self.device), target.to(self.device)

            # apply perturbation
            if self.train_pert_fn is not None:
                data = self.train_pert_fn(data)

            # forward + backward
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.loss_fn(out, target)
            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            #  steps
            self.optimizer.step()

            if self.update_sched_on_iter:
                self.scheduler.step()

            # update metrics
            self.train_loss.update(loss.item(), data.shape[0])
            self.train_acc.update(out, target)

            # update progress bar
            pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)
            pbar.update()

        # update scheduler if it is epoch-based
        if not self.update_sched_on_iter:
            self.scheduler.step()

        pbar.close()

    def _val_loop(self, epoch: int) -> None:
        # progress bar
        pbar = tqdm.tqdm(total=len(self.val_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # set to eval
        self.model.eval()

        # loop
        for data, target in self.val_loader:
            with torch.no_grad():
                # to device
                data, target = data.to(self.device), target.to(self.device)

                # apply perturbation
                if self.val_pert_fn is not None:
                    data = self.val_pert_fn(data)

                # forward
                out = self.model(data)
                loss = self.loss_fn(out, target)

                # update metrics
                self.val_loss.update(loss.item(), data.shape[0])
                for val_acc in self.val_accs:
                    val_acc.update(out, target)

                # update progress bar
                pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)
                pbar.update()

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float):
        # print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # save model
        if self.save_path is not None:
            self._save_model(os.path.join(self.save_path, "most_recent.pt"), epoch)

        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            loss_avg = self.val_loss.compute()
            if loss_avg < self.val_best:
                self.val_best = loss_avg
                self._save_model(os.path.join(self.save_path, "best.pt"), epoch)

        # clear metrics
        self.train_loss.reset()
        self.train_acc.reset()
        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            self.val_loss.reset()
            for val_acc in self.val_accs:
                val_acc.reset()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train loss: {self.train_loss.compute():.3f} "
        s += f"| Train acc: {self.train_acc.compute():.3f} "
        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            s += f"| Val loss: {self.val_loss.compute():.3f} "
            for val_acc in self.val_accs:
                s += f"| Top-{val_acc.k} acc: {val_acc.compute():.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"
        return s

    def _write_to_tb(self, epoch):
        self.writer.add_scalar("Loss/train", self.train_loss.compute(), epoch)
        self.writer.add_scalar("Accuracy/train", self.train_acc.compute(), epoch)

        if self.eval_freq > 0 and epoch % self.eval_freq == 0:
            self.writer.add_scalar("Loss/val", self.val_loss.compute(), epoch)
            for val_acc in self.val_accs:
                self.writer.add_scalar(
                    f"Accuracy/val/top-{val_acc.k}", val_acc.compute(), epoch
                )
