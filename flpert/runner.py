import logging
import os
from collections import OrderedDict
from typing import Callable, Optional

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter

from flpert.evaluator import ConvEvaluator, Evaluator, FlowEvaluator
from flpert.generator import FlowGenerator, Generator, Perturber
from flpert.layers import (
    InvertibleReshapeLayer,
    InvertibleSequential,
    StandardizationLayer,
)
from flpert.loaders import get_loaders
from flpert.scheduler import ExponentialDecayLR, WarmupLR
from flpert.trainer import ConvTrainer, FlowTrainer, Trainer


class Runner:
    """Parses a hydra configuration and prepares further specific tasks"""

    def __init__(self, cfg: DictConfig) -> None:
        self.device = torch.device(cfg.misc.device)
        self.logger = logging.getLogger()

        #   resolve checkpoint paths
        if cfg.checkpoint.model is not None:
            cfg.checkpoint.model = hydra.utils.to_absolute_path(cfg.checkpoint.model)
        if cfg.checkpoint.pert.flow is not None:
            cfg.checkpoint.pert.flow = hydra.utils.to_absolute_path(
                cfg.checkpoint.pert.flow
            )
        if cfg.checkpoint.pert.conv is not None:
            cfg.checkpoint.pert.conv = hydra.utils.to_absolute_path(
                cfg.checkpoint.pert.conv
            )

        # get data loaders and perturbation functions
        self.loaders = get_loaders(cfg.dataset)

        #  get model
        self.model = self._get_model(cfg.model, cfg.dataset)

        #  get perturbation
        if cfg.get("pert", None) is not None:
            self.pert = self._get_pert(cfg.pert, cfg.dataset, cfg.checkpoint)
        else:
            self.pert = None

        #  initialize pert fncs for loaders
        self.pert_fns = (
            self.pert
            if cfg.dataset.train.use and cfg.dataset.train.transform.pert
            else None,
            self.pert
            if cfg.dataset.val.use and cfg.dataset.val.transform.pert
            else None,
            self.pert
            if cfg.dataset.test.use and cfg.dataset.test.transform.pert
            else None,
        )

        # convert warmup "samples" to warmup "steps" by dividing to batch_size
        if cfg.hparams.scheduler.get("warmup_steps", None) is not None:
            cfg.hparams.scheduler.warmup_steps /= cfg.dataset.batch_size

        # save configuration
        self.cfg = cfg

    def _get_model(
        self, model_cfg: DictConfig, dataset_cfg: DictConfig
    ) -> torch.nn.Module:
        if model_cfg.get("standardize", None) is not None:
            with open_dict(model_cfg):
                standardize = model_cfg.pop("standardize")
        else:
            standardize = False

        if model_cfg.get("flatten", None) is not None:
            with open_dict(model_cfg):
                flatten = model_cfg.pop("flatten")
        else:
            flatten = False

        if standardize:  # only for convs
            return torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "standardize",
                            StandardizationLayer(
                                dataset_cfg.mean, dataset_cfg.std, self.device
                            ),
                        ),
                        ("model", instantiate(model_cfg).to(self.device)),
                    ]
                )
            )

        if flatten:  # only for flow
            assert (
                model_cfg.num_levels == 1
            ), "Flatten can only be used with single level flows!"

            dims = [dataset_cfg.dim[0], dataset_cfg.dim[1], dataset_cfg.dim[2]]
            total_dims = dims[0] * dims[1] * dims[2]

            #  update the in_channels to match input
            model_cfg.in_channels = total_dims // 4

            return InvertibleSequential(
                OrderedDict(
                    [
                        # squeeze will flatten to a full image
                        (
                            "flatten",
                            InvertibleReshapeLayer(
                                [-1, total_dims // 4, 2, 2], [-1, *dims]
                            ),
                        ),
                        ("model", instantiate(model_cfg).to(self.device)),
                    ]
                )
            )

        return instantiate(model_cfg).to(self.device)

    def _get_pert(
        self, pert_cfg: DictConfig, dataset_cfg: DictConfig, checkpoint_cfg: DictConfig
    ) -> Callable:
        # parse ord param if needed
        if pert_cfg.get("ord", None) is not None and isinstance(pert_cfg.ord, str):
            pert_cfg.ord = float(pert_cfg.ord)

        #  load models if needed
        if pert_cfg.get("model", None) is not None:
            models = {}  # container for loaded models

            if pert_cfg.model.get("flow", None) is not None:
                # initialize flow model for perturbation

                if pert_cfg.model.flow == "auto":
                    flow = self.model  #  reuse base model if auto selection is asked
                    self.logger.info("Auto selected the flow model for perturbation!")
                else:
                    assert (
                        checkpoint_cfg.pert.flow is not None
                    ), "Provide a checkpoint for perturbation flow!"

                    flow = self._get_model(pert_cfg.model.flow, dataset_cfg)

                    checkpoint = torch.load(
                        checkpoint_cfg.pert.flow, map_location=self.device
                    )
                    flow.load_state_dict(checkpoint["model"])
                    self.logger.info(
                        f"Loaded checkpoint {checkpoint_cfg.pert.flow} for perturbation flow!"
                    )
                models["flow"] = flow

            if pert_cfg.model.get("conv", None) is not None:
                # initialize conv model for perturbation

                if pert_cfg.model.conv == "auto":
                    if pert_cfg.model.get("flow", None) is not None:
                        assert (
                            not pert_cfg.model.flow == "auto"
                        ), "Can't select flow and conv models automatically!"

                    conv = self.model  #  reuse base model if auto selection is asked
                    self.logger.info("Auto selected the conv model for perturbation!")
                else:
                    assert (
                        checkpoint_cfg.pert.conv is not None
                    ), "Provide a checkpoint for perturbation feedback conv!"

                    conv = self._get_model(pert_cfg.model.conv, dataset_cfg)

                    checkpoint = torch.load(
                        checkpoint_cfg.pert.conv, map_location=self.device
                    )
                    conv.load_state_dict(checkpoint["model"])
                    self.logger.info(
                        f"Loaded checkpoint {checkpoint_cfg.pert.conv} for perturbation feedback conv!"
                    )

                #  initialize conv & loss function for feedback
                models["conv"] = conv
                models["loss_fn"] = instantiate(pert_cfg.model.loss_fn).to(self.device)

            # initialize perturbation with models
            with open_dict(pert_cfg):
                pert_cfg.pop("model")
            return instantiate(pert_cfg, **models)

        else:
            return instantiate(pert_cfg)

    def get_trainer(self, task: str) -> Trainer:
        """Returns a trainer object"""

        #  get optimizer and scheduler
        optimizer = instantiate(self.cfg.hparams.optimizer, self.model.parameters())
        scheduler = instantiate(self.cfg.hparams.scheduler, optimizer)

        # prepare train and val loaders
        train_loader, val_loader, test_loader = self.loaders
        train_pert_fn, val_pert_fn, test_pert_fn = self.pert_fns
        if val_loader is None:
            val_loader = test_loader
            val_pert_fn = test_pert_fn
            self.logger.info("Using test set as validation set in training.")

        # get hyperparams
        epochs = self.cfg.hparams.epochs
        max_grad_norm = self.cfg.hparams.max_grad_norm
        update_sched_on_iter = (
            True
            if isinstance(scheduler, WarmupLR)
            or isinstance(scheduler, ExponentialDecayLR)
            else False
        )

        # tensorboard
        if self.cfg.misc.tensorboard:
            writer = SummaryWriter(os.getcwd())
            text = f"<pre>{OmegaConf.to_yaml(self.cfg)}</pre>"
            writer.add_text("config", text)
        else:
            writer = None

        # saving
        save_path = os.getcwd() if self.cfg.misc.save else None

        if task == "flow":
            return FlowTrainer(
                seed=self.cfg.misc.seed,
                epochs=epochs,
                device=self.device,
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                train_pert_fn=train_pert_fn,
                val_pert_fn=val_pert_fn,
                update_sched_on_iter=update_sched_on_iter,
                max_grad_norm=max_grad_norm,
                eval_freq=self.cfg.eval.freq,
                writer=writer,
                save_path=save_path,
                checkpoint_path=self.cfg.checkpoint.model,
                sample_epoch_frequency=self.cfg.misc.sample_epoch_frequency,
                sample_temperature=self.cfg.misc.sample_temperature,
            )
        elif task == "conv":
            return ConvTrainer(
                seed=self.cfg.misc.seed,
                epochs=epochs,
                device=self.device,
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                train_pert_fn=train_pert_fn,
                val_pert_fn=val_pert_fn,
                update_sched_on_iter=update_sched_on_iter,
                max_grad_norm=max_grad_norm,
                eval_freq=self.cfg.eval.freq,
                eval_top_acc=self.cfg.eval.top,
                writer=writer,
                save_path=save_path,
                checkpoint_path=self.cfg.checkpoint.model,
            )
        else:
            raise ValueError("Only supports trainers for flow and conv!")

    def get_evaluator(self, task: str) -> Evaluator:
        """Returns an evaluator object"""

        _, _, test_loader = self.loaders
        _, _, test_pert_fn = self.pert_fns

        if task == "flow":
            return FlowEvaluator(
                seed=self.cfg.misc.seed,
                device=self.device,
                model=self.model,
                data_loader=test_loader,
                checkpoint_path=self.cfg.checkpoint.model,
                pert_fn=test_pert_fn,
            )
        elif task == "conv":
            return ConvEvaluator(
                seed=self.cfg.misc.seed,
                device=self.device,
                model=self.model,
                data_loader=test_loader,
                checkpoint_path=self.cfg.checkpoint.model,
                pert_fn=test_pert_fn,
                eval_top_acc=self.cfg.eval.top,
            )
        else:
            raise ValueError("Only supports evaluators for flow and conv!")

    def get_generator(self) -> Generator:
        """Returns a generator object"""

        _, _, test_loader = self.loaders
        _, _, test_pert_fn = self.pert_fns
        save_path = os.getcwd()

        if self.cfg.checkpoint.model is not None:
            checkpoint = torch.load(self.cfg.checkpoint.model, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.logger.info(
                f"Loaded checkpoint {self.cfg.checkpoint.model} for base model!"
            )

            # save perturbed samples
            return Perturber(
                seed=self.cfg.misc.seed,
                device=self.device,
                num_classes=self.cfg.dataset.num_classes,
                data_loader=test_loader,
                pert_fn=test_pert_fn,
                save_path=save_path,
                make_grid=self.cfg.misc.make_grid,
            )
        else:
            if self.cfg.misc.get("num_samples", None) is None:
                self.logger.info(
                    '"misc.num_samples" is not provided, geneating 1000 samples'
                )
                num_samples = 1000
            else:
                num_samples = self.cfg.misc.num_samples

            # generate clean flow samples
            return FlowGenerator(
                seed=self.cfg.misc.seed,
                device=self.device,
                model=self.model,
                dim=self.cfg.dataset.dim,
                batch_size=self.cfg.dataset.batch_size,
                num_samples=num_samples,
                save_path=save_path,
                checkpoint_path=self.cfg.checkpoint.model,
                sample_temperature=self.cfg.misc.sample_temperature,
                make_grid=self.cfg.misc.make_grid,
            )
