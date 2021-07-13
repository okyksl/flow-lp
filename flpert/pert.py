from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class Augmentation(ABC):
    def __init__(
        self,
        ord: Optional[float] = float("inf"),
        clip: Optional[float] = None,
        clamp: Optional[Tuple[float, float]] = None,
        ratio: float = 1.0,
    ) -> None:
        if ord:
            assert (
                ord == float("inf") or ord == 2
            ), "Perturbation order should be inf or 2."

        self.ratio = ratio
        self.ord = ord
        self.clip = clip
        self.clamp = clamp

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor images of size (N, C, H, W) to be perturbed.
        Returns:
            Tensor: Augmented Tensor image.
        """
        # perturb tensor
        tensor_x = self.perturb(tensor)

        # post process tensor
        tensor_x = self.post_process(tensor, tensor_x)

        #  use original vs perturbed according to given ratio
        tensor_p = torch.rand(tensor_x.shape[0], 1, 1, 1).to(tensor_x.device)

        return torch.where(tensor_p > self.ratio, tensor, tensor_x)

    def post_process(
        self, tensor: torch.Tensor, tensor_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Original tensor images
            tensor_x (Tensor): Perturbed tensor images
        Returns:
            Tensor: Clipped and clamped perturbed tensor images
        """

        # clip with distance in image space
        if self.clip is not None:
            if self.ord == float("inf"):  # L-inf
                tensor_x = torch.min(tensor_x, tensor + self.clip)
                tensor_x = torch.max(tensor_x, tensor - self.clip)
            else:
                # get perturbation
                tensor_dx = tensor_x - tensor
                tensor_dx = torch.reshape(tensor_dx, (tensor.shape[0], -1))

                #  rescale perturbation
                delta_norm = torch.linalg.norm(
                    tensor_dx, ord=self.ord, dim=1, keepdim=True
                )
                tensor_dx = tensor_dx / torch.max(
                    delta_norm / self.clip, torch.ones_like(delta_norm)
                )

                #  add scaled perturbation
                tensor_dx = torch.reshape(tensor_dx, tensor.shape)
                tensor_x = tensor + tensor_dx

        # clamp with image space boundaries
        if self.clamp is not None:
            tensor_x = torch.clamp(tensor_x, *self.clamp)

        return tensor_x

    @abstractmethod
    def perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        pass


class CleanPert(Augmentation):
    """Placeholder for no perturbation"""

    def perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class PGDPert(Augmentation):
    """PGD adversarial perturbation in image space"""

    def __init__(
        self,
        conv: torch.nn.Module = None,
        loss_fn: torch.nn.Module = None,
        num_steps: int = 1,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        random_start: bool = True,
        ord: float = 2,
        clip: Optional[float] = None,
        clamp: Optional[Tuple[float, float]] = None,
        ratio: float = 1.0,
    ) -> None:
        super().__init__(ord=ord, clip=clip, clamp=clamp, ratio=ratio)
        self.conv = conv
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start

    def perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        # use preds of the network (avoid label leaking)
        with torch.no_grad():
            preds = self.conv(tensor)
            labels = torch.argmax(preds, dim=-1)

        #  randomly init start point
        if self.random_start:
            tensor_delta = torch.empty_like(tensor, device=tensor.device).uniform_(
                -self.epsilon, self.epsilon
            )

            #  rescale if necessary
            if self.ord != float("inf"):
                tensor_delta = torch.reshape(tensor_delta, (tensor.shape[0], -1))
                delta_norm = torch.linalg.norm(
                    tensor_delta, ord=self.ord, dim=1, keepdims=True
                )
                tensor_delta /= torch.max(
                    delta_norm / self.epsilon, torch.ones_like(delta_norm)
                )
                tensor_delta = torch.reshape(tensor_delta, tensor.shape)
        else:
            tensor_delta = torch.zeros_like(tensor, device=tensor.device)

        for _ in range(self.num_steps):
            tensor_delta.requires_grad = True

            with torch.enable_grad():
                # clean gradients
                self.conv.zero_grad()

                #  feed-forward
                preds = self.conv(tensor + tensor_delta)
                loss = self.loss_fn(preds, labels)

                # get gradient
                grad = torch.autograd.grad(
                    loss, tensor_delta, retain_graph=False, create_graph=False
                )[0]

            with torch.no_grad():
                # resize for normalization
                grad = torch.reshape(grad, (tensor.shape[0], -1))
                tensor_delta = torch.reshape(tensor_delta, (tensor.shape[0], -1))

                #  normalize gradient
                if self.ord == float("inf"):
                    grad = torch.sign(grad)
                else:
                    grad = grad / torch.linalg.norm(
                        grad, ord=self.ord, dim=1, keepdims=True
                    )

                # apply perturbation
                tensor_delta += grad * self.alpha

                #  clip with L-ord epsilon
                if self.ord == float("inf"):
                    tensor_delta = torch.clamp(
                        tensor_delta, -self.epsilon, +self.epsilon
                    )
                else:
                    delta_norm = torch.linalg.norm(
                        tensor_delta, ord=self.ord, dim=1, keepdims=True
                    )
                    tensor_delta /= torch.max(
                        delta_norm / self.epsilon, torch.ones_like(delta_norm)
                    )

                #  reshape to input shape
                tensor_delta = torch.reshape(tensor_delta, tensor.shape)
                tensor_delta = tensor_delta.detach()

        #  clean gradients
        self.conv.zero_grad()

        return tensor + tensor_delta


class FlowPerturbation(Augmentation):
    """Augments a tensor image with a normalizing flow and a perturbation function.
    Given normalizing flow: ``N(x)``and perturbation function ``F(z)``, this transform
    will generate a new tensor image ``N^{-1}(F(N(x)))``
    """

    def __init__(
        self,
        flow: torch.nn.Module = None,
        epsilon: float = 0.1,
        ord: Optional[float] = None,
        clip: Optional[float] = None,
        clamp: Optional[Tuple[float, float]] = None,
        ratio: float = 1.0,
    ) -> None:
        super().__init__(ord=ord, clip=clip, clamp=clamp, ratio=ratio)
        self.flow = flow
        self.epsilon = epsilon

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(flow={0}, pert={1})".format(
            self.flow.__class__.__name__, self.pert_fn.__class__.__name__
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor images of size (N, C, H, W) to be perturbed.
        Returns:
            Tensor: Augmented Tensor image.
        """
        # get latent representation
        self.flow.eval()
        with torch.no_grad():
            tensor_z, _ = self.flow(tensor, reverse=False)

        # perturb latent code
        tensor_z = self.perturb(tensor_z)

        #  generate image
        with torch.no_grad():
            tensor_x, _ = self.flow(tensor_z, reverse=True)

        #  post process
        tensor_x = self.post_process(tensor, tensor_x)

        return tensor_x

    @abstractmethod
    def _perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    def perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_pert = self._perturb(tensor)

        if self.ord is not None:
            if self.ord == float("inf"):
                tensor_pert = torch.clamp(tensor_pert, -self.epsilon, +self.epsilon)
            else:
                tensor_pert = torch.reshape(tensor_pert, (tensor.shape[0], -1))
                pert_norm = torch.linalg.norm(
                    tensor_pert, ord=self.ord, dim=1, keepdims=True
                )
                tensor_pert /= torch.max(
                    pert_norm / self.epsilon, torch.ones_like(pert_norm)
                )
                tensor_pert = torch.reshape(tensor_pert, tensor.shape)
        return tensor + tensor_pert


class FlowRandPert(FlowPerturbation):
    """Randomized perturbation in Flow's latent space"""

    def _perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(tensor, device=tensor.device).normal_(
            mean=0, std=self.epsilon
        )


class FlowAwayPert(FlowPerturbation):
    """Randomized away perturbation (sign directing away from latent center) in Flow's latent space"""

    def _perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sign(tensor) * torch.abs(
            torch.empty_like(tensor, device=tensor.device).normal_(
                mean=0, std=self.epsilon
            )
        )


class FlowAdvPert(Augmentation):
    """Adversarial perturbation in Flow's latent space"""

    def __init__(
        self,
        flow: torch.nn.Module = None,
        conv: torch.nn.Module = None,
        loss_fn: torch.nn.Module = None,
        num_steps: int = 1,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        random_start: bool = True,
        ord: float = 2,
        clip: Optional[float] = None,
        clamp: Optional[Tuple[float, float]] = None,
        ratio: float = 1.0,
    ) -> None:
        super().__init__(ord=ord, clip=clip, clamp=clamp, ratio=ratio)
        self.flow = flow
        self.conv = conv
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start

    def perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        # use preds of the network (avoid label leaking)
        with torch.no_grad():
            preds = self.conv(tensor)
            labels = torch.argmax(preds, dim=-1)

        # get latent representation
        self.flow.eval()
        with torch.no_grad():
            tensor, _ = self.flow(tensor, reverse=False)

        #  randomly init start point
        if self.random_start:
            tensor_delta = torch.empty_like(tensor, device=tensor.device).uniform_(
                -self.epsilon, self.epsilon
            )

            #  rescale if necessary
            if self.ord != float("inf"):
                tensor_delta = torch.reshape(tensor_delta, (tensor.shape[0], -1))
                delta_norm = torch.linalg.norm(
                    tensor_delta, ord=self.ord, dim=1, keepdims=True
                )
                tensor_delta /= torch.max(
                    delta_norm / self.epsilon, torch.ones_like(delta_norm)
                )
                tensor_delta = torch.reshape(tensor_delta, tensor.shape)
        else:
            tensor_delta = torch.zeros_like(tensor, device=tensor.device)

        for _ in range(self.num_steps):
            tensor_delta.requires_grad = True

            with torch.enable_grad():
                # clean gradients
                self.flow.zero_grad()
                self.conv.zero_grad()

                #  feed-forward
                tensor_x, _ = self.flow(tensor + tensor_delta, reverse=True)
                preds = self.conv(tensor_x)
                loss = self.loss_fn(preds, labels)

                # get gradient
                grad = torch.autograd.grad(
                    loss, tensor_delta, retain_graph=False, create_graph=False
                )[0]

            with torch.no_grad():
                # resize for normalization
                grad = torch.reshape(grad, (tensor.shape[0], -1))
                tensor_delta = torch.reshape(tensor_delta, (tensor.shape[0], -1))

                #  normalize gradient
                if self.ord == float("inf"):
                    grad = torch.sign(grad)
                else:
                    grad = grad / torch.linalg.norm(
                        grad, ord=self.ord, dim=1, keepdims=True
                    )

                # apply perturbation
                tensor_delta += grad * self.alpha

                #  clip with L-ord epsilon
                if self.ord == float("inf"):
                    tensor_delta = torch.clamp(
                        tensor_delta, -self.epsilon, +self.epsilon
                    )
                else:
                    delta_norm = torch.linalg.norm(
                        tensor_delta, ord=self.ord, dim=1, keepdims=True
                    )
                    tensor_delta /= torch.max(
                        delta_norm / self.epsilon, torch.ones_like(delta_norm)
                    )

                #  reshape to input shape
                tensor_delta = torch.reshape(tensor_delta, tensor.shape)
                tensor_delta = tensor_delta.detach()

        #  clean gradients
        self.flow.zero_grad()
        self.conv.zero_grad()

        #  generate image
        with torch.no_grad():
            tensor_x, _ = self.flow(tensor + tensor_delta, reverse=True)

        return tensor_x
