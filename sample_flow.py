import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision
from omegaconf import DictConfig

import flpert.utils as utils
from flpert.pert import CleanPert, FlowAdvPert, FlowAwayPert, FlowRandPert, PGDPert
from flpert.runner import Runner


@hydra.main(config_path="conf", config_name="sample_flow")
def gen(cfg: DictConfig):
    utils.display_config(cfg)
    runner = Runner(cfg)

    # generate
    generator = runner.get_generator()

    #  get models
    flow = generator.pert_fn.flow
    conv = generator.pert_fn.conv
    loss_fn = generator.pert_fn.loss_fn

    # set params
    ord = 2
    epsilon = 10.0
    pgd_epsilon, pgd_alpha, pgd_k = 2.0, 0.5, 10
    adv_epsilon, adv_alpha, adv_k = 2.0, 0.5, 2

    perts = [
        CleanPert(),
        PGDPert(
            conv=conv,
            loss_fn=loss_fn,
            epsilon=pgd_epsilon,
            alpha=pgd_alpha,
            num_steps=pgd_k,
            ord=ord,
            clamp=[0, 1],
        ),
        FlowRandPert(flow=flow, epsilon=epsilon, ord=ord, clamp=[0, 1]),
        FlowAwayPert(flow=flow, epsilon=epsilon, ord=ord, clamp=[0, 1]),
        FlowAdvPert(
            flow=flow,
            conv=conv,
            loss_fn=loss_fn,
            epsilon=adv_epsilon,
            alpha=adv_alpha,
            num_steps=adv_k,
            ord=ord,
            clamp=[0, 1],
        ),
    ]

    legend = [
        "$P_{clean}$",
        "$P_{pgd}^{\ell_" + f"{ord}" + "}(\epsilon=" + f"{pgd_epsilon})$",
        "$P_{rand}^{\ell_" + f"{ord}" + "}(\epsilon=" + f"{epsilon})$",
        "$P_{away}^{\ell_" + f"{ord}" + "}(\epsilon=" + f"{epsilon})$",
        "$P_{adv}^{\ell_" + f"{ord}" + "}(\epsilon=" + f"{adv_epsilon})$",
    ]

    sns.set_context("poster")
    sns.set_style("darkgrid")

    nsamples = 3
    fig, ax = plt.subplots(
        nrows=nsamples, ncols=len(perts), figsize=(3 * len(perts), 3 * nsamples)
    )
    ax = np.reshape(ax, (nsamples, len(perts)))

    flow.eval()
    for data, target in generator.data_loader:
        data = data.to(generator.device)

        for j, pert in enumerate(perts):
            pert_data = pert(data[:nsamples, ...])

            for i in range(nsamples):
                image = pert_data[i].cpu()
                ax[i][j].imshow(np.transpose(image, (1, 2, 0)))
                ax[i][j].set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])

                if i == 0:
                    ax[i][j].set_title(legend[j])
        break

    fig.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("samples.png", pad_inches=0)


if __name__ == "__main__":
    gen()
