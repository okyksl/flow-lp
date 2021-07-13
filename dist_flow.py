import hydra
import torch
import tqdm
from omegaconf import DictConfig

import flpert.utils as utils
from flpert.generator import Perturber
from flpert.runner import Runner


@hydra.main(config_path="conf", config_name="dist_flow")
def dist(cfg: DictConfig):
    utils.display_config(cfg)
    runner = Runner(cfg)

    # generate
    generator = runner.get_generator()
    assert isinstance(
        generator, Perturber
    ), "Need a perturbation to map distances between domains"

    # parse ord param if needed
    if isinstance(cfg.ord, float) or isinstance(cfg.ord, str):
        cfg.ord = [cfg.ord]

    for i in range(len(cfg.ord)):
        if isinstance(cfg.ord[i], str):
            cfg.ord[i] = float(cfg.ord[i])

    # progress bar
    pbar = tqdm.tqdm(total=len(generator.data_loader), leave=False)
    pbar.set_description(f"Perturber")

    #  loop through data
    dist = [[] for i in range(len(cfg.ord))]
    for data, target in generator.data_loader:
        data = data.to(generator.device)

        # apply perturbation
        if generator.pert_fn is not None:
            data_pert = generator.pert_fn(data)

        # measure distance
        diff = data - data_pert
        diff = torch.reshape(diff, (data.shape[0], -1))

        #  save dist
        dic = {}
        for i in range(len(cfg.ord)):
            norm = torch.linalg.norm(diff, ord=cfg.ord[i], dim=1)
            dic[str(cfg.ord[i])] = norm.mean().item()
            dist[i].append(norm)

        pbar.set_postfix(**dic)
        pbar.update()

    pbar.close()

    #  accumulate and report mean
    for i in range(len(cfg.ord)):
        dist[i] = torch.cat(dist[i])
        mean, std = dist[i].mean().item(), dist[i].std().item()
        generator.logger.info(f"L-{cfg.ord[i]} distance: {mean}±{std}")


if __name__ == "__main__":
    dist()
