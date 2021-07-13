import hydra
from omegaconf import DictConfig

import flpert.utils as utils
from flpert.runner import Runner


@hydra.main(config_path="conf", config_name="eval_conv")
def eval(cfg: DictConfig):
    utils.display_config(cfg)
    runner = Runner(cfg)

    # eval
    evaluator = runner.get_evaluator("conv")
    evaluator.evaluate()


if __name__ == "__main__":
    eval()
