import hydra
from omegaconf import DictConfig

import flpert.utils as utils
from flpert.runner import Runner


@hydra.main(config_path="conf", config_name="train_flow")
def train(cfg: DictConfig):
    utils.display_config(cfg)
    runner = Runner(cfg)

    #  train
    trainer = runner.get_trainer("flow")
    trainer.train()


if __name__ == "__main__":
    train()
