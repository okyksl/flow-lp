import hydra
from omegaconf import DictConfig

import flpert.utils as utils
from flpert.runner import Runner


@hydra.main(config_path="conf", config_name="gen_flow")
def gen(cfg: DictConfig):
    utils.display_config(cfg)
    runner = Runner(cfg)

    # generate
    generator = runner.get_generator()
    generator.generate()


if __name__ == "__main__":
    gen()
