# Flow Latent Perturbations

For MNIST and FMNIST experiments, we use the Jupyter notebook provided under the folder `notebook`.

For CIFAR-10/100, we use [Hydra](https://hydra.cc/) framework to manage experiments.
Below, we describe the usage of this Hydra-based experimentation suite.
Note that you can train and run experiments on MNIST and FMNIST with Hydra as well.
However, the conditional normalizing flow used in the paper is only implemented in the provided notebook.

## Usage

Use `train_flow.py` to train normalizing flows and `train_conv.py` to train convolutional classifiers.

You can provide the following options:

* the dataset, `mnist, fmnist, cifar10, cifar100`
* the model, `lenet, standard, resnet18, resnet20, resnet50, glow`
* the perturbation, `clean, away, rand, adv`

where `standard` uses the classifier in FMNIST experiments. 

Note that hyperparameters are automatically changed according to dataset and model. See `conf/hparams/cifar10_resnet18.yaml` for an example. If the hyperparameter configuration file for the dataset-model combination provided does not exist, Hydra will raise an error. In such case, use the example configuration files to prepare a new file named `{dataset}_{model}.yaml` under folder `conf/hparams`.

You can provide checkpoint with `checkpoint.model` option. For perturbations, normalizing flow or convolution network checkpoints can be provided with `checkpoint.model.flow` and `checkpoint.model.conv` respectively. 

Model types for perturbations can be provided with `model@pert.model.flow` or `model@pert.model.conv` but the specific model arguments need to be provided by hand (such as `pert.model.flow.in_channels=1` for `MNIST`). To receive the convolution feedback directly from the training model, use `auto` option for `pert.model.conv`.

For evaluating models, use `eval_flow.py` and `eval_conv.py`. You can also generate and save normalizing flow samples with `gen_flow.py`.

For more details on the provided options and hyperparams, use `--help` with any of the commands:

```
python train_flow.py --help
```

## Examples

* Train a Glow on CIFAR-10.

```
python train_flow.py dataset=cifar10 model=glow
```

* Train a ResNet-18 on CIFAR-10.

```
python train_flow.py dataset=cifar10 model=resnet18 pert=clean
```

* Train a ResNet-18 with randomized latent perturbations on CIFAR-10.

```
python train_conv.py dataset=cifar10 model=resnet18 pert=rand pert.ord=2 pert.epsilon=20.0 model@pert.model.flow=glow checkpoint.pert.flow=...
```

* Train a ResNet-18 with adversarial latent perturbations on CIFAR-10.

```
python train_conv.py dataset=cifar10 model=resnet18 pert=adv pert.ord=2 pert.epsilon=1.0 pert.alpha=0.5 pert.num_steps=3 model@pert.model.flow=glow checkpoint.pert.flow=...
```

* Train a 12-step Glow on MNIST or FMNIST.

```
python train_flow.py dataset=mnist model=glow model.num_levels=12 model.num_levels=1 model.in_channels=1
```

* Train a LeNet-5 with adversarial latent perturbations on MNIST.

```
python train_conv.py dataset=mnist model=lenet pert=adv pert.num_steps=5 pert.epsilon=0.3 pert.alpha=0.1 pert.ord=inf model@pert.model.flow=glow pert.model.flow.in_channels=1 pert.model.flow.num_levels=12 pert.model.flow.num_levels=1 checkpoint.pert.flow=...
```

## Output

Hydra will create outputs under the folder `outputs/run/{task}/{dataset}/{timestamp}` that includes all experiment parameters, logs, tensorboard files and (if training) model checkpoints.