# opcc-baselines

Code for baselines presented in research paper "Offline Policy Comparison with Confidence:
Baseline and Benchmarks"

![Python package](https://github.com/koulanurag/opcc-baselines/workflows/Python%20package/badge.svg)
![License](https://img.shields.io/github/license/koulanurag/opcc-baselines)

## Installation

#### 1. Setup [opcc](https://github.com/koulanurag/opcc)

#### 2. Python dependencies could be installed using:

```bash
git clone https://github.com/koulanurag/opcc-baselines.git
cd opcc-baselines
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage

| Required Arguments                                         | Description         |
|:-----------------------------------------------------------|:--------------------|
| `--job {train-dynamics,evaluate-queries,uncertainty-test}` | Job to be performed |

| Optional Arguments     | Description                                            |
|:-----------------------|:-------------------------------------------------------|
| `--no-cuda`            | no cuda usage (default: False)                         |
| `--result-dir`         | directory to store results (default: pwd)              |
| `--wandb-project-name` | name of the wandb project (default: opcc-baselines)    |
| `--use-wandb`          | use Weight and bias visualization lib (default: False) |

#### Train dynamics

| Optional Arguments                              | Description                                                                    |
|:------------------------------------------------|:-------------------------------------------------------------------------------|
| `--env-name`                                    | name of the environment (default: HalfCheetah-v2)                              |
| `--dataset-name`                                | name of the dataset (default: random)                                          |
| `--dynamics-type {feed-forward,autoregressive}` | type of dynamics model (default: feed-forward)                                 |
| `--deterministic`                               | if True, we use deterministic model otherwise<br/> stochastic (default: False) |
| `--dynamics-seed`                               | seed for training dynamics (default: 0)                                        |
| `--log-interval`                                | log interval for training dynamics (default: 1)                                |
| `--dynamics-checkpoint-interval`                | update interval to save dynamics checkpoint (default:1)                        |
| `--hidden-size`                                 | hidden size for Linear Layers (default: 200)                                   |
| `--update-count`                                | total batch update count for training (default: 100)                           |
| `--dynamics-batch-size`                         | batch size for Dynamics Learning (default: 256)                                |
| `--dynamics-checkpoint-interval`                | update interval to save dynamics checkpoint (default:1)                        |
| `--reward-loss-coeff`                           | reward loss coefficient for training (default: 1)                              |
| `--observation-loss-coeff`                      | obs. loss coefficient for training (default: 1)                                |
| `----num-ensemble`                              | number of dynamics for ensemble (default: 1)                                   |
| `--constant-prior-scale `                       | scale for constant priors (default: 0)                                         |


## Reproducibility

Please refer to Wiki for complete list of commands for each environment
