# opcc-baselines

Code for baselines presented in research paper "Offline Policy Comparison with Confidence:
Baseline and Benchmarks"

[![Python application](https://github.com/koulanurag/opcc-baselines/actions/workflows/python-app.yml/badge.svg)](https://github.com/koulanurag/opcc-baselines/actions/workflows/python-app.yml)
![License](https://img.shields.io/github/license/koulanurag/opcc-baselines)

## Installation

#### 1. Setup [opcc](https://github.com/koulanurag/opcc)
   _(We recommend familarizing with [opcc usage](https://github.com/koulanurag/opcc#usage) before using baselines)_

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

#### Train dynamics:

Example: 
```bash
python main.py --job train-dynamics --env-name d4rl:maze2d-open-v0 --dataset 1m --num-ensemble 10
```

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
| `--constant-prior-scale`                        | scale for constant priors (default: 0)                                         |


#### Query Evaluation:

Example:
- Restoring dynamics locally:
```bash
python main.py --job evaluate-queries --env-name d4rl:maze2d-open-v0 --dataset 1m --num-ensemble 10
```
- Restoring dynamics from wandb(if used in train-dynamics phase):
```bash
python main.py --job evaluate-queries --restore-dynamics-from-wandb --wandb-dynamics-run-path <username>/<project-name>/<run-id>
```

| Optional Arguments              | Description                                                                                             |
|:--------------------------------|:--------------------------------------------------------------------------------------------------------|
| `--restore-dynamics-from-wandb` | restore model from wandb run (default: False)                                                           |
| `--wandb-dynamics-run-path `      | wandb run id if restoring model (default: None)                                                         |
| `--mixture`                     | If enabled, randomly select ensemble models at<br/> each step of query evaluation<br/> (default: False) |
| `--eval-runs`                   | run count for each query evaluation (default: 1)                                                        |
| `--eval-batch-size`             | batch size for query evaluation (default: 128)                                                          |
| `--clip-obs`                    | clip the observation space with bounds for <br/>query evaluation (default: False)                       |
| `--clip-reward`                 | clip the reward with dataset bounds for <br/>query evaluation (default: False)                          |

#### Uncertainty-test :


Example:
- Restoring query evaluation data locally:
```bash
python main.py --job uncertainty-test --env-name d4rl:maze2d-open-v0 --dataset 1m --num-ensemble 10
```
- Restoring query evaluation data from wandb(If used in query-evaluation phase):
```bash
python main.py --job uncertainty-test --restore-query-eval-data-from-wandb --wandb-query-eval-data-run-path <username>/<project-name>/<run-id>
```

| Optional Arguments                                                                                                      | Description                                           |
|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|
| `--uncertainty-test-type`<br/>`{paired-confidence-interval,`<br/>`unpaired-confidence-interval,`<br/>`ensemble-voting}` | type of uncertainty test  (default:ensemble-voting)   |
| `--restore-query-eval-data-from-wandb`                                                                                  | get query evaluation data from wandb (default: False) |
| `--wandb-query-eval-data-run-path`                                                                                        | wandb run id having query eval data (default: None)   |

## Reproducibility

1. **Dynamics training** results can be found over [here](https://wandb.ai/koulanurag/opcc-baselines-train-dynamics) and corresponding commands for different configurations can be retrieved using following snippet.

```python
import json
import wandb

api = wandb.Api()
runs = api.runs(path='koulanurag/opcc-baselines-train-dynamics',
filters={"config.env_name": "HalfCheetah-v2",
         "config.dataset_name": "random", 
         "config.deterministic": True, # options: [True, False]
         "config.dynamics_type": "feed-forward", # options : ["feed-forward","autoregressive"]
         "config.constant_prior_scale": 5, # options: [0,5]
         "config.normalize": True, # options : [True, False]
         "config.dynamics_seed": 0 # options: [0,1,2,3,4]
         })

for run in runs:
    meta = json.load(run.file("wandb-metadata.json").download(replace=True))
    program = ["python"] + [meta["program"]] + meta["args"]
    print(" ".join(program))

```

2. **Query Evaluation** results can be found [here](https://wandb.ai/koulanurag/opcc-baselines-evaluate-queries) and corresponding commands can be retreived using following snippet.

```python
import json
import wandb

api = wandb.Api()
runs = api.runs(path='koulanurag/opcc-baselines-evaluate-queries',
filters={"config.env_name": "HalfCheetah-v2",
         "config.dataset_name": "random", 
         "config.deterministic": True, # options: [True, False]
         "config.dynamics_type": "feed-forward", # options : ["feed-forward","autoregressive"]
         "config.constant_prior_scale": 5, # options: [0,5]
         "config.normalize": True, # options : [True, False]
         "config.dynamics_seed": 0, # options: [0,1,2,3,4]
         "config.clip_obs": True, # options: [True]
         "config.clip_reward": True # options: [True]
         })

for run in runs:
    meta = json.load(run.file("wandb-metadata.json").download(replace=True))
    program = ["python"] + [meta["program"]] + meta["args"]
    print(" ".join(program))
    
    # if you run the printed command , it will download
    # pre-trained dynamics and run query evaluation using it.

```


3. **Uncertainty Test** results can be found [here](https://wandb.ai/koulanurag/opcc-baselines-uncertainty-test) and corresponding commands can be retrieved using following snippet.

```python
import json
import wandb

api = wandb.Api()
runs = api.runs(path='koulanurag/opcc-baselines-uncertainty-test',
filters={"config.env_name": "HalfCheetah-v2",
         "config.dataset_name": "random", 
         "config.deterministic": True, # options: [True, False]
         "config.dynamics_type": "feed-forward", # options : ["feed-forward","autoregressive"]
         "config.constant_prior_scale": 5, # options: [0,5]
         "config.normalize": True, # options : [True, False]
         "config.dynamics_seed": 0, # options: [0,1,2,3,4]
         "config.clip_obs": True, # options: [True]
         "config.clip_reward": True, # options: [True]
         "config.uncertainty_test_type": "ensemble-voting" # options: [ensemble-voting, paired-confidence-interval, unpaired-confidence-interval]
         })

for run in runs:
    meta = json.load(run.file("wandb-metadata.json").download(replace=True))
    program = ["python"] + [meta["program"]] + meta["args"]
    print(" ".join(program))
    
    # if you run the printed command , it will download
    # query-evaluation results and run uncertainty-tests over
    # them to report opcc metrics.
```

## Contact

If you have any questions or suggestions , you can contact me at koulanurag@gmail.com or open an issue on this GitHub repository.
