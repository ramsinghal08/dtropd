# D-TrOPD: Decentralized Trust-Region Policy Distillation

A PyTorch implementation of D-TrOPD for networked multi-agent reinforcement
learning, with baselines (IPPO, MAPPO, HAPPO, naive distillation), an
instrumented cooperative-capture environment, and a pre-registered experiment
suite.

Each agent trains with PPO on its own rollouts and additionally distills from
its graph neighbours. The amount it imports from each neighbour is gated by a
trust weight computed from their behavioural divergence, measured as a
closed-form reverse KL between tanh-Gaussian policies. There is no central
critic and no sequential update order.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer. All experiments run on CPU; `train.device=cuda` moves the
networks to a single GPU.

## Usage

```bash
python run.py --help
python run.py validate --config experiments/configs/ippo_homogeneous.yaml
python run.py smoke                                  # 2-agent CPU run, under 2 minutes
python run.py smoke --algo dtropd
python run.py train --config experiments/configs/dtropd_heterogeneous.yaml --seeds 3
python run.py eval  --config experiments/configs/dtropd_heterogeneous.yaml \
    --checkpoint runs/dtropd_heterogeneous/dtropd_seed0/checkpoint_final.pt
python run.py analyze --run runs/dtropd_heterogeneous/dtropd_seed0
```

Any config value can be overridden from the command line:

```bash
python run.py train --config experiments/configs/smoke.yaml \
    --set train.total_epochs=20 --set ppo.lr=1e-3
```

Run the test suite with `pytest`.

## Algorithms

| name | critic | update order | distillation |
|---|---|---|---|
| `ippo` | local | simultaneous | none |
| `mappo` | centralised | simultaneous | none |
| `happo` | local | sequential, random order | none |
| `naive_distill` | local | simultaneous | all neighbours, unit weight |
| `dtropd` | local | simultaneous | trust-gated by divergence |

## Repository layout

```
dtropd/          library: environment, networks, algorithms, training, analysis
experiments/     YAML configs, calibrations, analysis and acceptance scripts
tests/           unit, invariant and regression tests
docs/            specification, implementation notes, theory notes, papers
figures/         generated figures
```

## Documentation

* [`docs/SPEC.md`](docs/SPEC.md): algorithm specification, invariants and
  pre-registered hypotheses.
* [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md): design
  choices, calibration procedure, environment details, and validation results.
* [`docs/THEORY_NOTES.md`](docs/THEORY_NOTES.md): findings that bear on the
  theoretical claims.
* [`docs/theory.pdf`](docs/theory.pdf) and
  [`docs/empirical_plan.pdf`](docs/empirical_plan.pdf): the theory write-up and
  the empirical plan.

## Reproducibility

Every run writes `metadata.json` (config, git commit, seed, package versions),
`metrics.jsonl` (one record per epoch) and a final checkpoint. All randomness is
derived from a single master seed, so a run is reproducible from its config and
seed alone.

## Author

Ram Singhal
