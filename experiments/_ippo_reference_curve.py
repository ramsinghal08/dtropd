"""Emit an IPPO learning curve. Runs under both the Phase 2 and current trees."""
import json, sys, os
sys.path.insert(0, os.getcwd())
from dtropd.config import Config
from dtropd.train import train
from dtropd.utils import read_metrics

cfg = Config()
cfg.env.n_agents = 6
cfg.env.n_targets = 2
cfg.env.capture_group_size = 3
cfg.env.max_cycles = 25
cfg.env.local_ratio = 0.5
cfg.network.hidden_sizes = [32, 32]
cfg.train.algo = "ippo"
cfg.train.num_envs = 8
cfg.train.total_epochs = 12
cfg.train.eval_interval = 0
cfg.train.eval_episodes = 4
cfg.ppo.lr = 1e-3
cfg.train.log_dir = sys.argv[1]
cfg.validate(emit=False)
r = train(cfg, seed=0, progress=False)
recs = read_metrics(r.run_dir / "metrics.jsonl")
KEYS = ["episode_return_mean","explained_variance","update_kl","value_loss",
        "policy_loss","entropy","clip_fraction","inner_epochs_used","mean_sigma"]
curve = {k: [rec[k] for rec in recs] for k in KEYS}
json.dump({"config_summary": {"n_agents":6,"n_targets":2,"max_cycles":25,"num_envs":8,
                              "epochs":12,"seed":0,"hidden":[32,32],"lr":1e-3},
           "curve": curve}, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2])
