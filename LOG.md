# log

## 2026-10-18
- set up repo structure, gitignore, and dependencies.
- downloaded mnist, verified normalisation (0.1307, 0.3081), and cached as tensors.
- implemented partition strategies in `src/partition.py`: iid, pathological shards, dirichlet (0.1, 0.5, 10), quantity skew, feature rotation.
- verified model parameter counts against paper 1: 2nn has 199,210 params, cnn has 1,663,370 params.
- centralized training: 2nn gets 97.8%, cnn gets 98.9%.
- implemented fedavg and fedsgd. confirmed that increasing local computation (e=5, b=10) cuts rounds to target accuracy compared to fedsgd.
- tested non-iid splits: pathological shards hurt convergence the most due to extreme label separation.
- reproduced figure 1 from paper 1: models starting from the same init interpolate cleanly with a loss dip at theta=0.5, while models from random inits hit an obstacle with loss above 2.4.
- ran survey demos from paper 2: cross-silo vs cross-device, client dropouts, communication bytes, and update cosine drift.
- saved metrics in `results/` and generated plots in `plots/`.
