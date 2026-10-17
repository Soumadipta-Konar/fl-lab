# fl-lab

Implementation and empirical reproduction of two foundational federated learning papers:

1. **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (McMahan et al., AISTATS 2017)
2. **FL Survey & Taxonomy**: [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) (Kairouz et al., 2021)

---

## Structure

```
fl-lab/
├── notebooks/
│   ├── data_and_partitions.ipynb
│   ├── centralized_baseline.ipynb
│   ├── fedsgd_fedavg.ipynb
│   ├── noniid.ipynb
│   ├── common_init.ipynb
│   ├── survey_demos.ipynb
│   └── plots.ipynb
├── src/
│   ├── partition.py     # iid, shards, dirichlet, quantity skew, feature skew
│   ├── models.py        # TwoNN (199,210 params) and CNN (1,663,370 params)
│   ├── fl.py            # client_update, fedavg_round, run_fl
│   ├── utils.py         # seeding, tensor loading, evaluate, param count
│   └── plots.py         # plotting functions reading from results/
├── results/             # experiment logs (.csv)
├── plots/               # generated figures (.png)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

Launch Jupyter or run notebooks directly:

```bash
jupyter notebook notebooks/
```

Notebook flow:
- `data_and_partitions.ipynb`: downloads MNIST and builds partition files in `data/`.
- `centralized_baseline.ipynb`: trains TwoNN and CNN baselines.
- `fedsgd_fedavg.ipynb`: evaluates client fraction $C$, local epochs $E$, and batch size $B$.
- `noniid.ipynb`: compares IID, pathological shards, and Dirichlet splits.
- `common_init.ipynb`: verifies model weight interpolation (Paper 1 Figure 1).
- `survey_demos.ipynb`: cross-silo vs cross-device, client dropouts, communication volume, and client drift.
- `plots.ipynb`: loads CSVs from `results/` and generates all figures into `plots/`.

## Key Findings

### Local Computation Accelerates Convergence
Increasing local epochs $E$ and using mini-batches $B$ reduces the number of communication rounds needed:

![FedAvg Computation](plots/fedavg_computation.png)

### Non-IID Data and Client Drift
Pathological shards and high label skew slow convergence and cause update variance across clients:

![Non-IID Comparison](plots/noniid.png)

### Common vs Independent Initialisation
Models averaged from a shared initialisation point interpolate smoothly with low loss, while models trained from independent starts encounter a severe barrier:

![Common Init](plots/common_init.png)
