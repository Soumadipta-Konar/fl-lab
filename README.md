# FL-Lab: Reproducing Foundational Federated Learning

A clean, reproducible implementation and empirical study of the foundational algorithms and theoretical concepts in Federated Learning (FL), based on:

1. **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (McMahan et al., AISTATS 2017)
2. **FL Survey & Taxonomy**: [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) (Kairouz et al., 2021)

This repository is built for researchers and students who want to understand how federated optimization actually behaves under the hood—from mathematical formulation to convergence dynamics, data heterogeneity, loss landscape geometry, and client drift.

---

## Table of Contents

- [Core Theory & Mathematics](#core-theory--mathematics)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Model Architectures](#model-architectures)
- [Empirical Results & Plot Explanations](#empirical-results--plot-explanations)
  - [1. Data Partitioning & Heterogeneity Modeling](#1-data-partitioning--heterogeneity-modeling)
  - [2. Centralized Baseline (The Accuracy Ceiling)](#2-centralized-baseline-the-accuracy-ceiling)
  - [3. FedSGD vs. FedAvg (Communication Efficiency)](#3-fedsgd-vs-fedavg-communication-efficiency)
  - [4. The Non-IID Penalty & Convergence Degradation](#4-the-non-iid-penalty--convergence-degradation)
  - [5. Weight Space Geometry: Why Common Initialisation is Mandatory](#5-weight-space-geometry-why-common-initialisation-is-mandatory)
  - [6. Practical System Dynamics (Paper 2 Survey Demos)](#6-practical-system-dynamics-paper-2-survey-demos)
- [Summary: Rules of Thumb for FL Practitioners](#summary-rules-of-thumb-for-fl-practitioners)

---

## Core Theory & Mathematics

Federated Learning solves an optimization problem over $K$ decentralized clients without communicating their private datasets $\mathcal{D}_1, \dots, \mathcal{D}_K$:

$$\min_{w} f(w) = \sum_{k=1}^K \frac{n_k}{n} F_k(w) \quad \text{where} \quad F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)$$

Here $n_k = |\mathcal{D}_k|$ is the local sample size, and $n = \sum_k n_k$ is the total sample count.

### 1. FedSGD
In each communication round, the server samples a subset $S_t$ of $m = \max(C \cdot K, 1)$ clients ($C \in (0, 1]$). Each client computes the exact average gradient on its entire local dataset:

$$g_k^t = \nabla F_k(w_t)$$

The server averages these gradients and updates the global model:

$$w_{t+1} = w_t - \eta \sum_{k \in S_t} \frac{n_k}{n_t} g_k^t$$

FedSGD requires 1 communication round for every single gradient step. In bandwidth-constrained distributed systems, this is impractically slow.

### 2. Federated Averaging (FedAvg)
Instead of communicating gradients, clients perform $E$ local epochs of SGD using mini-batch size $B$ starting from the global model $w_t$. Each client computes:

$$u = \frac{E \cdot n_k}{B} \quad \text{local gradient steps}$$

obtaining updated local parameters $w_k^{t+1}$. The server aggregates the weights:

$$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{n_t} w_k^{t+1} = w_t + \sum_{k \in S_t} \frac{n_k}{n_t} (w_k^{t+1} - w_t)$$

By adding local computation ($E > 1, B < \infty$), FedAvg dramatically reduces the number of communication rounds required to train a global model.

---

## Repository Structure

```
fl-lab/
├── notebooks/
│   ├── data_and_partitions.ipynb    # Data downloading, partition generation, distributions
│   ├── centralized_baseline.ipynb   # Centralized training ceilings (TwoNN & CNN)
│   ├── fedsgd_fedavg.ipynb          # Paper 1 sweeps: C participation and E, B computation
│   ├── noniid.ipynb                 # IID vs. Dirichlet vs. Pathological shard benchmarks
│   ├── common_init.ipynb            # Loss interpolation (Figure 1 reproduction)
│   ├── survey_demos.ipynb           # Paper 2: cross-silo/device, dropout, drift, bandwidth
│   └── plots.ipynb                  # Generates all publication-ready figures
├── src/
│   ├── partition.py                 # Partition algorithms (IID, shards, Dirichlet, skew)
│   ├── models.py                    # Exact PyTorch TwoNN and CNN definitions
│   ├── fl.py                        # Federated optimization: client_update, fedavg_round
│   ├── utils.py                     # Seeding, tensor batching, evaluation, parameter counting
│   └── plots.py                     # Matplotlib plotting routines reading from results/
├── results/                         # Raw experiment metrics logged as CSV files
├── plots/                           # Generated 150 DPI figures
├── requirements.txt                 # Minimal dependency manifest
└── README.md
```

---

## Installation & Quickstart

```bash
git clone https://github.com/your-username/fl-lab.git
cd fl-lab
pip install -r requirements.txt
```

Launch the interactive Jupyter interface:

```bash
jupyter notebook notebooks/
```

Or execute any experiment directly in headless mode:

```bash
jupyter execute notebooks/fedsgd_fedavg.ipynb
```

---

## Model Architectures

Implemented in [`src/models.py`](file:///c:/Users/Soumadipta%20Konar/Desktop/FL-Lab/src/models.py), verified against the exact parameter counts from McMahan et al. 2017:

| Model | Architecture | Parameters | Role in FL-Lab |
| :--- | :--- | :--- | :--- |
| **TwoNN** | Fully-connected: $784 \to 200 \to 200 \to 10$ with ReLU activations | **199,210** | Fast CPU benchmarking of loss landscapes and client drift |
| **CNN** | Two $5\times 5$ conv layers (32, 64 channels, same-padding) with $2\times 2$ max-pooling, fully connected layer with 512 units, ReLU, and 10 outputs | **1,663,370** | Realistic vision benchmark establishing the high-capacity accuracy ceiling |

---

## Empirical Results & Plot Explanations

### 1. Data Partitioning & Heterogeneity Modeling

Federated datasets deviate from the standard Independent and Identically Distributed (IID) assumption. We model three distinct distributions across $K = 100$ clients ($600$ samples/client):

#### IID Partition
Every client receives an identical, uniform draw across all 10 digit classes ($\sim 60$ samples per digit).

![IID Distribution](plots/iid_dist.png)

*Explanation:* The heatmap rows represent the 100 clients; columns represent the 10 digit classes. The uniform light-blue coloration shows that every client has equal representation of all classes. Gradient vectors calculated locally point in approximately the same direction as the global gradient.

#### Pathological Non-IID (200 Shards)
The 60,000 training images are sorted by label and partitioned into 200 shards of 300 images each. Each client receives exactly 2 random shards.

![Pathological Distribution](plots/pathological_dist.png)

*Explanation:* Each horizontal row has exactly two dark-blue entries ($\sim 300$ samples) and 8 white entries (0 samples). Most clients see only 2 digits. When a client performs local training, its parameters overfit to only these 2 classes. This represents the extreme worst-case benchmark for testing federated stability.

#### Dirichlet Non-IID Label Skew ($\alpha = 0.5$)
For each class $c$, proportions $p_c \sim \text{Dir}(\alpha \cdot \mathbf{1}_K)$ determine the sample allocation across clients. Smaller $\alpha$ produces higher variance and sharper skew.

![Dirichlet Distribution](plots/dirichlet_dist.png)

*Explanation:* Unlike the artificial 2-shard split, Dirichlet allocation produces a continuous, realistic heterogeneity spectrum. Clients hold varying proportions of different digits, closely mirroring real-world hospital and edge-device silos.

---

### 2. Centralized Baseline (The Accuracy Ceiling)

Before evaluating federated algorithms, we train both models on the unified dataset (60,000 samples) to establish the theoretical performance ceiling:

![Centralized Baseline](plots/centralized.png)

*Key Findings:*
- **TwoNN Ceiling:** Reaches **97.8%** test accuracy after 10 epochs.
- **CNN Ceiling:** Reaches **98.9%** test accuracy within 5 epochs.
- Any federated algorithm's objective is to approach these centralized ceilings with minimal communication rounds.

---

### 3. FedSGD vs. FedAvg (Communication Efficiency)

#### Client Participation Fraction ($C$) in FedSGD
In FedSGD, each selected client runs only $E=1$ epoch with batch size $B = \text{all data}$ ($600$ samples). We sweep $C \in \{0.0, 0.1, 0.2, 0.5, 1.0\}$.

![FedSGD Participation](plots/fedsgd_participation.png)

*Analysis:*
- $C = 0.0$ (sampling 1 client/round) produces noisy, slow convergence.
- Increasing client participation $C \ge 0.1$ stabilizes the gradient estimate, but after 8 communication rounds, test accuracy remains below **65%**.
- **Takeaway:** Merely adding more clients per round in FedSGD does not solve the fundamental communication bottleneck: the model only takes one gradient step per round.

#### Local Computation in FedAvg ($E$ and $B$)
FedAvg allows clients to take multiple gradient steps locally before communicating. We fix $C = 0.1$ (10 clients/round) and evaluate:
1. $E = 1, B = \text{all}$ ($u = 1$ step/round)
2. $E = 1, B = 50$ ($u = 12$ steps/round)
3. $E = 5, B = 10$ ($u = 300$ steps/round)

![FedAvg Computation](plots/fedavg_computation.png)

*Analysis:*
- $E = 5, B = 10$ surges past **90% accuracy in just 1 round**, reaching **94.8% by round 8**.
- In contrast, $E = 1, B = \text{all}$ reaches only **62% by round 8**.
- **Theoretical Insight:** Adding local computation ($u=300$) accelerates convergence per communication round by nearly an order of magnitude on IID data, confirming the core thesis of McMahan et al. 2017.

---

### 4. The Non-IID Penalty & Convergence Degradation

We test FedAvg ($E = 5, B = 10, C = 0.1$) across identical seeds on IID, Dirichlet ($\alpha = 0.5$), Dirichlet ($\alpha = 0.1$), and Pathological 200-shard partitions:

![Non-IID Comparison](plots/noniid.png)

*Analysis:*
- **IID (Green):** Smooth, monotonic convergence directly reaching $>94\%$.
- **Dirichlet $\alpha = 0.5$ (Orange):** Mild heterogeneity; slightly slower initial ramp but reaches $>93\%$.
- **Dirichlet $\alpha = 0.1$ (Blue):** Severe label skew; plateaus near $\sim 83\%$.
- **Pathological 200-shard (Red):** Severe oscillation and instability. Accuracy fluctuates around $\sim 65\%$.
- **Takeaway:** When clients possess disjoint data distributions, local gradient steps drift toward local minima that conflict with the global objective. Averaging these drifted weights causes an immediate performance penalty.

---

### 5. Weight Space Geometry: Why Common Initialisation is Mandatory

Paper 1 (Figure 1) noted an essential property: **averaging neural network weights only works if models are trained from a shared initialisation point.**

To demonstrate this, we train two TwoNN models ($w_A$ and $w_B$) on disjoint 600-example MNIST subsets under two regimes:
1. **Shared Initialisation:** Both models start from the exact same random weights $w_0$.
2. **Independent Initialisation:** Models start from different seeds $w_0^{(A)} \ne w_0^{(B)}$.

We evaluate the full-dataset loss across the interpolation line:

$$w(\theta) = \theta w_A + (1 - \theta) w_B \quad \text{for } \theta \in [-0.2, 1.2]$$

![Common Initialisation](plots/common_init.png)

*Analysis:*
- **Shared Initialisation (Orange):** The loss curve forms a convex valley. At $\theta = 0.5$ (direct weight average), the loss dips **below** both individual models ($\sim 0.33$). The two models descended into the same basin of attraction.
- **Independent Initialisation (Blue):** At $\theta = 0.5$, the loss spikes above **$1.2$** (a high loss barrier).
- **Theoretical Insight:** Neural loss surfaces contain infinite permuted global minima due to neuron permutation symmetry. Models trained from different initialisations wander into different valleys; linearly averaging their weights produces an invalid model that sits on the saddle/barrier between valleys. Shared initialisation is a strict prerequisite for Federated Averaging.

---

### 6. Practical System Dynamics (Paper 2 Survey Demos)

Kairouz et al. (2021) formalized the distinction between practical federated deployments and operational failure modes.

#### Cross-Silo vs. Cross-Device
- **Cross-Silo:** Few clients ($K = 10$), large local data ($6000$ samples), full participation ($C = 1.0$). Typical of hospital consortia.
- **Cross-Device:** Many clients ($K = 100$), smaller data ($600$ samples), partial participation ($C = 0.1$). Typical of mobile devices.

![Settings Comparison](plots/settings.png)

*Analysis:* Cross-silo converges faster and with zero variance because the aggregator sees updates from 100% of the data in every round. Cross-device experiences sampling variance due to client subsampling.

#### Non-IID Heterogeneity Types
Data skew is not limited to class labels. We evaluate three distinct forms of heterogeneity:
1. **Label Skew:** Dirichlet class distribution ($\alpha = 0.5$).
2. **Quantity Skew:** Identical label proportions, but client dataset sizes follow a Dirichlet distribution ($n_k$ varies from 80 to 2,400 samples).
3. **Feature Skew:** Identical labels and sample counts, but each client's images are rotated by a client-specific angle $\phi_k = \frac{360^\circ \cdot k}{K}$.

![Heterogeneity Types](plots/noniid_types.png)

*Analysis:* Feature rotation skew degrades accuracy more severely than label or quantity skew. When input features are fundamentally transformed across silos, standard FedAvg requires domain-adaptation or personalization layers.

#### Unreliable Clients & Dropout
In mobile FL, clients drop out mid-round due to battery or network loss. We simulate client dropouts with probability $p \in \{0.0, 0.2, 0.5\}$.

![Partial Participation](plots/partial_participation.png)

*Analysis:* Random dropouts act as a stochastic subsampler. While convergence speed is slightly reduced as $p \to 0.5$, FedAvg remains robust to unbiased client dropouts.

#### Communication Cost
Communication volume is the primary bottleneck in edge deployments. Total bytes transferred per round for client subset $S_t$:

$$\text{Bytes} = 2 \times |S_t| \times |\text{parameters}| \times 4 \text{ bytes (float32)}$$

![Communication Cost](plots/communication_cost.png)

*Analysis:* For TwoNN ($199,210$ params), each 10-client round transfers $\approx 15.2\text{ MB}$. Cross-silo achieves high accuracy with fewer total megabytes because every round aggregates all data.

#### Direct Measurement of Client Drift
To quantify client drift directly, we record the client update vectors $\Delta w_k^t = w_k^t - w_t$ and compute the average pairwise cosine similarity:

$$\text{Sim}(t) = \frac{1}{\binom{|S_t|}{2}} \sum_{i < j} \frac{\langle \Delta w_i^t, \Delta w_j^t \rangle}{\|\Delta w_i^t\| \|\Delta w_j^t\|}$$

![Client Drift](plots/client_drift.png)

*Analysis:*
- **IID (Blue):** High cosine similarity ($>0.9$ in early rounds), showing that all clients agree on the descent direction. As the model converges near the optimum, gradients naturally disperse.
- **Pathological Non-IID (Orange):** Cosine similarity starts low ($\sim 0.2$) and drops to nearly **$0.05$**. Client updates point in almost orthogonal directions. This provides explicit, quantitative proof of client drift.

---

## Summary: Rules of Thumb for FL Practitioners

1. **Always use common initialisation:** Never average weights from models initialized with different random seeds.
2. **Local steps accelerate IID training:** On IID data, push $E$ as high as possible and lower $B$ to minimize communication rounds.
3. **Beware of excessive local epochs on non-IID data:** When data is highly non-IID (low $\alpha$ or disjoint classes), large $E$ induces severe client drift. Practitioners must tune $E$ downward or add proximal penalties (e.g., FedProx).
4. **Bandwidth determines algorithm choice:** If network bandwidth is cheap, FedSGD offers steady theoretical convergence; in edge/mobile networks, FedAvg is mandatory.
5. **Feature skew is harder than label skew:** Image rotations and sensor variations require normalization or personalized heads; simple parameter averaging struggles to reconcile rotated features.
