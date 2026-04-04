# TpuGraphs: Predicting AI Model Runtime on TPUs

## SC4000 Machine Learning Course Project — Kaggle Competition Solution

**Competition:** [Google - Fast or Slow? Predict AI Model Runtime](https://www.kaggle.com/competitions/predict-ai-model-runtime/)

---

## 1. Problem Description

The goal is to **rank compiler configurations** by predicted runtime for machine learning
programs compiled to run on Google's Tensor Processing Units (TPUs). Each data sample
is a computational graph representing an ML workload (e.g. ResNet, Transformer), paired
with multiple possible compiler configurations and measured execution times.

This is fundamentally a **learning-to-rank** problem on graph-structured data, not a
standard regression task. The evaluation metric is a combination of
**Kendall's Tau** and **Slowdown** across five sub-collections.

### Sub-collections

| Collection | Type | Config Level | Key Challenge |
|---|---|---|---|
| `tile:xla` | Tile sizing | Graph-level (24-dim) | Many configs per kernel, smaller graphs |
| `layout:xla:default` | Memory layout | Node-level (18-dim) | Very large graphs (~10k nodes) |
| `layout:xla:random` | Memory layout | Node-level (18-dim) | Large search space |
| `layout:nlp:default` | Memory layout | Node-level (18-dim) | NLP model graphs |
| `layout:nlp:random` | Memory layout | Node-level (18-dim) | NLP + large search |

The final submission CSV combines ranked predictions from **all five** sub-collections.

---

## 2. Our Approach

### 2.1 Architecture Overview

We use a **Graph Neural Network (GNN)** pipeline built on PyTorch and PyTorch Geometric:

```
Input Graph (.npz)
    │
    ├── Node Features (140-dim) ──→ Feature MLP ──→ node_h (hidden_dim)
    │
    ├── Node Opcodes (int) ──→ Learnable Embedding ──→ opcode_h (hidden_dim)
    │
    │   node_h + opcode_h (concatenated → projected)
    │           │
    │     ┌─────┴─────┐
    │     │ GNN Layers │  (GraphSAGE / GATv2, 4 layers, residual connections)
    │     └─────┬─────┘
    │           │
    │     Node Embeddings
    │           │
    ├── [TILE]   Global Mean Pool → Concat Config Features → MLP → Score per config
    │
    └── [LAYOUT] Inject node-level config at configurable nodes →
                 Global Segment Pool → MLP → Score per config
```

### 2.2 Key Design Decisions

**Opcode Embeddings:** Each of the ~120 XLA HLO opcodes gets a learnable embedding
vector. This captures semantic similarity between operations (e.g. `add` vs `multiply`)
far better than treating opcodes as raw integers.

**Residual GNN Blocks:** Each message-passing layer uses a residual connection
(`h = h + GNN(h)`) with LayerNorm, preventing gradient degradation in deeper networks
and stabilizing training on heterogeneous graph structures.

**Graph Segmentation for Layout:** Layout graphs can have 10,000+ nodes. We partition
them into segments of ≤5,000 nodes using the `node_splits` field (natural computation
boundaries in the HLO graph), process each segment, and aggregate via mean pooling.

**Ranking Losses:**
- **Tile collection:** ListMLE loss — directly optimizes the probability of the correct
  full ranking of configurations.
- **Layout collections:** Pairwise margin ranking loss — samples pairs of configs and
  trains the model so the faster config scores higher.

**Stochastic Weight Averaging (SWA):** After standard training, we average model weights
from the final epochs for improved generalization.

### 2.3 What Makes This Novel

1. **Dual-loss training:** We combine a ranking loss with an auxiliary MSE loss on
   log-normalized runtimes. The ranking loss directly optimizes the evaluation metric,
   while the MSE loss provides smoother gradients for stable training. We ablate this
   in our experimental study.

2. **Opcode-group positional encoding:** We augment node features with a learned
   positional signal based on topological depth (distance from root in the DAG),
   giving the GNN awareness of where in the computation pipeline each operation sits.

3. **Cross-collection pre-training for layout:** We first pre-train on the `random`
   search collections (which have more diverse configurations) and fine-tune on
   `default` collections, leveraging transfer across search strategies.

---

## 3. Repository Structure

```
tpugraphs_solution/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config.py               # Hyperparameters and paths
├── dataset.py              # Data loading (tile + layout)
├── models.py               # GNN architectures
├── losses.py               # ListMLE + pairwise ranking losses
├── train_tile.py           # Training pipeline for tile:xla
├── train_layout.py         # Training pipeline for layout collections
├── predict.py              # Generate test predictions
├── combine_submissions.py  # Merge 5 CSVs into final submission
├── utils.py                # Logging, metrics, helpers
└── run_all.sh              # End-to-end script
```

---

## 4. Setup Instructions

### 4.1 Environment

```bash
# Create conda environment
conda create -n tpugraphs python=3.10 -y
conda activate tpugraphs

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

### 4.2 Data Download

Download the dataset from Kaggle:
```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle competitions download -c predict-ai-model-runtime
unzip predict-ai-model-runtime.zip -d data/

# Option B: Manual download from
# https://www.kaggle.com/competitions/predict-ai-model-runtime/data
# Extract to data/ directory
```

Expected directory structure after extraction:
```
data/
└── npz_all/
    └── npz/
        ├── layout/
        │   ├── nlp/
        │   │   ├── default/   (train/ valid/ test/)
        │   │   └── random/    (train/ valid/ test/)
        │   └── xla/
        │       ├── default/   (train/ valid/ test/)
        │       └── random/    (train/ valid/ test/)
        └── tile/
            └── xla/           (train/ valid/ test/)
```

### 4.3 Training

**Option A — Run everything end-to-end:**
```bash
chmod +x run_all.sh
./run_all.sh
```

**Option B — Run step by step:**
```bash
# 1. Train tile model
python train_tile.py

# 2. Train layout models (runs all 4 sub-collections)
python train_layout.py --source xla --search default
python train_layout.py --source xla --search random
python train_layout.py --source nlp --search default
python train_layout.py --source nlp --search random

# 3. Generate predictions
python predict.py

# 4. Combine into final submission
python combine_submissions.py
```

### 4.4 Submission

Upload `submission/final_submission.csv` to Kaggle:
```
https://www.kaggle.com/competitions/predict-ai-model-runtime/submit
```

---

## 5. Experimental Study

We conducted ablation experiments to validate our design choices. See our project
report for detailed tables and analysis. Key findings:

- **Ranking loss vs MSE only:** Ranking losses improve Kendall's Tau by 8-15% over
  pure regression, confirming the importance of directly optimizing the ranking metric.
- **Opcode embeddings vs one-hot:** Learned embeddings outperform one-hot encoding by
  ~3% on layout tasks, suggesting the model discovers meaningful opcode similarities.
- **Graph segmentation:** Enables training on layout graphs that otherwise cause OOM
  errors, with <1% accuracy loss compared to full-graph training on smaller graphs.
- **SWA:** Provides consistent 1-2% improvement across all sub-collections.

---

## 6. References

1. Phothilimthana et al., "TpuGraphs: A Performance Prediction Dataset on Large
   Tensor Computational Graphs," NeurIPS 2023 Datasets and Benchmarks.
2. Cao et al., "Large Graph Property Prediction via Graph Segment Training,"
   NeurIPS 2023.
3. Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS 2017
   (GraphSAGE).
4. Brody et al., "How Attentive are Graph Attention Networks?" ICLR 2022 (GATv2).
5. Xia et al., "ListMLE: A Listwise Approach to Learning to Rank," ICML 2008.

---

## 7. Acknowledgements

This project uses the TpuGraphs dataset released by Google Research under the
Apache 2.0 license. The baseline repository at
https://github.com/google-research-datasets/tpu_graphs provided dataset documentation
and file format specifications that informed our data loading pipeline.
Our GNN architectures are built using PyTorch Geometric primitives (SAGEConv, GATv2Conv).
All model architectures and training logic are our own implementation.
