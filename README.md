# DistillHGNN: A Knowledge Distillation Approach for High-Speed Hypergraph Neural Networks

[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue.svg)](https://iclr.cc/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)

> **Published at The Thirteenth International Conference on Learning Representations (ICLR) 2025**

## 🎯 Overview

**DistillHGNN** is a breakthrough framework that revolutionizes Hypergraph Neural Networks (HGNNs) by delivering **up to 82× faster inference** while maintaining competitive accuracy. Our approach addresses the critical bottleneck of HGNNs in real-world applications through an innovative teacher-student knowledge distillation strategy that preserves both soft labels and structural knowledge.

### Why DistillHGNN?
- 🚀 **Extreme Speed**: 69-82% inference time reduction compared to traditional HGNNs
- 💾 **Memory Efficient**: Significantly reduced memory footprint for large-scale deployment
- 🎯 **High Accuracy**: Maintains performance comparable to full HGNNs across diverse datasets
- 🔄 **Dual Knowledge Transfer**: Novel combination of soft labels and structural knowledge preservation
- 🏭 **Production Ready**: Optimized for real-time, large-scale industrial applications

## 🏗️ Architecture

<div align="center">

### Teacher-Student Framework

| **Teacher Model** | **Student Model** |
|-------------------|-------------------|
| • **HGNN**: Captures high-order relationships | • **TinyGCN**: Lightweight single-layer GCN |
| • **MLP**: Generates soft labels | • **MLP**: Optimized for fast inference |
| • **Complex Structure**: Full hypergraph processing | • **Streamlined**: No activation functions |

</div>

### 🔬 Technical Innovation

#### 1. **Comprehensive Knowledge Distillation**
```
Teacher HGNN → Soft Labels + Structural Embeddings → Student TinyGCN
```
- **Soft Label Transfer**: Class probability distributions
- **Structural Knowledge**: High-order relationship preservation
- **Contrastive Learning**: Embedding alignment between teacher and student

#### 2. **TinyGCN Architecture**
- **Single Layer**: Minimal computational complexity
- **No Activation Functions**: Streamlined for speed
- **Linear Aggregation**: Direct neighbor information processing
- **Formula**: `Z_s = Â_s XW_s` where `Â_s = A_s + I`

#### 3. **Dual Loss Optimization**
- **Teacher Loss**: `L_teacher = L_bpr + γL_con`
- **Student Loss**: `L_student = MSE_loss + λKL_divergence`
- **Contrastive Loss**: InfoNCE for embedding alignment

## 📊 Performance Highlights

### Inference Speed Comparison
| Model | IMDB-AW (ms) | DBLP (ms) | Speedup |
|-------|--------------|-----------|---------|
| HGNN | 175.56 | 168.84 | 1× |
| **DistillHGNN** | **2.23** | **2.06** | **79-82×** |
| LightHGNN | 4.8 | 4.2 | 35-40× |

### Accuracy Performance
| Dataset | HGNN | LightHGNN | **DistillHGNN** |
|---------|------|-----------|-----------------|
| CC-Cora | 65.52% | 64.11% | **65.68%** |
| CC-Citeseer | 61.39% | 60.11% | **61.88%** |
| IMDB-AW | 53.31% | 51.84% | **53.93%** |
| DBLP | 83.55% | 81.88% | **83.77%** |

### High-Order Preservation Score
| Method | CC-Cora | CC-Citeseer | IMDB-AW | DBLP | **Mean** |
|--------|----------|-------------|---------|------|----------|
| GLNN | 0.54 | 0.58 | 0.51 | 0.67 | 0.575 |
| KRD | 0.56 | 0.62 | 0.54 | 0.71 | 0.608 |
| LightHGNN | 0.72 | 0.78 | 0.74 | 0.83 | 0.768 |
| **DistillHGNN** | **0.78** | **0.84** | **0.81** | **0.88** | **0.828** |

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/DistillHGNN.git
cd DistillHGNN

# Create conda environment
conda create -n distillhgnn python=3.8
conda activate distillhgnn

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.9.0
torch-geometric>=2.0.0
dhg>=0.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

### Basic Usage
```python
from distillhgnn import DistillHGNN
from dhg import Hypergraph

# Initialize model
model = DistillHGNN(
    num_features=1433,
    num_classes=7,
    hidden_dim=128,
    tau=0.5,
    gamma=0.4,
    lambda_=0.2
)

# Train the model
model.fit(hypergraph, features, labels)

# Fast inference
predictions = model.predict(test_features)
```

### Training Script
```bash
# Train on CC-Cora dataset
python train.py --dataset CC-Cora --epochs 1000 --lr 0.001 --hidden_dim 128

# Train on DBLP dataset with custom parameters
python train.py --dataset DBLP --epochs 800 --tau 0.5 --gamma 0.4 --lambda 0.2
```

## 📁 Project Structure
```
DistillHGNN/
├── distillhgnn/
│   ├── models/
│   │   ├── hgnn.py          # Teacher HGNN implementation
│   │   ├── tinygcn.py       # Student TinyGCN implementation
│   │   └── distillhgnn.py   # Main DistillHGNN framework
│   ├── utils/
│   │   ├── data_loader.py   # Dataset handling
│   │   ├── losses.py        # Loss functions
│   │   └── metrics.py       # Evaluation metrics
│   └── __init__.py
├── experiments/
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── ablation_study.py    # Ablation experiments
├── data/                    # Dataset directory
├── results/                 # Experimental results
├── requirements.txt
└── README.md
```

## 🔬 Experimental Results

### Datasets
We evaluate DistillHGNN on 8 benchmark datasets:
- **Citation Networks**: CC-Cora, CC-Citeseer
- **Movie Database**: IMDB, IMDB-AW
- **Bibliographic**: DBLP, DBLP-Paper, DBLP-Term, DBLP-Conf

### Key Findings
1. **Speed vs Accuracy Trade-off**: DistillHGNN achieves the optimal balance
2. **Scalability**: Performance improvements increase with dataset size
3. **Memory Efficiency**: Constant memory usage regardless of hypergraph complexity
4. **Robustness**: Consistent performance across diverse domains

## 🔍 Ablation Studies

### Knowledge Transfer Components
- **Soft Labels Only**: 3-7% accuracy drop
- **Structural Knowledge Only**: 5-10% accuracy drop
- **Combined Approach (Ours)**: Optimal performance

### Contrastive Learning Impact
- **Without CL**: Reduced high-order relationship preservation
- **With CL**: Enhanced embedding alignment and accuracy

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{forouzandeh2025distillhgnn,
  title={DistillHGNN: A Knowledge Distillation Approach for High-Speed Hypergraph Neural Networks},
  author={Forouzandeh, Saman and Moradi, Parham and Jalili, Mahdi},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=...}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black distillhgnn/
isort distillhgnn/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research is partially supported by Australian Research Council through projects DP240100963, LP230100439, and IM240100042.

## 📧 Contact

- **Saman Forouzandeh** - saman.forouzandeh@rmit.edu.au
  
**RMIT University, School of Engineering, Melbourne, Australia**

---
