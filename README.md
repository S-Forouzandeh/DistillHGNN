# DistillHGNN
DistillHGNN: A Knowledge Distillation Approach for High-Speed Hypergraph Neural Networks

## The Thirteenth International Conference on Learning Representations (ICLR) - 2025

## Overview

DistillHGNN is a novel framework that significantly enhances the inference speed and memory efficiency of Hypergraph Neural Networks (HGNNs) while maintaining high accuracy. Our approach employs an advanced teacher-student knowledge distillation strategy that makes HGNNs practical for real-world, large-scale applications.

## Key Features

- Fast Inference: Significantly reduced inference time compared to traditional HGNNs
- Memory Efficient: Lower memory requirements through efficient knowledge distillation
- High Accuracy: Maintains comparable accuracy to full HGNNs
- Dual Transfer: Unique combination of soft label and structural knowledge transfer
- Real-time Ready: Optimized for real-world, large-scale applications

## Architecture

### Teacher Model
- HGNN for complex relationship capture
- MLP for soft label generation
- Embedding generation for knowledge transfer

### Student Model
- TinyGCN (lightweight single-layer GCN)
- Optimized MLP
- No non-linear activation functions for efficiency

## Technical Highlights

1. Comprehensive Knowledge Transfer
   - Soft label transfer
   - Structural knowledge preservation
   - High-order relationship maintenance

2. Contrastive Learning Integration
   - Maximizes embedding similarity
   - Preserves structural information
   - Efficient knowledge transfer

3. Streamlined Architecture
   - Single-layer TinyGCN
   - Removed activation functions
   - Optimized for speed
