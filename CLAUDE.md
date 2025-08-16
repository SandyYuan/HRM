# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Read it on every run. If unsure, **plan first**, then wait for approval.

## What to optimize for
- **Correctness & reproducibility > speed.** Prefer small, reviewable diffs.
- Preserve **numerical behavior** and **API stability** unless explicitly asked.
- For risky edits (sensitivity-critical math, instrument constants), **propose a plan** and wait.

## Safety & permissions
- **Ask before**: changing public APIs, upgrading deps, schema/format changes, editing files under `src/pipeline/*` or `src/spec/*`.
- Do **not** run destructive shell commands without explicit approval.
- Do **not** attribute commits to AI tools.

## Coding standards
- **Tests:**  
  - New features require tests; bug fixes require regression tests.  
  - Use realistic shapes (large-ish arrays) and assert **numeric tolerances** explicitly.  
  - Seed randomness and avoid time/network dependencies.


## Repository Overview

This is the Hierarchical Reasoning Model (HRM) implementation - a novel neural architecture for complex reasoning tasks. HRM uses a two-level hierarchical system (H-level for abstract planning, L-level for detailed computation) with adaptive computation time (ACT) for dynamic halting.

## NERSC Perlmutter Setup

**Status**: ✅ Fully installed and working using Shifter containers

### Installation Summary:
- **Container**: `nvcr.io/nvidia/pytorch:24.07-py3` with GLIBC 2.35
- **PyTorch**: 2.4.0 with CUDA support  
- **FlashAttention**: 2.4.2 (working)
- **Dependencies**: adam-atan2, hydra-core, omegaconf, argdantic, coolname, wandb
- **HRM models**: Import and config loading successful

### Container Usage:
All HRM commands must be run inside the Shifter container due to GLIBC compatibility requirements for FlashAttention:

```bash
shifter --image=docker:nvcr.io/nvidia/pytorch:24.07-py3 --volume=/global/u1/s/sihany/hrm/HRM:/workspace bash -c "cd /workspace && [your_command]"
```

**Example training command:**
```bash
shifter --image=docker:nvcr.io/nvidia/pytorch:24.07-py3 --volume=/global/u1/s/sihany/hrm/HRM:/workspace bash -c "cd /workspace && OMP_NUM_THREADS=8 python pretrain.py"
```

**Note**: The native NERSC environment has GLIBC 2.31 which is incompatible with FlashAttention (requires 2.32+). The container provides the necessary newer GLIBC version.

## Common Development Commands

### Dataset Preparation
```bash
# Initialize submodules first
git submodule update --init --recursive

# ARC datasets
python dataset/build_arc_dataset.py  # ARC-1 (960 examples)
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 (1120 examples)

# Sudoku datasets
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000  # 1K sample
python dataset/build_sudoku_dataset.py  # Full dataset

# Maze datasets
python dataset/build_maze_dataset.py  # 1K examples
```

### Training Commands

**Single GPU (laptop/smaller setup):**
```bash
# Quick Sudoku demo (~10 hours on RTX 4070)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

**Multi-GPU (8-GPU setup):**
```bash
# ARC-1 training (~24 hours)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py

# ARC-2 training (~24 hours, often sufficient after 8 hours)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000

# Sudoku Extreme (~10 minutes)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Maze solving (~1 hour)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

### Evaluation Commands
```bash
# Evaluate trained model
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>

# For ARC evaluation, use arc_eval.ipynb notebook after running evaluate.py
```

### Dependencies Installation
```bash
# Install requirements
pip install -r requirements.txt

# CUDA setup (if needed)
# See README.md for detailed CUDA 12.6 and FlashAttention installation

# W&B login for experiment tracking
wandb login
```

## Code Architecture

### Core Model Structure (`models/hrm/hrm_act_v1.py`)
- **HierarchicalReasoningModel_ACTV1**: Main model with ACT wrapper
- **HierarchicalReasoningModel_ACTV1_Inner**: Core hierarchical reasoning engine
- **Two-level hierarchy**: H-level (2 cycles, abstract) + L-level (2 cycles, detailed)
- **Non-autoregressive**: Single forward pass, no sequential generation
- **Adaptive halting**: Q-learning based dynamic computation time

### Configuration System
- **Hydra-based**: YAML configs in `config/` directory
- **Main config**: `cfg_pretrain.yaml` contains hyperparameters
- **Architecture**: `config/arch/hrm_v1.yaml` defines model structure
- **Override via CLI**: `python pretrain.py lr=1e-5 epochs=1000`

### Dataset Handling (`puzzle_dataset.py`)
- **Memory-mapped data**: Efficient large dataset handling  
- **Distributed support**: Multi-GPU data loading
- **Batch packing**: Dynamic batching with sequence padding
- **Data augmentation**: Dihedral transformations (rotations, reflections)

### Loss Functions (`models/losses.py`)
- **ACTLossHead**: Multi-objective loss combining:
  - Language modeling loss (next token prediction)
  - Halt prediction loss (Q-learning for adaptive computation)
  - Stablemax cross-entropy for numerical stability

### Key Implementation Notes
- **Mixed precision**: bfloat16 forward + float32 gradients
- **Sparse embeddings**: Memory-efficient puzzle-specific representations
- **Custom optimizers**: AdamATan2 for standard params, SignSGD for sparse embeddings
- **FlashAttention**: Required for efficient attention computation
- **Torch compilation**: Automatic when available

### Evaluation & Visualization
- **Metrics**: Check `eval/exact_accuracy` in W&B dashboard
- **ARC evaluation**: Use `arc_eval.ipynb` notebook for detailed analysis
- **Puzzle browser**: Open `puzzle_visualizer.html` to explore datasets
- **Early stopping**: Recommended for Sudoku-Extreme to avoid overfitting

### Important Training Characteristics
- **No pre-training**: Trains from scratch on puzzle data
- **Sample efficient**: Strong performance with only 1000 examples
- **Small model**: ~27M parameters
- **Variance**: Small-sample learning has ±2 point accuracy variance
- **Numerical stability**: Use early stopping when training accuracy approaches 100%