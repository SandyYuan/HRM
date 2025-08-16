# HRM Training Guide for NERSC Perlmutter

This guide provides step-by-step instructions for training the Hierarchical Reasoning Model (HRM) on NERSC Perlmutter.

## Prerequisites

- NERSC account with access to the `desi` project
- HRM repository cloned to `/global/homes/s/sihany/hrm/HRM`
- Virtual environment set up (see Setup section)

## Setup (One-time)

### 1. Environment Setup
The HRM has been modified to work with standard PyTorch attention instead of FlashAttention:
- ✅ Modified `models/layers.py` to fall back to `F.scaled_dot_product_attention`
- ✅ Virtual environment created with all dependencies
- ✅ Datasets already built and available

### 2. Available Datasets
```
data/
├── sudoku-extreme-small/     # Small Sudoku dataset for testing
└── sudoku-extreme-1k-aug-1000/  # Larger Sudoku dataset (1K examples)
```

## Training Instructions

### Step 1: Get GPU Compute Node
**Important**: Always run training on a dedicated GPU compute node, not on login nodes.

```bash
# Allocate a GPU node for 2 hours
salloc -C gpu -G 1 -N 1 -A desi -q interactive -t 02:00:00

# You should see something like:
# salloc: Nodes nid001044 are ready for job
```

### Step 2: Activate Environment
```bash
cd /global/homes/s/sihany/hrm/HRM
source .venv/bin/activate
```

### Step 3: Run Training

#### Basic Training Command
```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python pretrain.py \
  data_path=data/sudoku-extreme-small \
  epochs=500 \
  eval_interval=100 \
  global_batch_size=4 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.hidden_size=32 \
  arch.H_layers=1 \
  arch.L_layers=1 \
  arch.num_heads=1 \
  arch.expansion=2
```

#### Memory-Optimized Settings (Recommended)
For limited GPU memory, use these smaller model configurations:

```bash
# Ultra-small model (fits in ~1GB GPU memory)
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python pretrain.py \
  data_path=data/sudoku-extreme-small \
  epochs=1000 \
  eval_interval=200 \
  global_batch_size=2 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.hidden_size=16 \
  arch.H_layers=1 \
  arch.L_layers=1 \
  arch.num_heads=1 \
  arch.expansion=2
```

#### Full-size Model (Requires dedicated GPU with >10GB free)
```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=20000 \
  eval_interval=2000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0
```

## Parameter Explanations

### Environment Variables
- `DISABLE_COMPILE=1` - Disables torch compilation to avoid Triton/CUDA issues
- `OMP_NUM_THREADS=8` - Sets OpenMP threads for CPU operations

### Training Parameters
- `data_path` - Path to the dataset directory
- `epochs` - Number of training epochs
- `eval_interval` - How often to run evaluation (in epochs)
- `global_batch_size` - Total batch size across all GPUs
- `lr` - Learning rate for main model parameters
- `puzzle_emb_lr` - Learning rate for puzzle embeddings
- `weight_decay` - L2 regularization strength
- `puzzle_emb_weight_decay` - L2 regularization for puzzle embeddings

### Model Architecture Parameters
- `arch.hidden_size` - Hidden dimension size (32, 64, 128, 256, 512)
- `arch.H_layers` - Number of high-level reasoning layers
- `arch.L_layers` - Number of low-level reasoning layers  
- `arch.num_heads` - Number of attention heads
- `arch.expansion` - MLP expansion ratio

## Memory Guidelines

| GPU Memory Available | Recommended Settings |
|---------------------|---------------------|
| < 2GB | `hidden_size=16, batch_size=2, H_layers=1, L_layers=1` |
| 2-5GB | `hidden_size=32, batch_size=4, H_layers=1, L_layers=1` |
| 5-10GB | `hidden_size=64, batch_size=8, H_layers=2, L_layers=2` |
| > 10GB | Default settings from config files |

## Monitoring Training

### Weights & Biases Dashboard
Training metrics are automatically logged to W&B:
- **Project**: `Sudoku-extreme-small ACT-torch`
- **URL**: https://wandb.ai/sihany/Sudoku-extreme-small%20ACT-torch

Key metrics to watch:
- `train/lm_loss` - Language modeling loss
- `eval/exact_accuracy` - Exact sequence accuracy
- `train/lr` - Learning rate schedule

### Local Monitoring
```bash
# Check training progress
tail -f wandb/latest-run/logs/debug.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check for saved checkpoints
ls -la checkpoints/
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce model size or batch size:
```bash
# Use smaller parameters
arch.hidden_size=16 global_batch_size=2
```

#### 2. Training Gets Stuck
If training stops progressing:
- Kill the process: `pkill -f pretrain.py`
- Restart with smaller batch size
- Check GPU memory: `nvidia-smi`

#### 3. Module Not Found Errors
```
ModuleNotFoundError: No module named 'coolname'
```
**Solution**: Activate the virtual environment:
```bash
source .venv/bin/activate
```

#### 4. FlashAttention Errors
The model has been modified to work without FlashAttention, but if you see FA errors:
- Ensure `FLASH_ATTN_AVAILABLE = False` in the code
- The fallback to standard PyTorch attention should work automatically

## Background Training

To run training in the background (survives terminal disconnection):

```bash
# Start in background
nohup bash -c "
source .venv/bin/activate
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python pretrain.py \
  data_path=data/sudoku-extreme-small \
  epochs=1000 \
  eval_interval=200 \
  global_batch_size=4 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.hidden_size=32 \
  arch.H_layers=1 \
  arch.L_layers=1 \
  arch.num_heads=1 \
  arch.expansion=2
" > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## Expected Results

### Training Speed
- **Small model** (hidden_size=32): ~50-60 iterations/second
- **Medium model** (hidden_size=64): ~30-40 iterations/second
- **Large model** (hidden_size=512): ~5-10 iterations/second

### Convergence
- **Sudoku-extreme-small**: Should reach >90% accuracy within 500-1000 epochs
- **Training loss**: Should decrease from ~8.0 to <2.0
- **Exact accuracy**: Should increase from 0% to >80%

## File Outputs

### Checkpoints
```
checkpoints/
└── Sudoku-extreme-small ACT-torch/
    └── HierarchicalReasoningModel_ACTV1 <run-name>/
        ├── step_<N>              # Model weights
        ├── all_config.yaml       # Training configuration
        └── hrm_act_v1.py        # Model source code
```

### Logs
```
wandb/
└── run-<timestamp>-<id>/
    ├── logs/
    └── files/
```

## Next Steps

1. **Monitor training** via W&B dashboard
2. **Evaluate model** using `evaluate.py` once training completes
3. **Scale up** to larger datasets once small model works
4. **Experiment** with different hyperparameters

---

## Quick Start Summary

```bash
# 1. Get GPU node
salloc -C gpu -G 1 -N 1 -A desi -q interactive -t 02:00:00

# 2. Run training
cd /global/homes/s/sihany/hrm/HRM
source .venv/bin/activate
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-small epochs=500 eval_interval=100 global_batch_size=4 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.hidden_size=32 arch.H_layers=1 arch.L_layers=1 arch.num_heads=1 arch.expansion=2

# 3. Monitor at: https://wandb.ai/sihany/Sudoku-extreme-small%20ACT-torch
```

**Success!** 🎉 The HRM model is now running on NERSC Perlmutter with standard PyTorch attention.
