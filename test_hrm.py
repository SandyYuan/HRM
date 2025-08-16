#!/usr/bin/env python3
"""Test script to verify HRM installation and functionality."""

print("Testing HRM installation...")

try:
    # Test basic imports
    import torch
    print(f"✓ PyTorch {torch.__version__} imported")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    # Test HRM imports
    from models.layers import FLASH_ATTN_AVAILABLE
    print(f"✓ FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    print("✓ HRM model imported successfully")
    
    # Test model instantiation
    config = {
        'batch_size': 2,
        'seq_len': 100,
        'puzzle_emb_ndim': 512,
        'num_puzzle_identifiers': 10,
        'vocab_size': 1000,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 512,
        'expansion': 4.0,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'halt_max_steps': 16,
        'halt_exploration_prob': 0.1
    }
    
    model = HierarchicalReasoningModel_ACTV1(config)
    print("✓ HRM model instantiated successfully")
    
    # Test forward pass
    if torch.cuda.is_available():
        model = model.cuda()
        
    batch = {
        'inputs': torch.randint(0, 1000, (2, 100)).cuda() if torch.cuda.is_available() else torch.randint(0, 1000, (2, 100)),
        'puzzle_identifiers': torch.randint(0, 10, (2,)).cuda() if torch.cuda.is_available() else torch.randint(0, 10, (2,))
    }
    
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    print("✓ Forward pass successful")
    
    print("\n🎉 All tests passed! HRM is ready to use with standard PyTorch attention.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

