# PagedAttention Example Implementation for vLLM-style Serving

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass

@dataclass
class PagedAttentionConfig:
    """Configuration for PagedAttention"""
    block_size: int = 16  # Number of tokens per block
    max_blocks: int = 1024  # Maximum number of blocks in memory
    num_heads: int = 8
    head_dim: int = 64
    dtype: torch.dtype = torch.float16

class BlockTable:
    """
    Block table for managing logical to physical block mapping
    Similar to OS page tables
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.logical_to_physical: Dict[int, List[int]] = {}  # seq_id -> [physical_block_ids]
        self.physical_blocks_used: set = set()
        self.free_blocks: List[int] = list(range(config.max_blocks))
        
    def allocate_blocks(self, seq_id: int, num_blocks: int) -> List[int]:
        """Allocate physical blocks for a sequence"""
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"Not enough free blocks. Need {num_blocks}, have {len(self.free_blocks)}")
        
        allocated_blocks = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            allocated_blocks.append(block_id)
            self.physical_blocks_used.add(block_id)
        
        if seq_id not in self.logical_to_physical:
            self.logical_to_physical[seq_id] = []
        self.logical_to_physical[seq_id].extend(allocated_blocks)
        
        return allocated_blocks
    
    def get_physical_blocks(self, seq_id: int) -> List[int]:
        """Get physical blocks for a sequence"""
        return self.logical_to_physical.get(seq_id, [])
    
    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence"""
        if seq_id in self.logical_to_physical:
            blocks = self.logical_to_physical[seq_id]
            for block_id in blocks:
                self.physical_blocks_used.remove(block_id)
                self.free_blocks.append(block_id)
            del self.logical_to_physical[seq_id]

class PagedKVCache:
    """
    Paged KV Cache implementation inspired by vLLM
    """
    
    def __init__(self, config: PagedAttentionConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Allocate physical memory for all blocks
        # Shape: (max_blocks, block_size, num_heads, head_dim)
        self.key_cache = torch.zeros(
            config.max_blocks, config.block_size, config.num_heads, config.head_dim,
            dtype=config.dtype, device=device
        )
        self.value_cache = torch.zeros(
            config.max_blocks, config.block_size, config.num_heads, config.head_dim,
            dtype=config.dtype, device=device
        )
        
        self.block_table = BlockTable(config)
    
    def store_kv(self, seq_id: int, keys: torch.Tensor, values: torch.Tensor, 
                 start_pos: int = 0):
        """
        Store keys and values in paged cache
        
        Args:
            seq_id: Sequence identifier
            keys: Key tensor of shape (seq_len, num_heads, head_dim)
            values: Value tensor of shape (seq_len, num_heads, head_dim)
            start_pos: Starting position in the sequence
        """
        seq_len = keys.shape[0]
        
        # Calculate number of blocks needed
        total_tokens = start_pos + seq_len
        num_blocks_needed = (total_tokens + self.config.block_size - 1) // self.config.block_size
        
        # Get existing blocks or allocate new ones
        existing_blocks = self.block_table.get_physical_blocks(seq_id)
        if len(existing_blocks) < num_blocks_needed:
            new_blocks = self.block_table.allocate_blocks(
                seq_id, num_blocks_needed - len(existing_blocks)
            )
        
        physical_blocks = self.block_table.get_physical_blocks(seq_id)
        
        # Store keys and values in blocks
        for i, (key, value) in enumerate(zip(keys, values)):
            token_pos = start_pos + i
            block_idx = token_pos // self.config.block_size
            pos_in_block = token_pos % self.config.block_size
            physical_block = physical_blocks[block_idx]
            
            self.key_cache[physical_block, pos_in_block] = key
            self.value_cache[physical_block, pos_in_block] = value
    
    def get_kv(self, seq_id: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve keys and values for a sequence
        
        Returns:
            keys: (seq_len, num_heads, head_dim)
            values: (seq_len, num_heads, head_dim)
        """
        physical_blocks = self.block_table.get_physical_blocks(seq_id)
        
        # Calculate number of blocks needed
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size
        
        keys = []
        values = []
        
        for token_pos in range(seq_len):
            block_idx = token_pos // self.config.block_size
            pos_in_block = token_pos % self.config.block_size
            
            if block_idx < len(physical_blocks):
                physical_block = physical_blocks[block_idx]
                key = self.key_cache[physical_block, pos_in_block]
                value = self.value_cache[physical_block, pos_in_block]
                keys.append(key)
                values.append(value)
        
        return torch.stack(keys), torch.stack(values)

class PagedAttention(nn.Module):
    """
    PagedAttention implementation for efficient LLM serving
    """
    
    def __init__(self, config: PagedAttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.head_dim ** -0.5
        
        # Linear projections
        hidden_size = config.num_heads * config.head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: PagedKVCache,
        seq_ids: List[int],
        input_positions: List[int],
        is_prefill: bool = False
    ) -> torch.Tensor:
        """
        PagedAttention forward pass
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            kv_cache: Paged KV cache
            seq_ids: List of sequence IDs for each batch item
            input_positions: Current positions in each sequence
            is_prefill: Whether this is prefill or decode phase
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to queries
        queries = self.q_proj(hidden_states)
        queries = queries.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        
        if is_prefill:
            # Prefill phase: compute K, V and store in cache
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)
            
            keys = keys.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
            values = values.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
            
            # Store in paged cache
            for i, seq_id in enumerate(seq_ids):
                kv_cache.store_kv(
                    seq_id, 
                    keys[i], 
                    values[i], 
                    start_pos=input_positions[i]
                )
            
            # Use current keys and values for attention
            attention_output = self._compute_attention(queries, keys, values)
        
        else:
            # Decode phase: only compute current queries, retrieve K,V from cache
            current_keys = self.k_proj(hidden_states)
            current_values = self.v_proj(hidden_states)
            
            current_keys = current_keys.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
            current_values = current_values.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
            
            # Store current K,V in cache
            for i, seq_id in enumerate(seq_ids):
                kv_cache.store_kv(
                    seq_id,
                    current_keys[i],
                    current_values[i],
                    start_pos=input_positions[i]
                )
            
            # Retrieve all keys and values from cache
            all_keys = []
            all_values = []
            
            for i, seq_id in enumerate(seq_ids):
                cache_len = input_positions[i] + seq_len
                cached_keys, cached_values = kv_cache.get_kv(seq_id, cache_len)
                all_keys.append(cached_keys)
                all_values.append(cached_values)
            
            # Pad sequences to same length for batched computation
            max_len = max(k.shape[0] for k in all_keys)
            
            padded_keys = []
            padded_values = []
            
            for keys, values in zip(all_keys, all_values):
                if keys.shape[0] < max_len:
                    pad_len = max_len - keys.shape[0]
                    padded_k = F.pad(keys, (0, 0, 0, 0, 0, pad_len))
                    padded_v = F.pad(values, (0, 0, 0, 0, 0, pad_len))
                else:
                    padded_k = keys
                    padded_v = values
                
                padded_keys.append(padded_k)
                padded_values.append(padded_v)
            
            cached_keys = torch.stack(padded_keys)
            cached_values = torch.stack(padded_values)
            
            # Compute attention with cached K,V
            attention_output = self._compute_attention_with_cache(
                queries, cached_keys, cached_values, input_positions
            )
        
        # Output projection
        attention_output = attention_output.view(batch_size, seq_len, -1)
        return self.out_proj(attention_output)
    
    def _compute_attention(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> torch.Tensor:
        """Standard attention computation"""
        # queries, keys, values: (batch_size, seq_len, num_heads, head_dim)
        
        queries = queries.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        seq_len = queries.shape[2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=queries.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values)
        
        return output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
    
    def _compute_attention_with_cache(
        self,
        queries: torch.Tensor,
        cached_keys: torch.Tensor,
        cached_values: torch.Tensor,
        positions: List[int]
    ) -> torch.Tensor:
        """Attention computation using cached K,V"""
        batch_size, seq_len, num_heads, head_dim = queries.shape
        
        queries = queries.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        cached_keys = cached_keys.transpose(1, 2)
        cached_values = cached_values.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(queries, cached_keys.transpose(-2, -1)) * self.scale
        
        # Apply causal mask based on positions
        for i, pos in enumerate(positions):
            mask_len = pos + seq_len
            causal_mask = torch.triu(
                torch.ones(seq_len, mask_len, device=queries.device), 
                diagonal=pos + 1
            )
            scores[i, :, :, :mask_len] = scores[i, :, :, :mask_len].masked_fill(
                causal_mask.bool(), float('-inf')
            )
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, cached_values)
        
        return output.transpose(1, 2)

# Example usage
def example_pagedattention_usage():
    """Example of how to use PagedAttention"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = PagedAttentionConfig(
        block_size=16,
        max_blocks=1024,
        num_heads=8,
        head_dim=64,
        dtype=torch.float16
    )
    
    # Initialize PagedAttention and KV cache
    model = PagedAttention(config).to(device)
    kv_cache = PagedKVCache(config, device)
    
    # Example: Process two sequences
    seq_ids = [0, 1]
    
    # Prefill phase
    print("=== Prefill Phase ===")
    prefill_tokens = 50
    hidden_size = config.num_heads * config.head_dim
    
    prefill_input = torch.randn(
        2, prefill_tokens, hidden_size, 
        dtype=config.dtype, device=device
    )
    
    # Both sequences start at position 0
    prefill_positions = [0, 0]
    
    prefill_output = model(
        prefill_input,
        kv_cache,
        seq_ids,
        prefill_positions,
        is_prefill=True
    )
    
    print(f"Prefill input shape: {prefill_input.shape}")
    print(f"Prefill output shape: {prefill_output.shape}")
    
    # Decode phase - generate tokens one by one
    print("\n=== Decode Phase ===")
    current_positions = [prefill_tokens, prefill_tokens]
    
    for step in range(5):  # Generate 5 tokens
        decode_input = torch.randn(
            2, 1, hidden_size,
            dtype=config.dtype, device=device
        )
        
        decode_output = model(
            decode_input,
            kv_cache,
            seq_ids,
            current_positions,
            is_prefill=False
        )
        
        print(f"Decode step {step + 1}: input {decode_input.shape} -> output {decode_output.shape}")
        
        # Update positions
        current_positions = [pos + 1 for pos in current_positions]
    
    # Memory usage information
    total_blocks_used = len(kv_cache.block_table.physical_blocks_used)
    total_blocks_available = config.max_blocks
    memory_utilization = total_blocks_used / total_blocks_available * 100
    
    print(f"\n=== Memory Usage ===")
    print(f"Blocks used: {total_blocks_used}/{total_blocks_available}")
    print(f"Memory utilization: {memory_utilization:.1f}%")
    print(f"Memory waste: ~{(1 - memory_utilization/100) * 100:.1f}% (near-zero fragmentation)")
    
    return model, kv_cache

if __name__ == "__main__":
    print("=== PagedAttention Example ===")
    model, kv_cache = example_pagedattention_usage()