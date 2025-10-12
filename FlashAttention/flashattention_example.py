# FlashAttention Example Implementation in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class FlashAttention(nn.Module):
    """
    A simplified PyTorch implementation of FlashAttention for educational purposes.
    This demonstrates the core concepts but doesn't include CUDA optimizations.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
        block_size_q: int = 64,
        block_size_kv: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing FlashAttention tiling strategy
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply FlashAttention tiling
        output = self._flash_attention_tiled(q, k, v)
        
        # Transpose back and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(output)
    
    def _flash_attention_tiled(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Tiled attention computation implementing FlashAttention algorithm
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize output and statistics
        output = torch.zeros_like(q)
        m = torch.full((batch_size, num_heads, seq_len), float('-inf'), device=q.device)
        l = torch.zeros((batch_size, num_heads, seq_len), device=q.device)
        
        # Tile over sequence length
        num_q_blocks = (seq_len + self.block_size_q - 1) // self.block_size_q
        num_kv_blocks = (seq_len + self.block_size_kv - 1) // self.block_size_kv
        
        for i in range(num_q_blocks):
            q_start = i * self.block_size_q
            q_end = min((i + 1) * self.block_size_q, seq_len)
            q_block = q[:, :, q_start:q_end, :]
            
            # Initialize block statistics
            o_block = torch.zeros_like(q_block)
            m_block = torch.full((batch_size, num_heads, q_end - q_start), 
                               float('-inf'), device=q.device)
            l_block = torch.zeros((batch_size, num_heads, q_end - q_start), device=q.device)
            
            for j in range(num_kv_blocks):
                kv_start = j * self.block_size_kv
                kv_end = min((j + 1) * self.block_size_kv, seq_len)
                
                # Skip future tokens for causal attention
                if self.causal and kv_start > q_end - 1:
                    continue
                
                k_block = k[:, :, kv_start:kv_end, :]
                v_block = v[:, :, kv_start:kv_end, :]
                
                # Compute attention scores for this block
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                # Apply causal mask if needed
                if self.causal:
                    causal_mask = torch.triu(
                        torch.ones(q_end - q_start, kv_end - kv_start, device=q.device),
                        diagonal=kv_start - q_start + 1
                    )
                    scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
                
                # Online softmax computation (FlashAttention's key innovation)
                m_new = torch.maximum(m_block, scores.max(dim=-1)[0])
                
                # Compute exponentials with numerical stability
                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                exp_m_diff = torch.exp(m_block - m_new)
                
                # Update statistics
                l_new = exp_m_diff * l_block + exp_scores.sum(dim=-1)
                
                # Update output
                o_block = (exp_m_diff.unsqueeze(-1) * o_block + 
                          torch.matmul(exp_scores, v_block)) / l_new.unsqueeze(-1)
                
                m_block = m_new
                l_block = l_new
            
            output[:, :, q_start:q_end, :] = o_block
            
        return output

# Example usage
def example_flashattention_usage():
    """Example of how to use FlashAttention"""
    
    # Model parameters
    batch_size = 2
    seq_len = 1024
    dim = 512
    num_heads = 8
    
    # Create model and input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlashAttention(
        dim=dim,
        num_heads=num_heads,
        causal=True,  # For autoregressive modeling
        block_size_q=64,
        block_size_kv=64,
    ).to(device)
    
    # Random input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Memory efficient attention completed!")
    
    return model, output

# Alternative: Using PyTorch's built-in FlashAttention
def pytorch_flashattention_example():
    """Using PyTorch's native FlashAttention implementation"""
    
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create Q, K, V tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Use PyTorch's scaled_dot_product_attention with FlashAttention backend
    with torch.backends.cuda.sdp_kernel(enable_flash=True):
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # Enables causal masking
        )
    
    print(f"PyTorch FlashAttention output shape: {output.shape}")
    return output

if __name__ == "__main__":
    print("=== FlashAttention Examples ===")
    
    # Example 1: Custom implementation
    print("\n1. Custom FlashAttention Implementation:")
    model, output = example_flashattention_usage()
    
    # Example 2: PyTorch native
    print("\n2. PyTorch Native FlashAttention:")
    pytorch_output = pytorch_flashattention_example()