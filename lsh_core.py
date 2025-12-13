import torch
import torch.nn.functional as F
import jax.numpy as jnp


# --- 核心数学函数 ---

def fast_walsh_hadamard_transform(x, normalize=True):
    """ PyTorch 版 FWHT """
    B, N = x.shape
    if (N & (N - 1)) != 0:
        raise ValueError(f"Dim {N} must be power of 2 for FWHT.")
    out = x.clone()
    h = 1
    while h < N:
        out = out.view(B, N // (2 * h), 2, h)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        out = torch.cat((x1 + x2, x1 - x2), dim=2)
        h *= 2
    out = out.view(B, N)
    if normalize:
        out = out / torch.sqrt(torch.tensor(N, float))
    return out


# ==========================================
# 1. 核心算子: SPU-Friendly FWHT
# ==========================================
def sf_fwht(x, normalize=False):
    """
    JAX/SPU 版快速沃尔什-哈达玛变换 (FWHT)
    修正了 Reshape/Stack 逻辑以确保数学正确性和 SPU 编译器友好性。
    """
    shape = x.shape
    N = shape[-1]
    
    # 检查维度是否为 2 的幂 (FWHT 硬性要求)
    if (N & (N - 1)) != 0:
        raise ValueError(f"FWHT input dim must be power of 2, got {N}")

    # 将最后两维展平处理，支持 Batch
    x_flat = x.reshape(-1, N)
    B_total = x_flat.shape[0]
    out = x_flat
    
    h = 1
    while h < N:
        # [Correctness Fix] 使用 stack 而非 concatenate 以明确内存布局
        # 这在 SPU IR 中更容易被优化为 View 操作
        out = out.reshape(B_total, N // (2 * h), 2, h)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        
        # Butterfly Operation
        # stack axis=2 -> (B, N/2h, 2, h)
        out = jnp.stack([x1 + x2, x1 - x2], axis=2)
        h *= 2
        
    out = out.reshape(shape)
    
    # [Theory Fix] 可选归一化，保持正交性
    # 注意: 在 Sign-LSH 中即使不归一化也不影响符号位，但在 MPC 中除法较慢
    if normalize:
        out = out * (1.0 / jnp.sqrt(N)) # 使用乘法替代除法加速
        
    return out

