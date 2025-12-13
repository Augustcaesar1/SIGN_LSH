import time
import warnings
import numpy as np
import secretflow as sf
import jax.numpy as jnp
from functools import partial  # <--- [FIX] 用于绑定静态参数

# ==========================================
# 1. 核心算子: SPU-Friendly FWHT
# ==========================================
def sf_fwht(x, normalize=False):
    """JAX/SPU 版快速沃尔什-哈达玛变换"""
    shape = x.shape
    N = shape[-1]
    
    if (N & (N - 1)) != 0:
        raise ValueError(f"FWHT input dim must be power of 2, got {N}")

    x_flat = x.reshape(-1, N)
    B_total = x_flat.shape[0]
    out = x_flat
    
    h = 1
    while h < N:
        out = out.reshape(B_total, N // (2 * h), 2, h)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        # Stack 优于 Concatenate
        out = jnp.stack([x1 + x2, x1 - x2], axis=2)
        h *= 2
        
    out = out.reshape(shape)
    
    if normalize:
        out = out * (1.0 / jnp.sqrt(N))
        
    return out


# ==========================================
# 2. 秘密 LSH 基类
# ==========================================
class SecretLSHBase:
    def __init__(self, spu, plain_model, alice, bob):
        self.spu = spu
        self.plain_model = plain_model
        self.alice = alice
        self.bob = bob
        self.secret_params = None

    def build_secret(self):
        t_start = time.time()
        self.secret_params = self._prepare_secret_params()
        return time.time() - t_start

    def query_secret(self, qs_np):
        if self.secret_params is None:
            raise ValueError("Run build_secret() first!")
            
        q_sf = sf.to(self.alice, qs_np).to(self.spu)
        t_start = time.time()
        
        # [FIX] 使用 self._run_spu 而不是直接传 self._spu_compute_fingerprint
        # 具体的 SPU 调度逻辑下沉到子类处理，确保不序列化 self
        fp_secret = self._run_spu(q_sf)
        
        fp_plain_np = sf.reveal(fp_secret)
        return fp_plain_np, time.time() - t_start

    def _prepare_secret_params(self): raise NotImplementedError
    def _run_spu(self, q_sf): raise NotImplementedError


# ==========================================
# 3. 统一修正版 FWHT 检索器
# ==========================================
class UnifiedSecretHadamardRetriever(SecretLSHBase):
    def __init__(self, spu, plain_model, alice, bob, 
                 num_tables=None, 
                 use_fwht=True, 
                 use_public_perm=True):
        super().__init__(spu, plain_model, alice, bob)
        
        raw_dim = plain_model.h_dim if hasattr(plain_model, 'h_dim') else plain_model.proj_dim
        self.pad_dim = 1 << (raw_dim - 1).bit_length()
        
        self.use_fwht = use_fwht
        self.use_public_perm = use_public_perm
        
        if num_tables is not None:
            self.num_tables = num_tables
        elif hasattr(plain_model, 'num_tables'):
            self.num_tables = plain_model.num_tables
        else:
            self.num_tables = 1

        print(f"\n[SecretModel Config] Tables={self.num_tables}, FWHT={self.use_fwht}")
        print(f"  - Padding: {raw_dim} -> {self.pad_dim}")
        
        if not self.use_public_perm:
            print("\033[93m[WARNING] use_public_perm=False. Expect severe slowdown!\033[0m")

    def query_secret(self, qs_np):
        if qs_np.shape[1] < self.pad_dim:
            qs_padded = np.pad(qs_np, ((0, 0), (0, self.pad_dim - qs_np.shape[1])))
        else:
            qs_padded = qs_np
        return super().query_secret(qs_padded)

    def _prepare_secret_params(self):
        if hasattr(self.plain_model, 'D_diags'):
            D_np = self.plain_model.D_diags.cpu().numpy()
        else:
            D_np = self.plain_model.D.cpu().numpy()
        
        if D_np.ndim == 1: D_np = D_np[None, :]
            
        if hasattr(self.plain_model, 'perms'):
            perm_np = self.plain_model.perms.cpu().numpy().astype(np.int32)
        else:
            perm_np = self.plain_model.perm.cpu().numpy().astype(np.int32)
            if perm_np.ndim == 1: perm_np = perm_np[None, :]

        curr_tables = min(self.num_tables, D_np.shape[0])
        D_np = D_np[:curr_tables]
        perm_np = perm_np[:curr_tables]

        pad_len = self.pad_dim - D_np.shape[1]
        if pad_len > 0:
            D_np = np.pad(D_np, ((0, 0), (0, pad_len)), constant_values=1.0)

        D_secret = sf.to(self.bob, D_np).to(self.spu)
        
        if self.use_public_perm:
            perm_param = perm_np 
        else:
            perm_param = sf.to(self.bob, perm_np).to(self.spu)

        return [D_secret, perm_param]

    # [FIX] 实现 _run_spu，使用 partial 绑定纯静态函数
    def _run_spu(self, q_sf):
        # 将 use_fwht 作为静态参数绑定到纯函数上
        # 这样 Ray 序列化的是这个 partial 对象，而不是 self
        spu_func = partial(self._pure_spu_compute, use_fwht=self.use_fwht)
        
        # 将函数调度到 SPU
        return self.spu(spu_func)(q_sf, *self.secret_params)

    # [FIX] 纯静态方法，不含 self，也没有副作用
    @staticmethod
    def _pure_spu_compute(x, D_diags, perms, use_fwht):
        """
        静态纯函数，包含了具体的计算逻辑。
        use_fwht 通过 partial 传入。
        """
        B = x.shape[0]
        
        # 1. Rotate
        x_expanded = jnp.expand_dims(x, axis=1)
        D_expanded = jnp.expand_dims(D_diags, axis=0)
        x_rot = x_expanded * D_expanded

        # 2. FWHT (根据 use_fwht 决定)
        if use_fwht:
            proj = sf_fwht(x_rot, normalize=False)
        else:
            proj = x_rot

        # 3. Sample
        gather_idx = jnp.broadcast_to(
            jnp.expand_dims(perms, axis=0),
            (B, perms.shape[0], perms.shape[1])
        )
        final_proj = jnp.take_along_axis(proj, gather_idx, axis=2)
        
        return (final_proj >= 0).astype(jnp.int32)