import time
import warnings
import numpy as np
import secretflow as sf
import jax.numpy as jnp
from lsh_core import sf_fwht

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
        fp_secret = self.spu(self._spu_compute_fingerprint)(q_sf, *self.secret_params)
        fp_plain_np = sf.reveal(fp_secret)
        
        return fp_plain_np, time.time() - t_start

    def _prepare_secret_params(self): raise NotImplementedError
    def _spu_compute_fingerprint(self, x, *args): raise NotImplementedError


# ==========================================
# 3. 统一修正版 FWHT 检索器
# ==========================================
class UnifiedSecretHadamardRetriever(SecretLSHBase):
    def __init__(self, spu, plain_model, alice, bob, 
                 num_tables=None, 
                 use_fwht=True, 
                 use_public_perm=True):
        super().__init__(spu, plain_model, alice, bob)
        
        # Padding 计算
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
        # =======================================================
        # [FIX] 智能处理 D 的维度
        # =======================================================
        if hasattr(self.plain_model, 'D_diags'):
            D_np = self.plain_model.D_diags.cpu().numpy()
        else:
            D_np = self.plain_model.D.cpu().numpy()
        
        # 核心修复：只有当它是 1D 数组时，才增加维度
        # 如果已经是 (Tables, Dim) 的 2D 数组，则保持原样
        if D_np.ndim == 1:
            D_np = D_np[None, :]
            
        # 2. 准备置换索引 Perm
        if hasattr(self.plain_model, 'perms'):
            perm_np = self.plain_model.perms.cpu().numpy().astype(np.int32)
        else:
            perm_np = self.plain_model.perm.cpu().numpy().astype(np.int32)
            if perm_np.ndim == 1:
                perm_np = perm_np[None, :]

        # 3. 截断与 Padding
        curr_tables = min(self.num_tables, D_np.shape[0])
        D_np = D_np[:curr_tables]
        perm_np = perm_np[:curr_tables]

        pad_len = self.pad_dim - D_np.shape[1]
        
        # 执行 Padding (确保此时 D_np 肯定是 2D)
        if pad_len > 0:
            D_np = np.pad(D_np, ((0, 0), (0, pad_len)), constant_values=1.0)

        # 4. 参数分发
        D_secret = sf.to(self.bob, D_np).to(self.spu)
        
        if self.use_public_perm:
            perm_param = perm_np 
        else:
            perm_param = sf.to(self.bob, perm_np).to(self.spu)

        return [D_secret, perm_param]

    def _spu_compute_fingerprint(self, x, D_diags, perms):
        B = x.shape[0]
        
        # 1. Rotate (x * D)
        x_expanded = jnp.expand_dims(x, axis=1)
        D_expanded = jnp.expand_dims(D_diags, axis=0)
        x_rot = x_expanded * D_expanded

        # 2. FWHT
        if self.use_fwht:
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