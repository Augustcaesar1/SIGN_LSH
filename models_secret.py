import time
import numpy as np
import secretflow as sf
import jax.numpy as jnp
from lsh_core import sf_fwht


# 秘密 LSH 基类
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
        t_cost = time.time() - t_start
        return t_cost

    def query_secret(self, qs_np):
        if self.secret_params is None:
            raise ValueError("Run build_secret() first!")
        # 客户端输入查询
        q_sf = sf.to(self.alice, qs_np).to(self.spu)
        t_start = time.time()
        # SPU 计算
        fp_secret = self.spu(self._spu_compute_fingerprint)(q_sf, *self.secret_params)
        fp_plain_np = sf.reveal(fp_secret)
        t_cost = time.time() - t_start
        return fp_plain_np, t_cost

    def _prepare_secret_params(self): raise NotImplementedError

    def _spu_compute_fingerprint(self, x, *args): raise NotImplementedError


# --- 模型 E: 秘密基础 FWHT ---
class SecretFastHadamardRetriever(SecretLSHBase):
    def __init__(self, spu, plain_model, alice, bob):
        super().__init__(spu, plain_model, alice, bob)
        self.pad_dim = plain_model.h_dim

    def query_secret(self, qs_np):
        if qs_np.shape[1] < self.pad_dim:
            qs_padded = np.pad(qs_np, ((0, 0), (0, self.pad_dim - qs_np.shape[1])))
        else:
            qs_padded = qs_np
        return super().query_secret(qs_padded)

    def _prepare_secret_params(self):
        D_np = self.plain_model.D.cpu().numpy()
        pad_len = max(0, self.pad_dim - len(D_np))
        D_np = np.pad(D_np, (0, pad_len), constant_values=1.0)
        perm_np = self.plain_model.perm.cpu().numpy().astype(np.int32)
        # 参数属于服务端 Bob
        return [sf.to(self.bob, D_np).to(self.spu), sf.to(self.bob, perm_np).to(self.spu)]

    @staticmethod
    def _spu_compute_fingerprint(x, D, perm):
        x_rot = x * D
        proj = sf_fwht(x_rot, normalize=False)
        proj_sampled = jnp.take(proj, perm, axis=1)
        return (proj_sampled >= 0).astype(jnp.int32)


# --- 模型 F: 秘密优化多表 FWHT (SecretOptimizedFastHadamardRetriever) ---
class SecretOptimizedFastHadamardRetriever(SecretLSHBase):
    def __init__(self, spu, plain_model, alice, bob):
        super().__init__(spu, plain_model, alice, bob)
        self.pad_dim = plain_model.h_dim

    def query_secret(self, qs_np):
        if qs_np.shape[1] < self.pad_dim:
            qs_padded = np.pad(qs_np, ((0, 0), (0, self.pad_dim - qs_np.shape[1])))
        else:
            qs_padded = qs_np
        return super().query_secret(qs_padded)

    def _prepare_secret_params(self):
        D_diags_np = self.plain_model.D_diags.cpu().numpy()
        pad_len = max(0, self.pad_dim - D_diags_np.shape[1])
        D_diags_np = np.pad(D_diags_np, ((0, 0), (0, pad_len)), constant_values=1.0)
        perms_np = self.plain_model.perms.cpu().numpy().astype(np.int32)
        return [sf.to(self.bob, D_diags_np).to(self.spu), sf.to(self.bob, perms_np).to(self.spu)]

    @staticmethod
    def _spu_compute_fingerprint(x, D_diags, perms):
        B = x.shape[0]
        T = D_diags.shape[0]
        # x: (B, H), D: (T, H) -> x_expanded: (B, 1, H)
        x_expanded = jnp.expand_dims(x, axis=1)
        x_rot = x_expanded * jnp.expand_dims(D_diags, axis=0)

        # SPU FWHT
        proj = sf_fwht(x_rot, normalize=False)

        # Gather/Take
        gather_idx = jnp.broadcast_to(jnp.expand_dims(perms, axis=0), (B, T, perms.shape[-1]))
        final_proj = jnp.take_along_axis(proj, gather_idx, axis=2)
        return (final_proj >= 0).astype(jnp.int32)


# --- 模型 G: 半隐私秘密基础 FWHT  ---
class SecretFastHadamardRetriever_PublicPerm(SecretLSHBase):
    def __init__(self, spu, plain_model, alice, bob):
        super().__init__(spu, plain_model, alice, bob)
        self.pad_dim = plain_model.h_dim

    def query_secret(self, qs_np):
        if qs_np.shape[1] < self.pad_dim:
            qs_padded = np.pad(qs_np, ((0, 0), (0, self.pad_dim - qs_np.shape[1])))
        else:
            qs_padded = qs_np
        return super().query_secret(qs_padded)

    def _prepare_secret_params(self):
        D_np = self.plain_model.D.cpu().numpy()
        pad_len = max(0, self.pad_dim - len(D_np))
        D_np = np.pad(D_np, (0, pad_len), constant_values=1.0)
        perm_np = self.plain_model.perm.cpu().numpy().astype(np.int32)
        return [sf.to(self.bob, D_np).to(self.spu), perm_np]

    @staticmethod
    def _spu_compute_fingerprint(x, D, perm_plain):
        x_rot = x * D
        proj = sf_fwht(x_rot, normalize=False)
        proj_sampled = jnp.take(proj, perm_plain, axis=1)
        return (proj_sampled >= 0).astype(jnp.int32)


# --- 模型 H: 半隐私秘密优化多表 FWHT  ---
class SecretOptimizedFastHadamardRetriever_PublicPerm(SecretLSHBase):
    def __init__(self, spu, plain_model, alice, bob):
        super().__init__(spu, plain_model, alice, bob)
        self.pad_dim = plain_model.h_dim

    def query_secret(self, qs_np):
        if qs_np.shape[1] < self.pad_dim:
            qs_padded = np.pad(qs_np, ((0, 0), (0, self.pad_dim - qs_np.shape[1])))
        else:
            qs_padded = qs_np
        return super().query_secret(qs_padded)

    def _prepare_secret_params(self):
        D_diags_np = self.plain_model.D_diags.cpu().numpy()
        pad_len = max(0, self.pad_dim - D_diags_np.shape[1])
        D_diags_np = np.pad(D_diags_np, ((0, 0), (0, pad_len)), constant_values=1.0)
        perms_np = self.plain_model.perms.cpu().numpy().astype(np.int32)
        return [sf.to(self.bob, D_diags_np).to(self.spu), perms_np]

    @staticmethod
    def _spu_compute_fingerprint(x, D_diags, perms_plain):
        B = x.shape[0]
        x_expanded = jnp.expand_dims(x, axis=1)
        x_rot = x_expanded * jnp.expand_dims(D_diags, axis=0)
        proj = sf_fwht(x_rot, normalize=False)
        gather_idx = jnp.broadcast_to(jnp.expand_dims(perms_plain, axis=0),(B, perms_plain.shape[0], perms_plain.shape[1]))
        final_proj = jnp.take_along_axis(proj, gather_idx, axis=2)
        return (final_proj >= 0).astype(jnp.int32)