import pytest
import time
import numpy as np
import torch
import torch.nn.functional as F
import secretflow as sf
import spu  # <--- [FIX 1] 必须导入 spu 包

# 导入你的模型
from models_secret import UnifiedSecretHadamardRetriever
from models_plain import UnifiedLSHRetriever
from data_loader import GISTDataLoader

# ==========================================
# 1. 辅助工具: 真值计算与位打包
# ==========================================
def compute_ground_truth(db, qs, k=100):
    """暴力计算 Top-K 真值 (基于 Cosine 相似度)"""
    print(f"⚡ Computing Ground Truth for {len(qs)} queries...")
    # 假设数据已归一化，使用矩阵乘法计算 Cosine
    scores = torch.mm(qs, db.t())
    _, indices = torch.topk(scores, k=k, largest=True)
    return indices

def pack_secret_output(fp_01_np, plain_model):
    """
    将 SPU 输出的未压缩 0/1 指纹打包成 int64 格式
    """
    device = plain_model.device
    fp_tensor = torch.tensor(fp_01_np, dtype=torch.int64, device=device)
    
    packed_fp = []
    bits_per_table = fp_tensor.shape[-1]
    
    # 按 64 位分块打包
    for i in range(0, bits_per_table, 64):
        chunk = fp_tensor[:, :, i:i + 64]
        if chunk.shape[2] < 64: 
            chunk = F.pad(chunk, (0, 64 - chunk.shape[2]))
        
        packed_chunk = plain_model._pack_bits(chunk)
        packed_fp.append(packed_chunk.unsqueeze(-1))
    
    return torch.cat(packed_fp, dim=-1)

# ==========================================
# 2. 测试环境 (已修正)
# ==========================================
class TestAccuracyAndPerformance:

    @pytest.fixture(scope="class")
    def sf_setup(self):
        """初始化 SecretFlow 环境"""
        # 1. 清理旧环境
        sf.shutdown()
        
        # 2. 初始化 (Standalone 模式)
        sf.init(['alice', 'bob'], address='local')
        alice = sf.PYU('alice')
        bob = sf.PYU('bob')
        
        # [核心修复 2] 使用 spu.ProtocolKind 而非 sf.utils.testing.spu_pb2
        cluster_def = sf.utils.testing.cluster_def(
            ['alice', 'bob'],
            runtime_config={
                'protocol': spu.ProtocolKind.SEMI2K,  # <--- [FIXED] 这里必须用 spu.ProtocolKind
                'field': spu.FieldType.FM64,          # <--- [FIXED] 这里必须用 spu.FieldType
                'enable_pphlo_profile': False
            }
        )
        spu_device = sf.SPU(cluster_def)
        
        yield alice, bob, spu_device
        
        sf.shutdown()

    @pytest.fixture(scope="class")
    def dataset(self):
        """加载数据并计算真值"""
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = GISTDataLoader()
        # 加载较多数据以保证 Recall 计算的统计意义
        db, qs = loader.load_data(device=DEVICE, train_limit=10000, test_limit=100)
        
        # 计算真值 (Top-100)
        gt_indices = compute_ground_truth(db, qs, k=100)
        
        return {
            "db": db, 
            "qs": qs, 
            "gt": gt_indices,
            "device": DEVICE
        }

    @pytest.fixture(scope="class")
    def model_configs(self):
        return [
            # 1. 生产级 (FWHT + PublicPerm)
            {
                "name": "🚀 PublicPerm (Prod)",
                "fwht": True, 
                "public_perm": True,
                "tables": 4
            },
            # 2. 消融实验 (No FWHT)
            {
                "name": "🧪 No-FWHT (Ablation)",
                "fwht": False, 
                "public_perm": True,
                "tables": 4
            },
            # 3. 全隐私 (SecretPerm)
            {
                "name": "🔒 SecretPerm (Basic)",
                "fwht": True, 
                "public_perm": False,
                "tables": 4
            }
        ]

    def test_recall_and_perf(self, sf_setup, dataset, model_configs):
        alice, bob, spu_device = sf_setup
        db, qs, gt_indices = dataset['db'], dataset['qs'], dataset['gt']
        device = dataset['device']
        
        BITS = 2048
        TOP_K = 100
        
        print("\n" + "="*110)
        print(f"📊 FULL BENCHMARK: Recall@{TOP_K} & QPS")
        print("="*110)
        print(f"{'Model Name':<25} | {'Recall':<8} | {'Latency(s)':<10} | {'QPS':<8} | {'Build(s)':<8}")
        print("-" * 110)
        
        for cfg in model_configs:
            # 1. 准备明文模型
            plain_model = UnifiedLSHRetriever(
                input_dim=960, 
                total_bits=BITS, 
                num_tables=cfg['tables'], 
                projection_type='hadamard',
                device=device
            )
            plain_model.train(db)
            
            # 2. 实例化秘密模型
            secret_model = UnifiedSecretHadamardRetriever(
                spu_device, plain_model, alice, bob,
                num_tables=cfg['tables'],
                use_fwht=cfg['fwht'],
                use_public_perm=cfg['public_perm']
            )
            
            # 3. Build 计时
            t_build = secret_model.build_secret()
            
            # 4. Query & Recall
            qs_np = qs.cpu().numpy()
            
            try:
                # SecretPerm 模式下只测少量数据防超时
                if not cfg['public_perm']:
                    qs_subset = qs_np[:5]
                    gt_subset = gt_indices[:5]
                    bs = 5
                else:
                    qs_subset = qs_np
                    gt_subset = gt_indices
                    bs = len(qs_np)

                # --- 核心计时 ---
                fp_01, t_query = secret_model.query_secret(qs_subset)
                # ----------------
                
                # 计算 Recall
                q_fp_packed = pack_secret_output(fp_01, plain_model)
                _, pred_indices = plain_model.query_with_fingerprints(q_fp_packed, k=TOP_K)
                
                hits = 0
                for i in range(bs):
                    gt_set = set(gt_subset[i].tolist())
                    pred_set = set(pred_indices[i].tolist())
                    hits += len(gt_set & pred_set)
                
                recall = hits / (bs * TOP_K)
                latency = t_query / bs
                qps = bs / t_query
                
                print(f"{cfg['name']:<25} | {recall:.2%}   | {latency:.4f}     | {qps:.2f}     | {t_build:.4f}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{cfg['name']:<25} | ERROR: {str(e)[:30]}...")

    print("-" * 110)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])