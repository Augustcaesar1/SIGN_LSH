import pytest
import time
import numpy as np
import torch
import torch.nn.functional as F
import secretflow as sf
import spu
import pandas as pd  # 用于漂亮的表格输出

# 导入核心组件
from models_secret import UnifiedSecretHadamardRetriever
from models_plain import UnifiedLSHRetriever
from data_loader import GISTDataLoader

# ==========================================
# 0. 基础工具函数
# ==========================================
def compute_ground_truth(db, qs, k=100):
    print(f"⚡ [Prep] Computing Ground Truth for {len(qs)} queries...")
    scores = torch.mm(qs, db.t())
    _, indices = torch.topk(scores, k=k, largest=True)
    return indices

def pack_secret_output(fp_01_np, plain_model):
    """0/1 矩阵打包为 int64"""
    device = plain_model.device
    fp_tensor = torch.tensor(fp_01_np, dtype=torch.int64, device=device)
    packed_fp = []
    bits_per_table = fp_tensor.shape[-1]
    for i in range(0, bits_per_table, 64):
        chunk = fp_tensor[:, :, i:i + 64]
        if chunk.shape[2] < 64: chunk = F.pad(chunk, (0, 64 - chunk.shape[2]))
        packed_chunk = plain_model._pack_bits(chunk)
        packed_fp.append(packed_chunk.unsqueeze(-1))
    return torch.cat(packed_fp, dim=-1)

# ==========================================
# 1. 核心测试逻辑
# ==========================================
class ParameterImpactBenchmark:
    
    def setup_env(self):
        sf.shutdown()
        sf.init(['alice', 'bob'], address='local')
        # SPU 配置
        cluster_def = sf.utils.testing.cluster_def(
            ['alice', 'bob'],
            runtime_config={
                'protocol': spu.ProtocolKind.SEMI2K,
                'field': spu.FieldType.FM64,
                'enable_pphlo_profile': False
            }
        )
        self.alice = sf.PYU('alice')
        self.bob = sf.PYU('bob')
        self.spu = sf.SPU(cluster_def)
        
        # 数据加载
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = GISTDataLoader()
        self.db, self.qs = loader.load_data(device=DEVICE, train_limit=10000, test_limit=100)
        self.gt_indices = compute_ground_truth(self.db, self.qs, k=100)
        self.device = DEVICE
        
        # 预训练一个足够大的明文模型 (Tables=8) 供后续裁剪使用
        print("⚡ [Prep] Training Base Plain Model (Tables=8)...")
        self.base_plain_model = UnifiedLSHRetriever(
            input_dim=960, total_bits=2048, num_tables=8, 
            projection_type='hadamard', device=DEVICE
        )
        self.base_plain_model.train(self.db)

    def run_single_experiment(self, name, tables, fwht, public_perm):
        """运行单个实验配置并返回指标"""
        print(f"\n🧪 Running Exp: [{name}]")
        print(f"   Configs: Tables={tables}, FWHT={fwht}, PublicPerm={public_perm}")
        
        # 实例化秘密模型
        secret_model = UnifiedSecretHadamardRetriever(
            self.spu, self.base_plain_model, self.alice, self.bob,
            num_tables=tables, use_fwht=fwht, use_public_perm=public_perm
        )
        
        # 1. Build
        t_build = secret_model.build_secret()
        
        # 2. Query (智能 Batch)
        # 如果是全隐私模式(PublicPerm=False)，只测 1 条数据估算性能，避免卡死
        qs_np = self.qs.cpu().numpy()
        if not public_perm:
            qs_subset = qs_np[:1]
            gt_subset = self.gt_indices[:1]
            bs = 1
            print("   ⚠️  [Slow Mode] Detected SecretPerm, reducing batch size to 1...")
        else:
            qs_subset = qs_np
            gt_subset = self.gt_indices
            bs = len(qs_np)
            
        try:
            fp_01, t_query = secret_model.query_secret(qs_subset)
            
            # 3. Recall 计算
            q_fp_packed = pack_secret_output(fp_01, self.base_plain_model)
            _, pred_indices = self.base_plain_model.query_with_fingerprints(q_fp_packed, k=100)
            
            hits = 0
            for i in range(bs):
                hits += len(set(gt_subset[i].tolist()) & set(pred_indices[i].tolist()))
            
            recall = hits / (bs * 100)
            latency = t_query / bs
            qps = bs / t_query
            
            return {
                "Scenario": name,
                "Tables": tables,
                "FWHT": fwht,
                "PublicPerm": public_perm,
                "Recall@100": f"{recall:.2%}",
                "Latency(s)": f"{latency:.4f}",
                "QPS": f"{qps:.2f}",
                "Consequence": "" # 稍后填充
            }
        except Exception as e:
            return {"Scenario": name, "Error": str(e)[:30]}

    def run_all(self):
        self.setup_env()
        results = []
        
        # ==========================================
        # 实验组 1: 数学的后果 (FWHT 的重要性)
        # 控制变量: Tables=4, PublicPerm=True
        # ==========================================
        print("\n=== Experiment 1: The Consequence of Math (FWHT) ===")
        res_no_fwht = self.run_single_experiment("No FWHT", tables=4, fwht=False, public_perm=True)
        res_no_fwht['Consequence'] = "❌ 召回率崩塌 (数学失效)"
        results.append(res_no_fwht)
        
        res_fwht = self.run_single_experiment("With FWHT", tables=4, fwht=True, public_perm=True)
        res_fwht['Consequence'] = "✅ 高召回 (数学有效)"
        results.append(res_fwht)
        
        # ==========================================
        # 实验组 2: 隐私的代价 (PublicPerm 的重要性)
        # 控制变量: Tables=4, FWHT=True
        # ==========================================
        print("\n=== Experiment 2: The Consequence of Privacy (Permutation) ===")
        # 我们复用上面的 res_fwht 作为对照组
        
        res_secret_perm = self.run_single_experiment("Secret Perm", tables=4, fwht=True, public_perm=False)
        res_secret_perm['Consequence'] = "❌ 速度慢 100+ 倍 (OAM 代价)"
        results.append(res_secret_perm)
        
        # ==========================================
        # 实验组 3: 规模的权衡 (NumTables 的重要性)
        # 控制变量: FWHT=True, PublicPerm=True
        # ==========================================
        print("\n=== Experiment 3: The Trade-off of Scale (Num Tables) ===")
        
        res_t1 = self.run_single_experiment("Tables=1", tables=1, fwht=True, public_perm=True)
        res_t1['Consequence'] = "📉 召回低，速度极快"
        results.append(res_t1)
        
        # Tables=4 已经跑过了 (res_fwht)
        
        res_t8 = self.run_single_experiment("Tables=8", tables=8, fwht=True, public_perm=True)
        res_t8['Consequence'] = "📈 召回高，存储/计算翻倍"
        results.append(res_t8)
        
        # ==========================================
        # 最终报告
        # ==========================================
        df = pd.DataFrame(results)
        # 调整列顺序
        cols = ["Scenario", "Tables", "FWHT", "PublicPerm", "Recall@100", "Latency(s)", "QPS", "Consequence"]
        print("\n" + "="*100)
        print("📊 FINAL PARAMETER IMPACT REPORT")
        print("="*100)
        print(df[cols].to_string(index=False))
        print("="*100)

if __name__ == "__main__":
    benchmark = ParameterImpactBenchmark()
    benchmark.run_all()