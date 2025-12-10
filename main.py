import time
import torch
import torch.nn.functional as F
import secretflow as sf
import spu

from data_loader import GISTDataLoader
from models_plain import (
    RandomMatrixRetriever,
    MultiTableRandomRetriever,
    FastHadamardRetriever,
    OptimizedFastHadamardRetriever
)
from models_secret import (
    SecretFastHadamardRetriever,
    SecretOptimizedFastHadamardRetriever
)


def run_benchmark():
    # 1. 初始化 SecretFlow
    sf.shutdown()
    sf.init(['client', 's1', 's2'], address='local')

    # 2. 配置设备
    client = sf.PYU('client')
    s1 = sf.PYU('s1')
    s2 = sf.PYU('s2')

    # 使用 SEMI2K 协议 (半诚实模型)
    spu_device = sf.SPU(sf.utils.testing.cluster_def(['s1', 's2'], runtime_config={
        'protocol': spu.ProtocolKind.SEMI2K,
        'field': spu.FieldType.FM64
    }))

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # >>> 这里是你想要调大的参数 <<<
    BITS = 2048
    TOP_K_EVAL = 100

    print(f"Running Benchmark on: {DEVICE}")
    print(f"Total Hash Bits: {BITS}")
    print("=" * 110)

    # 3. 加载数据
    loader = GISTDataLoader()
    # 仅测试 5 条查询以加快 SPU 模拟速度
    db, qs = loader.load_data(device=DEVICE, test_limit=5)

    print("Computing Ground Truth (Top-10)...")
    scores = torch.mm(qs, db.t())
    _, gt_indices = torch.topk(scores, k=10, largest=True)
    if DEVICE == 'cuda': torch.cuda.synchronize()

    # 4. 运行明文模型
    models = [
        ("1. Random Matrix (Single)", RandomMatrixRetriever(960, BITS, DEVICE), "O(N^2) Mul | MPC不可用"),
        ("2. Random Matrix (Multi) ", MultiTableRandomRetriever(960, BITS, num_tables=4, device=DEVICE),
         "O(N^2) Mul | 对照组"),
        ("3. FWHT (Basic)          ", FastHadamardRetriever(960, BITS, DEVICE), "O(N log N) Add | 极速+无乘法"),
        ("4. FWHT (Optimized Multi)", OptimizedFastHadamardRetriever(960, BITS, num_tables=4, device=DEVICE),
         "O(N log N) Add | 冠军方案"),
    ]

    print(f"\n{'Model Name':<30} | {'Rec@10(Top100)':<15} | {'Build(s)':<9} | {'Query(s)':<9} | {'Algorithm Note'}")
    print("-" * 110)

    model_instances = []

    for name, model, note in models:
        t_start = time.time()
        model.train(db)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        t_build = time.time() - t_start

        t_start = time.time()
        _, pred_indices = model.query(qs, TOP_K_EVAL)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        t_query = time.time() - t_start

        hits = 0
        for i in range(len(qs)):
            gt_set = set(gt_indices[i].tolist())
            pred_set = set(pred_indices[i].tolist())
            hits += len(gt_set & pred_set)
        recall = hits / (len(qs) * 10)

        print(f"{name:<30} | {recall:.2%}           | {t_build:.4f}    | {t_query:.4f}    | {note}")
        model_instances.append(model)

    # 5. 运行 SecretFlow 秘密方案
    print("-" * 110)

    # >>> 修复了这里的元组，添加了第三个 note 字段 <<<
    secret_models = [
        ("5. Secret FWHT (Basic)",
         SecretFastHadamardRetriever(spu_device, model_instances[2], client, s1),
         "SPU MPC Implementation"),

        ("6. Secret FWHT (Opt)  ",
         SecretOptimizedFastHadamardRetriever(spu_device, model_instances[3], client, s1),
         "SPU MPC Implementation"),
    ]

    qs_np = qs.cpu().numpy()

    for name, secret_model, note in secret_models:
        # 构建秘密参数
        t_build = secret_model.build_secret()

        # 执行秘密查询
        fp_plain_np, t_query = secret_model.query_secret(qs_np)

        # 后处理：打包 bits 并进行海明距离检索 (这一步通常在客户端完成)
        fp_tensor = torch.tensor(fp_plain_np, device=DEVICE).bool()
        packed_fp = []

        # 处理不同的维度输出 (单表 vs 多表)
        if len(fp_tensor.shape) == 2:
            # (B, Total_Bits)
            for i in range(0, fp_tensor.shape[1], 64):
                chunk = fp_tensor[:, i:i + 64]
                if chunk.shape[1] < 64: chunk = F.pad(chunk, (0, 64 - chunk.shape[1]))
                packed_fp.append(secret_model.plain_model._pack_bits(chunk).unsqueeze(1))
            q_fp_packed = torch.cat(packed_fp, dim=1)
        else:
            # (B, Tables, Bits_per_table)
            for i in range(0, fp_tensor.shape[2], 64):
                chunk = fp_tensor[:, :, i:i + 64]
                if chunk.shape[2] < 64: chunk = F.pad(chunk, (0, 0, 0, 64 - chunk.shape[2]))
                packed_fp.append(secret_model.plain_model._pack_bits(chunk).unsqueeze(-1))
            q_fp_packed = torch.cat(packed_fp, dim=-1)

        # 检索 top-k
        _, pred_indices = secret_model.plain_model.query_with_fingerprints(q_fp_packed, TOP_K_EVAL)

        # 评估
        hits = 0
        for i in range(len(qs)):
            hits += len(set(gt_indices[i].tolist()) & set(pred_indices[i].tolist()))
        recall = hits / (len(qs) * 10)

        print(f"{name:<30} | {recall:.2%}           | {t_build:.4f}    | {t_query:.4f}    | {note}")

    print("-" * 110)


if __name__ == "__main__":
    run_benchmark()