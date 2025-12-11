import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock

# 导入要测试的模块
from models_secret import (
    SecretFastHadamardRetriever,
    SecretOptimizedFastHadamardRetriever,
    SecretFastHadamardRetriever_PublicPerm,
    SecretOptimizedFastHadamardRetriever_PublicPerm
)
from data_loader import GISTDataLoader


class MockPlainModel:
    """模拟 PlainModel 对象 - GIST-960 数据集适配"""
    def __init__(self, h_dim=512, num_tables=4, num_bits=64):
        # GIST-960 数据集配置
        self.h_dim = h_dim
        self.input_dim = 960  # GIST-960 的输入维度
        self.D = torch.randn(h_dim)
        self.perm = torch.randperm(h_dim)
        self.D_diags = torch.randn(num_tables, h_dim)
        self.perms = torch.randperm(h_dim).unsqueeze(0).repeat(num_tables, 1)


class MockSPU:
    """模拟 SPU 执行环境"""
    def __init__(self):
        pass
    
    def __call__(self, func):
        """模拟 SPU 计算"""
        def wrapper(*args, **kwargs):
            # 直接执行函数（不做实际秘密计算）
            return func(*args, **kwargs)
        return wrapper


class MockPlainModel:
    """模拟 PlainModel 对象 - GIST-960 数据集适配"""
    def __init__(self, h_dim=1024, num_tables=4, num_bits=64):
        self.h_dim = h_dim
        self.input_dim = 960  # GIST-960 的输入维度
        self.D = torch.randn(h_dim)
        self.perm = torch.randperm(h_dim)[:num_bits]
        self.D_diags = torch.randn(num_tables, h_dim)
        self.perms = torch.stack([torch.randperm(h_dim)[:num_bits] for _ in range(num_tables)])


class MockSPU:
    """模拟 SPU 执行环境"""
    def __call__(self, func):
        """模拟 SPU 计算"""
        return func


class TestPerformance:
    """性能测试 - GIST-960 数据集 - 多规模测试"""
    
    @pytest.fixture(scope="class", params=[
        ("small", 100, 20),
        ("medium", 1000, 100),
        ("large", 10000, 1000),
    ])
    def gist960_data(self, request):
        """加载不同规模的 GIST-960 测试数据"""
        scale_name, train_limit, test_limit = request.param
        print(f"\n{'='*80}")
        print(f"[SCALE: {scale_name.upper()}] train={train_limit}, test={test_limit}")
        print(f"{'='*80}")
        
        loader = GISTDataLoader()
        db, qs = loader.load_data(device='cpu', train_limit=train_limit, test_limit=test_limit)
        print(f"[INFO] Database: {db.shape}, Queries: {qs.shape}")
        
        return {
            'scale': scale_name,
            'db': db.cpu().numpy(),
            'qs': qs.cpu().numpy(),
            'train_limit': train_limit,
            'test_limit': test_limit
        }
    
    @pytest.fixture(scope="class")
    def retriever_configs(self):
        """检索器配置"""
        return [
            ("SecretFastHadamardRetriever", SecretFastHadamardRetriever),
            ("SecretOptimizedFastHadamardRetriever", SecretOptimizedFastHadamardRetriever),
            ("SecretFastHadamardRetriever_PublicPerm", SecretFastHadamardRetriever_PublicPerm),
            ("SecretOptimizedFastHadamardRetriever_PublicPerm", SecretOptimizedFastHadamardRetriever_PublicPerm),
        ]
    
    def test_build_performance(self, gist960_data, retriever_configs):
        """测试构建阶段性能"""
        scale = gist960_data['scale']
        db, qs = gist960_data['db'], gist960_data['qs']
        
        print("\n" + "="*80)
        print(f"BUILD PHASE PERFORMANCE TEST [{scale.upper()}]")
        print("="*80)
        
        for name, RetrieverClass in retriever_configs:
            plain_model = MockPlainModel(h_dim=1024, num_tables=4, num_bits=64)
            spu = MockSPU()
            alice = Mock()
            bob = Mock()
            
            retriever = RetrieverClass(spu, plain_model, alice, bob)
            
            # 测试构建时间
            t_start = time.time()
            t_build = retriever.build_secret()
            t_total = time.time() - t_start
            
            print(f"\n[{name}]")
            print(f"  Build time (reported): {t_build:.4f}s")
            print(f"  Build time (measured): {t_total:.4f}s")
            
            assert retriever.secret_params is not None
    
    def test_query_performance_single(self, gist960_data, retriever_configs):
        """测试单条查询性能"""
        scale = gist960_data['scale']
        db, qs = gist960_data['db'], gist960_data['qs']
        
        print("\n" + "="*80)
        print(f"SINGLE QUERY PERFORMANCE TEST [{scale.upper()}]")
        print("="*80)
        
        for name, RetrieverClass in retriever_configs:
            plain_model = MockPlainModel(h_dim=1024, num_tables=4, num_bits=64)
            spu = MockSPU()
            alice = Mock()
            bob = Mock()
            
            retriever = RetrieverClass(spu, plain_model, alice, bob)
            retriever.build_secret()
            
            # 测试单条查询
            single_query = qs[:1]
            fp, t_query = retriever.query_secret(single_query)
            
            print(f"\n[{name}]")
            print(f"  Single query time: {t_query:.4f}s")
            print(f"  Output shape: {fp.shape if hasattr(fp, 'shape') else 'N/A'}")
    
    def test_query_performance_batch(self, gist960_data, retriever_configs):
        """测试批量查询性能"""
        scale = gist960_data['scale']
        db, qs = gist960_data['db'], gist960_data['qs']
        
        print("\n" + "="*80)
        print(f"BATCH QUERY PERFORMANCE TEST [{scale.upper()}]")
        print("="*80)
        
        batch_sizes = [10, 50, 100]
        
        for name, RetrieverClass in retriever_configs:
            plain_model = MockPlainModel(h_dim=1024, num_tables=4, num_bits=64)
            spu = MockSPU()
            alice = Mock()
            bob = Mock()
            
            retriever = RetrieverClass(spu, plain_model, alice, bob)
            retriever.build_secret()
            
            print(f"\n[{name}]")
            
            for batch_size in batch_sizes:
                if batch_size > len(qs):
                    continue
                
                batch_query = qs[:batch_size]
                fp, t_query = retriever.query_secret(batch_query)
                
                avg_time = t_query / batch_size
                print(f"  Batch size {batch_size:3d}: {t_query:.4f}s total, {avg_time:.6f}s per query")
    
    def test_end_to_end_performance(self, gist960_data, retriever_configs):
        """端到端性能测试"""
        scale = gist960_data['scale']
        db, qs = gist960_data['db'], gist960_data['qs']
        train_limit = gist960_data['train_limit']
        test_limit = gist960_data['test_limit']
        
        print("\n" + "="*80)
        print(f"END-TO-END PERFORMANCE TEST [{scale.upper()}]")
        print("="*80)
        print(f"Database size: {len(db)}, Query size: {len(qs)}")
        print(f"Config: train_limit={train_limit}, test_limit={test_limit}")
        
        results = []
        
        for name, RetrieverClass in retriever_configs:
            plain_model = MockPlainModel(h_dim=1024, num_tables=4, num_bits=64)
            spu = MockSPU()
            alice = Mock()
            bob = Mock()
            
            retriever = RetrieverClass(spu, plain_model, alice, bob)
            
            # 构建阶段
            t_build_start = time.time()
            retriever.build_secret()
            t_build = time.time() - t_build_start
            
            # 查询阶段
            fp, t_query = retriever.query_secret(qs)
            
            # 总时间
            t_total = t_build + t_query
            avg_query_time = t_query / len(qs)
            
            results.append({
                'name': name,
                'build_time': t_build,
                'query_time': t_query,
                'total_time': t_total,
                'avg_query_time': avg_query_time
            })
            
            print(f"\n[{name}]")
            print(f"  Build time:         {t_build:.4f}s")
            print(f"  Query time (total): {t_query:.4f}s")
            print(f"  Avg query time:     {avg_query_time:.6f}s")
            print(f"  Total time:         {t_total:.4f}s")
        
        # 性能对比
        print("\n" + "="*80)
        print(f"PERFORMANCE COMPARISON [{scale.upper()}]")
        print("="*80)
        
        # 找最快的模型
        fastest_build = min(results, key=lambda x: x['build_time'])
        fastest_query = min(results, key=lambda x: x['avg_query_time'])
        fastest_total = min(results, key=lambda x: x['total_time'])
        
        print(f"\nFastest Build:      {fastest_build['name']} ({fastest_build['build_time']:.4f}s)")
        print(f"Fastest Query:      {fastest_query['name']} ({fastest_query['avg_query_time']:.6f}s)")
        print(f"Fastest Total:      {fastest_total['name']} ({fastest_total['total_time']:.4f}s)")
        print(f"\nTest Scale: {scale.upper()} (train={train_limit}, test={test_limit})")


if __name__ == "__main__":
    # 运行性能测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])
