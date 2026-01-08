import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from ..agents.agent_manager import AgentManager
import random

class ContractTestSuite:
    def __init__(self, agent_manager: AgentManager, metrics_df: pd.DataFrame):
        self.agent_manager = agent_manager
        self.metrics_df = metrics_df
        self.test_results = {}
    
    def run_all_contract_tests(self) -> Dict:
        """运行所有契约测试"""
        test_functions = [
            self.test_memory_integrity,
            self.test_response_consistency,
            self.test_agent_isolation,
            self.test_metric_bounds,
            self.test_system_stability,
            self.test_error_handling
        ]
        
        for test_func in test_functions:
            test_name = test_func.__name__
            self.test_results[test_name] = test_func()
        
        return self.test_results
    
    def test_memory_integrity(self) -> Dict:
        """测试内存完整性"""
        results = {
            'passed': True,
            'details': [],
            'metric': 'memory_integrity'
        }
        
        for agent in self.agent_manager.get_all_agents():
            memory_size = len(agent.memory.memory)
            expected_size = min(agent.memory.memory_size, len(agent.response_history))
            
            if memory_size > agent.memory.memory_size:
                results['passed'] = False
                results['details'].append(f"Agent {agent.agent_id}: Memory exceeds limit ({memory_size} > {agent.memory.memory_size})")
        
        results['score'] = 1.0 if results['passed'] else 0.0
        return results
    
    def test_response_consistency(self) -> Dict:
        """测试响应一致性"""
        results = {
            'passed': True,
            'details': [],
            'metric': 'response_consistency'
        }
        
        # 检查是否有空响应或错误响应
        empty_responses = 0
        error_responses = 0
        
        for _, row in self.metrics_df.iterrows():
            if row['success_rate'] == 0:  # 如果成功率过低
                results['passed'] = False
                results['details'].append(f"Episode {row['episode_id']}: Zero success rate")
        
        # 计算一致性分数
        consistency_score = 1.0 - (len(results['details']) / len(self.metrics_df))
        results['score'] = max(0.0, consistency_score)
        
        return results
    
    def test_agent_isolation(self) -> Dict:
        """测试代理隔离性"""
        results = {
            'passed': True,
            'details': [],
            'metric': 'agent_isolation'
        }
        
        # 检查不同代理的内存是否相互影响（通过嵌入相似度）
        agents = self.agent_manager.get_all_agents()
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j and len(agent1.memory.memory) > 0 and len(agent2.memory.memory) > 0:
                    # 检查最近的交互是否相似（不应该相似）
                    recent1 = agent1.memory.get_recent_interactions(1)
                    recent2 = agent2.memory.get_recent_interactions(1)
                    
                    if recent1 and recent2:
                        # 这里简单检查响应是否完全相同（应该不同）
                        if recent1[0]['response'] == recent2[0]['response']:
                            results['passed'] = False
                            results['details'].append(f"Agent {i} and {j} have identical responses (potential isolation issue)")
        
        results['score'] = 1.0 if results['passed'] else 0.5
        return results
    
    def test_metric_bounds(self) -> Dict:
        """测试指标边界"""
        results = {
            'passed': True,
            'details': [],
            'metric': 'metric_bounds'
        }
        
        # 检查指标是否在合理范围内
        metrics_bounds = {
            'success_rate': (0.0, 1.0),
            'kl_divergence': (0.0, float('inf')),
            'embedding_shift': (0.0, 2.0),  # 余弦距离最大为2
            'response_similarity': (0.0, 1.0)
        }
        
        for metric, (min_val, max_val) in metrics_bounds.items():
            if metric in self.metrics_df.columns:
                out_of_bounds = self.metrics_df[
                    (self.metrics_df[metric] < min_val) | 
                    (self.metrics_df[metric] > max_val)
                ]
                
                if len(out_of_bounds) > 0:
                    results['passed'] = False
                    results['details'].append(f"{metric}: {len(out_of_bounds)} values out of bounds [{min_val}, {max_val}]")
        
        results['score'] = 1.0 if results['passed'] else 0.8
        return results
    
    def test_system_stability(self) -> Dict:
        """测试系统稳定性"""
        results = {
            'passed': True,
            'details': [],
            'metric': 'system_stability'
        }
        
        # 检查指标的波动性
        for metric in ['success_rate', 'kl_divergence', 'embedding_shift', 'response_similarity']:
            if metric in self.metrics_df.columns:
                values = self.metrics_df[metric].values
                std = np.std(values)
                mean = np.mean(values)
                
                # 如果标准差相对于均值过大，认为不稳定
                if mean != 0 and (std / abs(mean)) > 0.5:  # 变异系数 > 0.5
                    results['passed'] = False
                    results['details'].append(f"{metric}: High variability (CV={std/abs(mean):.3f})")
        
        results['score'] = 1.0 if results['passed'] else 0.7
        return results
    
    def test_error_handling(self) -> Dict:
        """测试错误处理"""
        results = {
            'passed': True,
            'details': [],
            'metric': 'error_handling'
        }
        
        # 检查是否有NaN或无穷大值
        for col in self.metrics_df.columns:
            if col in ['episode_id']:  # 跳过episode_id列
                continue
                
            series = self.metrics_df[col]
            
            # 检查NaN值
            if series.isna().any():
                results['passed'] = False
                results['details'].append(f"Column {col}: Contains NaN values")
            
            # 检查无穷大值 - 修复原代码错误
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.isna().any():  # to_numeric可能产生NaN，表示无法转换的值
                non_numeric_count = series.isna().sum() - numeric_series.isna().sum()
                if non_numeric_count > 0:
                    results['passed'] = False
                    results['details'].append(f"Column {col}: Contains non-numeric values")
            
            # 检查无穷大值
            finite_mask = np.isfinite(numeric_series.astype(float))
            if not finite_mask.all():
                results['passed'] = False
                results['details'].append(f"Column {col}: Contains infinite values")
        
        results['score'] = 1.0 if results['passed'] else 0.6
        return results
    
    def generate_test_report(self) -> str:
        """生成测试报告"""
        report_parts = ["CONTRACT TEST SUITE REPORT", "="*50]
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        report_parts.append(f"Total Tests: {total_tests}")
        report_parts.append(f"Passed Tests: {passed_tests}")
        report_parts.append(f"Success Rate: {passed_tests/total_tests:.2%}")
        report_parts.append("")
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            report_parts.append(f"{test_name}: {status} (Score: {result['score']:.2f})")
            
            if result['details']:
                for detail in result['details']:
                    report_parts.append(f"  - {detail}")
            report_parts.append("")
        
        return "\n".join(report_parts)