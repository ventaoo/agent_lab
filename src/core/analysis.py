import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class PostExperimentAnalysis:
    def __init__(self, metrics_df: pd.DataFrame, drift_df: pd.DataFrame):
        self.metrics_df = metrics_df
        self.drift_df = drift_df
        self.episode_count = len(metrics_df)
    
    def analyze_convergence(self) -> Dict:
        """分析各项指标的收敛性"""
        convergence_results = {}
        
        metrics_to_check = ['success_rate', 'kl_divergence', 'embedding_shift', 'response_similarity']
        
        for metric in metrics_to_check:
            values = self.metrics_df[metric].values
            
            # 计算后半部分与前半部分的差异
            mid_point = len(values) // 2
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            # 检查均值差异
            mean_diff = abs(np.mean(second_half) - np.mean(first_half))
            std_diff = abs(np.std(second_half) - np.std(first_half))
            
            # 使用t检验检查显著性
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            
            convergence_results[metric] = {
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'converged': p_value > 0.05,  # p > 0.05 表示无显著差异，即收敛
                'final_value': values[-1],
                'initial_value': values[0]
            }
        
        return convergence_results
    
    def analyze_stability(self) -> Dict:
        """分析系统稳定性"""
        stability_results = {}
        
        # 计算滚动标准差来评估稳定性
        window_size = min(100, len(self.metrics_df) // 10)  # 使用10%的数据作为窗口
        
        for metric in ['success_rate', 'kl_divergence', 'embedding_shift', 'response_similarity']:
            values = self.metrics_df[metric].values
            
            # 计算滚动标准差
            rolling_std = pd.Series(values).rolling(window=window_size, center=True).std().dropna()
            
            stability_results[metric] = {
                'rolling_std_mean': float(np.mean(rolling_std)),
                'rolling_std_std': float(np.std(rolling_std)),
                'max_rolling_std': float(np.max(rolling_std)),
                'min_rolling_std': float(np.min(rolling_std)),
                'stability_score': 1.0 - (np.mean(rolling_std) / (np.std(values) + 1e-8))  # 0-1之间的稳定性分数
            }
        
        return stability_results
    
    def analyze_drift_patterns(self) -> Dict:
        """分析漂移模式"""
        drift_patterns = {}
        
        # 检查漂移指标的趋势
        for metric in ['kl_divergence', 'embedding_shift', 'response_similarity']:
            values = self.metrics_df[metric].values
            
            # 计算趋势线
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            drift_patterns[metric] = {
                'trend_slope': slope,
                'trend_r_squared': r_value ** 2,
                'trend_significance': p_value,
                'increasing': slope > 0,
                'trend_strength': abs(r_value)
            }
        
        return drift_patterns
    
    def detect_performance_degradation(self) -> Dict:
        """检测性能退化"""
        performance_metrics = ['success_rate', 'response_similarity']
        degradation_results = {}
        
        for metric in performance_metrics:
            values = self.metrics_df[metric].values
            
            # 将数据分为前半部分和后半部分
            mid_point = len(values) // 2
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            # 计算性能退化指标
            degradation_score = (np.mean(first_half) - np.mean(second_half)) / (np.mean(first_half) + 1e-8)
            
            degradation_results[metric] = {
                'degradation_score': degradation_score,
                'first_half_mean': float(np.mean(first_half)),
                'second_half_mean': float(np.mean(second_half)),
                'degraded': degradation_score > 0.05  # 如果退化超过5%，标记为退化
            }
        
        return degradation_results
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合分析报告"""
        convergence = self.analyze_convergence()
        stability = self.analyze_stability()
        drift_patterns = self.analyze_drift_patterns()
        degradation = self.detect_performance_degradation()
        
        overall_assessment = {
            'total_episodes': self.episode_count,
            'convergence_analysis': convergence,
            'stability_analysis': stability,
            'drift_patterns': drift_patterns,
            'degradation_analysis': degradation,
            'summary': self._generate_summary(convergence, stability, degradation)
        }
        
        return overall_assessment
    
    def _generate_summary(self, convergence, stability, degradation) -> str:
        """生成摘要文本"""
        summary_parts = []
        
        # 收敛性摘要
        converged_metrics = [m for m, v in convergence.items() if v['converged']]
        summary_parts.append(f"Converged metrics: {len(converged_metrics)}/{len(convergence)}")
        
        # 稳定性摘要
        stable_metrics = [m for m, v in stability.items() if v['stability_score'] > 0.7]
        summary_parts.append(f"Stable metrics: {len(stable_metrics)}/{len(stability)}")
        
        # 退化摘要
        degraded_metrics = [m for m, v in degradation.items() if v['degraded']]
        summary_parts.append(f"Degraded metrics: {len(degraded_metrics)}/{len(degradation)}")
        
        return "; ".join(summary_parts)
    
    def plot_long_term_trends(self, save_path: str = None):
        """绘制长期趋势图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Long-term Trends Analysis (1000 Episodes)', fontsize=16)
        
        # 成功率趋势
        axes[0, 0].plot(self.metrics_df['episode_id'], self.metrics_df['success_rate'], 
                       color='blue', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Success Rate Trend')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL散度趋势
        axes[0, 1].plot(self.metrics_df['episode_id'], self.metrics_df['kl_divergence'], 
                       color='red', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('KL Divergence Trend')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 嵌入偏移趋势
        axes[1, 0].plot(self.metrics_df['episode_id'], self.metrics_df['embedding_shift'], 
                       color='green', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Embedding Shift Trend')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Embedding Shift')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 响应相似度趋势
        axes[1, 1].plot(self.metrics_df['episode_id'], self.metrics_df['response_similarity'], 
                       color='orange', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('Response Similarity Trend')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Response Similarity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()