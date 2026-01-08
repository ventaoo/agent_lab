import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Optional
import seaborn as sns

class ExperimentVisualizer:
    def __init__(self, metrics_csv_path: str = 'metrics_over_time.csv', 
                 drift_csv_path: str = 'drift_analysis.csv'):
        self.metrics_df = pd.read_csv(metrics_csv_path)
        self.drift_df = pd.read_csv(drift_csv_path)
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_all_metrics(self, save_path: Optional[str] = None):
        """绘制所有主要指标的变化曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Agent Experiment Metrics Over Time', fontsize=16)
        
        # 任务成功率
        axes[0, 0].plot(self.metrics_df['episode_id'], self.metrics_df['success_rate'], 
                       color='blue', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Success Rate Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL散度
        axes[0, 1].plot(self.metrics_df['episode_id'], self.metrics_df['kl_divergence'], 
                       color='red', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('KL Divergence Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 嵌入偏移
        axes[1, 0].plot(self.metrics_df['episode_id'], self.metrics_df['embedding_shift'], 
                       color='green', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Embedding Shift Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Embedding Shift')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 响应相似度
        axes[1, 1].plot(self.metrics_df['episode_id'], self.metrics_df['response_similarity'], 
                       color='orange', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('Response Similarity Over Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Response Similarity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_success_rate_with_trend(self, window_size: int = 50, save_path: Optional[str] = None):
        """绘制任务成功率及其移动平均趋势"""
        plt.figure(figsize=(12, 6))
        
        # 原始成功率
        plt.plot(self.metrics_df['episode_id'], self.metrics_df['success_rate'], 
                color='lightblue', alpha=0.5, label='Raw Success Rate', linewidth=0.8)
        
        # 移动平均
        if len(self.metrics_df) >= window_size:
            rolling_mean = self.metrics_df['success_rate'].rolling(window=window_size, center=True).mean()
            plt.plot(self.metrics_df['episode_id'], rolling_mean, 
                    color='blue', linewidth=2, label=f'{window_size}-episode Moving Average')
        
        plt.title('Task Success Rate with Trend')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_drift_metrics_comparison(self, save_path: Optional[str] = None):
        """比较漂移指标"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # KL散度
        axes[0].plot(self.drift_df['episode_id'], self.drift_df['kl_divergence'], 
                    color='red', alpha=0.7)
        axes[0].set_title('KL Divergence')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('KL Divergence')
        axes[0].grid(True, alpha=0.3)
        
        # 嵌入偏移
        axes[1].plot(self.drift_df['episode_id'], self.drift_df['embedding_shift'], 
                    color='green', alpha=0.7)
        axes[1].set_title('Embedding Shift')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Embedding Shift')
        axes[1].grid(True, alpha=0.3)
        
        # 响应相似度
        axes[2].plot(self.drift_df['episode_id'], self.drift_df['response_similarity'], 
                    color='orange', alpha=0.7)
        axes[2].set_title('Response Similarity')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Response Similarity')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """绘制指标相关性热图"""
        # 选择要分析的列
        correlation_cols = ['success_rate', 'kl_divergence', 'embedding_shift', 'response_similarity']
        correlation_data = self.metrics_df[correlation_cols]
        
        plt.figure(figsize=(8, 6))
        correlation_matrix = correlation_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Correlation Heatmap of Metrics')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_drift_vs_performance(self, save_path: Optional[str] = None):
        """绘制漂移指标与性能指标的关系"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # KL散度 vs 成功率
        axes[0].scatter(self.metrics_df['kl_divergence'], self.metrics_df['success_rate'], 
                       alpha=0.6, color='red')
        axes[0].set_xlabel('KL Divergence')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('KL Divergence vs Success Rate')
        axes[0].grid(True, alpha=0.3)
        
        # 嵌入偏移 vs 成功率
        axes[1].scatter(self.metrics_df['embedding_shift'], self.metrics_df['success_rate'], 
                       alpha=0.6, color='green')
        axes[1].set_xlabel('Embedding Shift')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Embedding Shift vs Success Rate')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_episode_distribution(self, save_path: Optional[str] = None):
        """绘制各项指标的分布直方图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = [
            ('success_rate', 'Success Rate', 'blue'),
            ('kl_divergence', 'KL Divergence', 'red'),
            ('embedding_shift', 'Embedding Shift', 'green'),
            ('response_similarity', 'Response Similarity', 'orange')
        ]
        
        for idx, (col, title, color) in enumerate(metrics_to_plot):
            row, col_idx = idx // 2, idx % 2
            
            axes[row, col_idx].hist(self.metrics_df[col], bins=30, color=color, alpha=0.7, edgecolor='black')
            axes[row, col_idx].set_title(f'Distribution of {title}')
            axes[row, col_idx].set_xlabel(title)
            axes[row, col_idx].set_ylabel('Frequency')
            axes[row, col_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def generate_summary_statistics(self) -> dict:
        """生成汇总统计信息"""
        stats = {
            'total_episodes': len(self.metrics_df),
            'success_rate': {
                'mean': self.metrics_df['success_rate'].mean(),
                'std': self.metrics_df['success_rate'].std(),
                'min': self.metrics_df['success_rate'].min(),
                'max': self.metrics_df['success_rate'].max()
            },
            'kl_divergence': {
                'mean': self.metrics_df['kl_divergence'].mean(),
                'std': self.metrics_df['kl_divergence'].std(),
                'min': self.metrics_df['kl_divergence'].min(),
                'max': self.metrics_df['kl_divergence'].max()
            },
            'embedding_shift': {
                'mean': self.metrics_df['embedding_shift'].mean(),
                'std': self.metrics_df['embedding_shift'].std(),
                'min': self.metrics_df['embedding_shift'].min(),
                'max': self.metrics_df['embedding_shift'].max()
            },
            'response_similarity': {
                'mean': self.metrics_df['response_similarity'].mean(),
                'std': self.metrics_df['response_similarity'].std(),
                'min': self.metrics_df['response_similarity'].min(),
                'max': self.metrics_df['response_similarity'].max()
            }
        }
        return stats
    
    def print_summary(self):
        """打印汇总信息"""
        stats = self.generate_summary_statistics()
        
        print("="*50)
        print("EXPERIMENT SUMMARY STATISTICS")
        print("="*50)
        print(f"Total Episodes: {stats['total_episodes']}")
        print()
        
        for metric, values in stats.items():
            if metric != 'total_episodes':
                print(f"{metric.upper()}:")
                print(f"  Mean: {values['mean']:.4f}")
                print(f"  Std:  {values['std']:.4f}")
                print(f"  Min:  {values['min']:.4f}")
                print(f"  Max:  {values['max']:.4f}")
                print()