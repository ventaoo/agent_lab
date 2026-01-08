import numpy as np
import pandas as pd
from typing import Dict, List
from ..agents.agent_manager import AgentManager
from ..metrics.drift_metrics import DriftMetrics
from ..core.prompt_updater import PromptUpdater
from ..utils.config import Config
import json
from tqdm import tqdm
from datetime import datetime
import random

class MultiAgentExperiment:
    def __init__(self, config: Config):
        self.config = config
        self.agent_manager = AgentManager(config)
        self.drift_metrics = DriftMetrics()
        self.prompt_updater = PromptUpdater(drift_threshold=0.1)
        self.episode_data = []
        # 任务类型池
        self.task_pool = [
            ("Translate the following sentence to French: 'Hello, how are you?'", "Bonjour, comment allez-vous?"),
            ("Solve this math problem: What is 25 * 17?", "425"),
            ("Summarize the plot of Romeo and Juliet in one sentence", "Two young lovers from feuding families fall in love but die tragically"),
            ("What is the capital of Japan?", "Tokyo"),
            ("Explain what photosynthesis is", "The process by which plants convert light energy into chemical energy"),
            ("What year did World War II end?", "1945"),
            ("Convert 100 Fahrenheit to Celsius", "37.78 Celsius"),
            ("Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
            ("What is the largest planet in our solar system?", "Jupiter"),
            ("Explain the theory of relativity in simple terms", "Einstein's theory about space, time, and gravity")
        ]
        
    def run_episode(self, episode_id: int, tasks: List[tuple]) -> Dict:
        """Run a single episode with all agents"""
        episode_results = {
            'episode_id': episode_id,
            'agent_results': [],
            'aggregate_metrics': {},
            'drift_metrics': {}
        }
        
        agent_responses = []
        
        for agent_id, (task, expected_answer) in enumerate(tasks):
            agent = self.agent_manager.get_agent(agent_id % len(self.agent_manager.get_all_agents()))
            
            # Get context from memory
            context = ""
            recent_interactions = agent.memory.get_recent_interactions(3)
            if recent_interactions:
                context = " ".join([f"Q: {item['query']} A: {item['response']}" for item in recent_interactions])
            
            # Generate response
            response = agent.generate_response(task, context)
            
            # Update memory
            agent.update_memory(task, response)
            agent_responses.append(response)
            
            # Evaluate task success
            success = self._evaluate_task_success(task, response, expected_answer, agent)
            
            # Calculate metrics for this agent
            agent_metrics = {
                'agent_id': agent_id,
                'task': task,
                'expected_answer': expected_answer,
                'response': response,
                'response_length': len(response),
                'success': success,
                'embedding': agent.get_embedding(response).tolist()  # Store embedding for drift analysis
            }
            
            episode_results['agent_results'].append(agent_metrics)
        
        # Calculate drift metrics for this episode
        drift_metrics = self._calculate_drift_metrics(agent_responses)
        episode_results['drift_metrics'] = drift_metrics
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(episode_results)
        
        return episode_results
    
    def _evaluate_task_success(self, task: str, response: str, expected_answer: str, agent) -> bool:
        """Evaluate task success using LLM-based comparison"""
        # Simple string similarity check first
        response_lower = response.lower().strip()
        expected_lower = expected_answer.lower().strip()
        
        # If exact match, return True
        if expected_lower in response_lower or response_lower in expected_lower:
            return True
        
        # For more complex evaluation, use LLM to judge
        evaluation_prompt = f"""
        Task: {task}
        Expected Answer: {expected_answer}
        Agent Response: {response}
        
        Evaluate if the agent response correctly answers the task. 
        Return ONLY 'SUCCESS' if the answer is correct, or 'FAILURE' if it's incorrect.
        """
        
        try:
            # Use the agent itself to evaluate (or you could use a separate evaluator)
            eval_response = agent.generate_response(evaluation_prompt, "")
            return "SUCCESS" in eval_response.upper()
        except:
            # Fallback to simple keyword matching
            return any(keyword.lower() in response_lower for keyword in expected_lower.split()[:3])
    
    def _calculate_drift_metrics(self, agent_responses: List[str]) -> Dict:
        """Calculate various drift metrics"""
        if len(agent_responses) < 2:
            return {
                'kl_divergence': 0.0,
                'embedding_shift': 0.0,
                'response_similarity': 1.0
            }
        
        # Get embeddings for all responses
        embeddings = []
        for response in agent_responses:
            # Use the agent's own embedding model
            agent = self.agent_manager.get_agent(0)  # Use first agent for embedding
            emb = agent.get_embedding(response)
            embeddings.append(emb)
        
        # Calculate KL divergence using response length distributions
        response_lengths = [len(resp) for resp in agent_responses]
        if len(set(response_lengths)) > 1:
            # Normalize to create probability distributions
            total_length = sum(response_lengths)
            if total_length > 0:
                length_dist = np.array(response_lengths) / total_length
                uniform_dist = np.ones(len(response_lengths)) / len(response_lengths)
                kl_div = self.drift_metrics.calculate_kl_divergence(length_dist, uniform_dist)
            else:
                kl_div = 0.0
        else:
            kl_div = 0.0
        
        # Calculate embedding shift
        embedding_shift = self.drift_metrics.calculate_embedding_shift(embeddings)
        
        # Calculate response similarity
        similarity = self.drift_metrics.calculate_response_similarity(embeddings)
        
        return {
            'kl_divergence': kl_div,
            'embedding_shift': embedding_shift,
            'response_similarity': similarity
        }
    
    def _calculate_aggregate_metrics(self, episode_results: Dict):
        """Calculate aggregate metrics for the episode"""
        agent_results = episode_results['agent_results']
        
        # Task success rate
        success_rate = sum(1 for result in agent_results if result['success']) / len(agent_results)
        
        # Average response length
        avg_response_length = np.mean([result['response_length'] for result in agent_results])
        
        # Get drift metrics
        drift_metrics = episode_results['drift_metrics']
        
        episode_results['aggregate_metrics'] = {
            'success_rate': success_rate,
            'avg_response_length': avg_response_length,
            'total_agents': len(agent_results),
            'kl_divergence': drift_metrics['kl_divergence'],
            'embedding_shift': drift_metrics['embedding_shift'],
            'response_similarity': drift_metrics['response_similarity']
        }
    
    def run_experiment(self, episodes: int = 1000):
        """Run the complete experiment with tqdm progress bar"""
        print(f"Starting experiment with {episodes} episodes...")
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(
            range(episodes),
            desc="Running Episodes",
            unit="episode",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        success_rates = []  # 记录每个episode的成功率
        
        for episode in progress_bar:
            # Generate tasks for this episode from task pool
            tasks = []
            for i in range(self.config.get('experiment.num_agents')):
                task, expected = random.choice(self.task_pool)
                tasks.append((task, expected))
            
            episode_result = self.run_episode(episode, tasks)
            self.episode_data.append(episode_result)
            
            # 获取当前episode的成功率
            current_success_rate = episode_result['aggregate_metrics']['success_rate']
            success_rates.append(current_success_rate)
            
            # 计算移动平均成功率（最近10个episode）
            if len(success_rates) >= 10:
                recent_avg = sum(success_rates[-10:]) / 10
            else:
                recent_avg = sum(success_rates) / len(success_rates) if success_rates else 0
            
            # 更新进度条的后缀信息
            progress_bar.set_postfix({
                'Curr': f'{current_success_rate:.3f}',
                'Avg': f'{recent_avg:.3f}'
            })
        
        print("Experiment completed!")
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyze and return experiment results"""
        # Convert to DataFrame for analysis
        metrics_data = []
        drift_data = []
        
        for episode in self.episode_data:
            metrics = episode['aggregate_metrics'].copy()
            metrics['episode_id'] = episode['episode_id']
            metrics_data.append(metrics)
            
            # Also collect drift-specific data
            drift_entry = {
                'episode_id': episode['episode_id'],
                'kl_divergence': episode['drift_metrics']['kl_divergence'],
                'embedding_shift': episode['drift_metrics']['embedding_shift'],
                'response_similarity': episode['drift_metrics']['response_similarity'],
                'success_rate': episode['aggregate_metrics']['success_rate']
            }
            drift_data.append(drift_entry)
        
        metrics_df = pd.DataFrame(metrics_data)
        drift_df = pd.DataFrame(drift_data)
        
        # Calculate overall statistics
        overall_stats = {
            'total_episodes': len(metrics_df),
            'avg_success_rate': metrics_df['success_rate'].mean(),
            'std_success_rate': metrics_df['success_rate'].std(),
            'avg_response_length': metrics_df['avg_response_length'].mean(),
            'avg_kl_divergence': metrics_df['kl_divergence'].mean(),
            'avg_embedding_shift': metrics_df['embedding_shift'].mean(),
            'avg_response_similarity': metrics_df['response_similarity'].mean(),
            'success_rate_trend': metrics_df['success_rate'].tolist(),
            'kl_divergence_trend': metrics_df['kl_divergence'].tolist(),
            'embedding_shift_trend': metrics_df['embedding_shift'].tolist()
        }
        
        return {
            'metrics_over_time': metrics_df.to_dict('records'),
            'drift_analysis': drift_df.to_dict('records'),
            'overall_stats': overall_stats,
            'raw_data': self.episode_data
        }