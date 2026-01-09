import json
import yaml
import random
import time
from typing import List, Dict, Any
from src.agents.worker_agent import WorkerAgent
from src.agents.manager_agent import ManagerAgent
from src.metrics.drift_metrics import DriftMetrics

class ExperimentRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)  # Changed from json.load to yaml.safe_load
        
        self.num_episodes = self.config['experiment']['num_episodes']
        self.num_workers = self.config['experiment']['num_workers']
        self.drift_threshold = self.config['experiment']['drift_threshold']
        self.success_threshold = self.config['experiment']['success_threshold']
        self.update_frequency = self.config['experiment']['update_frequency']
        
        # Initialize workers with the same initial prompt
        self.initial_prompt = """You are a helpful AI assistant. Follow these guidelines:
1. Provide accurate and helpful responses
2. Be concise but thorough
3. Maintain a professional tone
4. If you're unsure about something, say so rather than making things up"""
        
        self.workers = [
            WorkerAgent(i, self.initial_prompt, config_path) 
            for i in range(self.num_workers)
        ]
        
        self.manager = ManagerAgent(config_path)
        self.metrics_calculator = DriftMetrics(config_path)
        
        # Load task pool
        with open('data/task_pool.json', 'r') as f:
            self.task_pool = json.load(f)
        
        # Store experiment results
        self.results_log = []
        self.intervention_log = []
    
    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single episode of the experiment"""
        print(f"Running episode {episode_num}/{self.num_episodes}")
        
        # Generate tasks for this episode
        num_tasks = random.randint(10, 20)  # Random number of tasks per episode
        tasks = random.sample(self.task_pool, min(num_tasks, len(self.task_pool)))
        
        # Assign tasks to workers randomly
        episode_results = []
        for task in tasks:
            worker = random.choice(self.workers)
            result = worker.execute_task(task)
            episode_results.append(result)
        
        # Evaluate workers for drift
        evaluation_results = self.manager.evaluate_workers(
            self.workers, episode_num, self.drift_threshold
        )
        
        # Apply interventions if needed
        if evaluation_results['interventions_needed']:
            interventions = self.manager.apply_interventions(
                self.workers, evaluation_results
            )
            self.intervention_log.extend(interventions)
            
            print(f"Applied {len(interventions)} interventions in episode {episode_num}")
        
        # Calculate overall metrics for this episode
        all_responses = [r for r in episode_results if r['status'] == 'success']
        baseline_responses = [self.initial_prompt] * len(all_responses)  # Simplified baseline
        
        episode_metrics = self.metrics_calculator.get_metrics_summary(
            'overall', all_responses, baseline_responses
        )
        
        episode_summary = {
            'episode': episode_num,
            'num_tasks': len(tasks),
            'num_workers': self.num_workers,
            'results': episode_results,
            'evaluation': evaluation_results,
            'metrics': episode_metrics,
            'interventions_count': len(evaluation_results['interventions_needed']),
            'timestamp': time.time()
        }
        
        self.results_log.append(episode_summary)
        
        return episode_summary
    
    def run_experiment(self):
        """Run the full experiment for all episodes"""
        print(f"Starting experiment with {self.num_episodes} episodes and {self.num_workers} workers")
        
        for episode in range(self.num_episodes):
            episode_result = self.run_episode(episode)
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Completed {episode + 1} episodes")
        
        print("Experiment completed!")
        self.save_results()
    
    def save_results(self):
        """Save experiment results to file"""
        with open('results/metrics_log.json', 'w') as f:
            json.dump({
                'experiment_config': self.config,
                'results_log': self.results_log,
                'intervention_log': self.intervention_log,
                'final_worker_prompts': [w.system_prompt for w in self.workers]
            }, f, indent=2)
        
        print(f"Results saved to results/metrics_log.json")
        print(f"Total interventions applied: {len(self.intervention_log)}")