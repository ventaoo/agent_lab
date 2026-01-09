import json
import yaml
from typing import Dict, Any, List
from src.llm.llm_client import LLMClient
from src.metrics.drift_metrics import DriftMetrics

class ManagerAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.llm_client = LLMClient(config_path)
        self.metrics_calculator = DriftMetrics(config_path)
        self.config_path = config_path
        
        # Store intervention history
        self.intervention_history = []
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)  # Changed from json.load to yaml.safe_load
    
    def evaluate_workers(self, workers: List['WorkerAgent'], 
                        episode: int, drift_threshold: float = 0.15) -> Dict[str, Any]:
        """Evaluate all workers and detect drift"""
        evaluation_results = {
            'episode': episode,
            'workers_evaluated': [],
            'drift_detected': [],
            'interventions_needed': []
        }
        
        for worker in workers:
            # Get recent history from worker
            recent_history = worker.get_recent_history(n=20)  # Last 20 tasks
            
            # Calculate metrics
            if recent_history:
                # Get baseline responses (first few responses when worker was initialized)
                baseline_responses = self._get_baseline_responses(worker)
                
                metrics = self.metrics_calculator.get_metrics_summary(
                    worker.id, recent_history, baseline_responses
                )
                
                drift_detected = self.metrics_calculator.detect_drift(
                    worker.id, recent_history, baseline_responses, drift_threshold
                )
                
                worker_result = {
                    'worker_id': worker.id,
                    'metrics': metrics,
                    'drift_detected': drift_detected,
                    'recent_success_rate': metrics['success_rate']
                }
                
                evaluation_results['workers_evaluated'].append(worker_result)
                
                if drift_detected:
                    evaluation_results['drift_detected'].append(worker_result)
                    evaluation_results['interventions_needed'].append(worker.id)
        
        return evaluation_results
    
    def _get_baseline_responses(self, worker: 'WorkerAgent') -> List[str]:
        """Get baseline responses for comparison"""
        # In a real implementation, this would be the worker's initial responses
        # For now, we'll use a default baseline
        return ["Default response for comparison", "Another baseline response"]
    
    def generate_optimized_prompt(self, worker_id: str, 
                                current_prompt: str, 
                                drift_metrics: Dict[str, float],
                                recent_history: List[Dict[str, Any]]) -> str:
        """Generate an optimized prompt based on drift analysis"""
        
        # Prepare analysis context
        analysis_context = {
            'worker_id': worker_id,
            'current_prompt': current_prompt,
            'drift_metrics': drift_metrics,
            'recent_history_sample': recent_history[-5:] if recent_history else []
        }
        
        # Construct meta-prompt for prompt optimization
        meta_prompt = f"""
You are an expert prompt optimizer for language model agents. Your task is to analyze the current performance of a worker agent and generate an improved system prompt that addresses identified issues.

Current worker analysis:
- Worker ID: {analysis_context['worker_id']}
- Current system prompt: {analysis_context['current_prompt']}
- Performance metrics: {json.dumps(analysis_context['drift_metrics'], indent=2)}
- Recent task examples: {json.dumps(analysis_context['recent_history_sample'], indent=2)}

Based on this analysis, please generate an optimized system prompt that:
1. Maintains the original purpose and constraints of the agent
2. Addresses specific performance issues identified in the metrics
3. Improves consistency and reliability
4. Is clear and actionable

The new prompt should be formatted as JSON with a single key "optimized_prompt" containing the new system prompt text.
"""
        
        try:
            response = self.llm_client.generate(meta_prompt)
            
            # Parse the response to extract the optimized prompt
            try:
                parsed_response = json.loads(response)
                optimized_prompt = parsed_response.get('optimized_prompt', current_prompt)
            except json.JSONDecodeError:
                # If parsing fails, try to extract the prompt using string operations
                if '"optimized_prompt":' in response:
                    start = response.find('"optimized_prompt":') + len('"optimized_prompt":')
                    end = response.find('}', start)
                    prompt_text = response[start:end].strip().strip('"')
                    optimized_prompt = prompt_text
                else:
                    optimized_prompt = current_prompt  # Fallback to current prompt
            
            # Log the intervention
            intervention_record = {
                'worker_id': worker_id,
                'episode': 'unknown',  # Will be set when called from experiment runner
                'old_prompt': current_prompt,
                'new_prompt': optimized_prompt,
                'drift_metrics': drift_metrics,
                'timestamp': 'unknown'
            }
            self.intervention_history.append(intervention_record)
            
            return optimized_prompt
            
        except Exception as e:
            print(f"Error generating optimized prompt for {worker_id}: {e}")
            return current_prompt  # Return original prompt if optimization fails
    
    def apply_interventions(self, workers: List['WorkerAgent'], 
                           evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply prompt optimizations to workers that need intervention"""
        interventions_applied = []
        
        for worker_result in evaluation_results.get('drift_detected', []):
            worker_id = worker_result['worker_id']
            drift_metrics = worker_result['metrics']
            
            # Find the actual worker object
            target_worker = None
            for worker in workers:
                if worker.id == worker_id:
                    target_worker = worker
                    break
            
            if target_worker:
                # Get recent history for analysis
                recent_history = target_worker.get_recent_history(n=10)
                
                # Generate optimized prompt
                old_prompt = target_worker.system_prompt
                new_prompt = self.generate_optimized_prompt(
                    worker_id, old_prompt, drift_metrics, recent_history
                )
                
                # Apply the new prompt
                target_worker.update_system_prompt(new_prompt)
                
                intervention_record = {
                    'worker_id': worker_id,
                    'old_prompt': old_prompt,
                    'new_prompt': new_prompt,
                    'drift_metrics': drift_metrics,
                    'timestamp': 'unknown'
                }
                interventions_applied.append(intervention_record)
        
        return interventions_applied