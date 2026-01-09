import json
from typing import Dict, Any, List
from src.memory.vector_memory import VectorMemory
from src.llm.llm_client import LLMClient

class WorkerAgent:
    def __init__(self, worker_id: int, initial_prompt: str, config_path: str = "config/config.yaml"):
        self.worker_id = worker_id
        self.system_prompt = initial_prompt
        self.id = f"worker_{worker_id}"
        
        self.llm_client = LLMClient(config_path)
        self.memory = VectorMemory(config_path)
        
        # Store execution history
        self.execution_history = []
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task and return the result"""
        task_prompt = task['prompt']
        
        try:
            response = self.llm_client.generate(task_prompt, self.system_prompt)
            
            result = {
                'worker_id': self.id,
                'task_id': task.get('id', 'unknown'),
                'task_type': task.get('type', 'general'),
                'input': task_prompt,
                'response': response,
                'status': 'success',
                'timestamp': task.get('timestamp', 'unknown')
            }
        except Exception as e:
            result = {
                'worker_id': self.id,
                'task_id': task.get('id', 'unknown'),
                'task_type': task.get('type', 'general'),
                'input': task_prompt,
                'response': f"Error: {str(e)}",
                'status': 'error',
                'timestamp': task.get('timestamp', 'unknown')
            }
        
        # Add to execution history
        self.execution_history.append(result)
        
        # Store in memory
        memory_entry = {
            'task': task_prompt,
            'response': result['response'],
            'status': result['status'],
            'timestamp': result['timestamp']
        }
        self.memory.add_entry(
            f"Task: {task_prompt}\nResponse: {result['response']}",
            metadata=memory_entry
        )
        
        return result
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt based on manager's optimization"""
        self.system_prompt = new_prompt
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-n:] if len(self.execution_history) >= n else self.execution_history
    
    def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search in memory for relevant entries"""
        return self.memory.search(query, k)