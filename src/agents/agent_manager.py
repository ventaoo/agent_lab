from .base_agent import BaseAgent
from ..utils.config import Config
from typing import List
import numpy as np

class AgentManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_agents = config.get('experiment.num_agents', 10)
        self.agents: List[BaseAgent] = []
        
        # Create agents
        for i in range(self.num_agents):
            agent = BaseAgent(agent_id=i, config=config)
            self.agents.append(agent)
    
    def get_agent(self, agent_id: int) -> BaseAgent:
        return self.agents[agent_id]
    
    def get_all_agents(self) -> List[BaseAgent]:
        return self.agents