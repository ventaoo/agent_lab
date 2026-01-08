import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
        return value

    def show(self,):
        for key, value in self.config.items():
            print(f"{key}:")
            for v in value.items():
                print(f"  {v[0]}: {v[1]}")


if __name__ == "__main__":
    # Function test
    config = Config()
    config.show()