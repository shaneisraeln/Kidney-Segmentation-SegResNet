import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration dict to YAML file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
