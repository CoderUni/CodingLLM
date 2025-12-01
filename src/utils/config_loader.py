import os
from dotenv import load_dotenv
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"

# Load .env
def load_env():
    env_file = CONFIG_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print(f"No .env file found at: {env_file}. Make sure to copy the example.env file.")

# Load YAML
def load_yaml(name="config.yaml"):
    yaml_path = CONFIG_DIR / name
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Initialize on import
load_env()
config = load_yaml()
