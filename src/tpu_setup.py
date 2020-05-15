import json
import os

with open("tpu_vars.json", "r") as f:
    env_var = json.load(f)
for k, v in env_var.items():
    os.environ[k] = v
