import torch

# Import the model class from the main file
from src.Classifier import Classifier

import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")

# Data parameters testing
num_classes = 10
input_shape = 784

def build_model_and_log(config, model, model_name="MLP", model_description="Simple MLP"):
    with wandb.init(project="MLOps-Pycon2023", name=f"initialize Model ExecId-{args.IdExecution}", job_type="initialize-model", config=config) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"initialized_model_{model_name}.pth"

        torch.save(model.state_dict(), f"./model/{name_artifact_model}")
        # ➕ another way to add a file to an Artifact
        model_artifact.add_file(f"./model/{name_artifact_model}")

        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)


# MLP
# Testing config
model_config = {"input_shape":input_shape,
                "hidden_layer_1": 32,
                "hidden_layer_2": 64,
                "num_classes":num_classes}

model = Classifier(**model_config)

build_model_and_log(model_config, model, "linear","Simple Linear Classifier")