import torch
import torchvision
from torch.utils.data import TensorDataset

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

def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # Scale images to the [0, 1] range
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Make sure images have shape (1, 28, 28)
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)

def preprocess_and_log(steps):

    with wandb.init(project="MLOps-Pycon2023",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    
        processed_data = wandb.Artifact(
            "mnist-preprocess", type="dataset",
            description="Preprocessed MNIST dataset",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('mnist-raw:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)

# Configure preprocessing steps
steps = {"normalize": True,
         "expand_dims": True}

preprocess_and_log(steps)