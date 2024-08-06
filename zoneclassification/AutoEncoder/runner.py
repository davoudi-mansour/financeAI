import torch
from dataset import get_dataloaders
from model import LSTMAutoencoder
from train import Trainer
from test import Tester
from finetune import FineTuner
import numpy as np
import yaml
import sys

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config, device):
    # Hyperparameters
    input_size = config['model_params']['input_size']
    hidden_size = config['model_params']['hidden_size']
    num_layers = config['model_params']['num_layers']
    num_epochs = config['training_params']['num_epochs']
    learning_rate = config['training_params']['learning_rate']
    patience = config['training_params']['patience']
    batch_size = config['training_params']['batch_size']

    # Get dataloaders
    train_dataloader, val_dataloader, _ = get_dataloaders(config['data']['data_path'], batch_size=batch_size)

    # Initialize model
    model = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)

    # Train model
    trainer = Trainer(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, patience)
    trainer.train()

def test_model(config, device):
    # Hyperparameters
    input_size = config['model_params']['input_size']
    hidden_size = config['model_params']['hidden_size']
    num_layers = config['model_params']['num_layers']
    batch_size = config['testing_params']['batch_size']

    # Get dataloaders
    _, val_dataloader, test_dataloader = get_dataloaders(config['data']['data_path'], batch_size=batch_size)

    # Initialize model
    model = get_model(input_size, hidden_size, num_layers).to(device)

    # Load the best model
    model.load_state_dict(torch.load('saved_models/best_autoencoder.pth'))

    # Calculate the threshold based on the validation set errors using the best model
    val_errors = []
    model.eval()
    with torch.no_grad():
        for samples, labels in val_dataloader:
            samples, labels = samples.to(device), labels.to(device)

            # Compute errors only for label 0 samples
            mask = labels == 0
            samples = samples[mask]
            if len(samples) == 0:
                continue

            reconstructed, _ = model(samples)
            reconstruction_error = torch.mean((reconstructed - samples) ** 2, dim=[1, 2])
            val_errors.append(reconstruction_error)

    val_errors = torch.cat(val_errors).cpu().numpy()
    threshold = np.percentile(val_errors, 70)
    print(f"Computed Threshold: {threshold}")

    # Write the threshold value to a file
    with open('threshold.value', 'w') as f:
        f.write(str(threshold))

    # Test the model
    tester = Tester(model, test_dataloader, device, threshold)
    tester.test()

def finetune_model(config, device):
    # Hyperparameters
    input_size = config['model_params']['input_size']
    hidden_size = config['model_params']['hidden_size']
    num_layers = config['model_params']['num_layers']
    batch_size = config['finetuning_params']['batch_size']
    finetune_num_epochs = config['finetuning_params']['num_epochs']
    finetune_learning_rate = config['finetuning_params']['learning_rate']
    finetune_patience = config['finetuning_params']['patience']

    # Get dataloaders for fine-tuning
    finetune_data_path = config['data']['data_path']
    train_dataloader, val_dataloader, _ = get_dataloaders(finetune_data_path, batch_size=batch_size)

    # Initialize model
    model = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)

    # Load the best model
    model.load_state_dict(torch.load('saved_models/best_autoencoder.pth'))

    # Fine-tune the model
    finetuner = FineTuner(model, train_dataloader, val_dataloader, device, num_epochs=finetune_num_epochs, learning_rate=finetune_learning_rate, patience=finetune_patience)
    finetuner.fine_tune()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runner.py <config_file> <mode>")
        print("Modes: train, test, finetune")
        sys.exit(1)

    config_path = 'config.yaml'
    mode = sys.argv[1]

    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == "train":
        train_model(config, device)
    elif mode == "test":
        test_model(config, device)
    elif mode == "finetune":
        finetune_model(config, device)
    else:
        print("Invalid mode. Choose from train, test, or finetune.")
        sys.exit(1)
