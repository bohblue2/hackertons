import os
import wandb
import torch
import pandas as pd
from src.config import CFG
from src.data_processing import load_data, preprocess_data, get_label_encoded, split_data
from src.models import get_model_and_tokenizer
from src.datasets import TextDataset
from src.training import train_model
from src.inference import get_test_predictions
from src import root_path
from src.utils import get_sample_submission
from torch.utils.data import DataLoader

def main():
    CFG.model_type = "single"
    CFG.learning_rate = 1e-5 # initial learning rate 1e-5
    CFG.weight_decay = 0.05
    CFG.max_len = 256 + 128
    CFG.batch_size = 24
    CFG.model_name = "klue/roberta-large"
    CFG.patience = 5
    wrun = wandb.init(project="dacon-gbt-2024-hackerton", config=CFG)
    CFG.experiment_name = wrun.name

    # Data loading and preprocessing
    train_df, test_df = load_data(root_path=root_path)
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_df, label_encoder = get_label_encoded(train_df, model_type=CFG.model_type)
    train_df, val_df = split_data(train_df)

    # Model and tokenizer initialization
    model, tokenizer = get_model_and_tokenizer(CFG.model_name, len(label_encoder), CFG.attention_type)

    # Dataset and DataLoader creation
    train_dataset = TextDataset(train_df.text.tolist(), train_df.label.tolist(), tokenizer, max_len=CFG.tokenizer_max_len)
    val_dataset = TextDataset(val_df.text.to_list(), val_df.label.tolist(), tokenizer, max_len=CFG.tokenizer_max_len)
    test_dataset = TextDataset(test_df.text.tolist(), None, tokenizer, max_len=CFG.tokenizer_max_len)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

    # Model training
    trained_model = train_model(model, train_loader, val_loader, CFG)

    # Inference
    label_decoder = {i: label for label, i in label_encoder.items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = get_test_predictions(trained_model, test_loader, device, label_decoder)

if __name__ == "__main__":
    main()