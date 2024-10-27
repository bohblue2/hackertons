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

    trained_model_path = f"{root_path}/single_magic-frost-42.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()
    model.to(device)

    # Inference
    label_decoder = {i: label for label, i in label_encoder.items()}
    predictions = get_test_predictions(model, test_loader, device, label_decoder)
    sample_submission = pd.read_csv(os.path.join(root_path, "./sample_submission.csv"))
    sample_submission["분류"] = predictions 
    submission_filepath = f"./single_magic-frost-42.csv"
    sample_submission.to_csv(submission_filepath, encoding='UTF-8-sig', index=False)

    from dacon_submit_api import dacon_submit_api 

    dacon_token = 'd5ea5ec7e519de6412291ab093463dc54315baa080104aeee57ae0ab51543149'
    result = dacon_submit_api.post_submission_file(
    submission_filepath, 
    dacon_token, 
    '236372', 
    '월화수목김토일', 
    'submission 메모 내용' )

if __name__ == '__main__':
    main()
