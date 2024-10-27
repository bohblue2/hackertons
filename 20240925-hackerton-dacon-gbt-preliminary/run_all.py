from src.config import CFG
from src.data_processing import get_label_encoded, load_data, preprocess_data
from src import root_path
import torch

from src.models import get_model_and_tokenizer


def load_and_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df = load_data(root_path=root_path)
    test_df = preprocess_data(test_df)
    _, primary_label_encoder = get_label_encoded(train_df, model_type='primary')
    _, secondary_label_encoder = get_label_encoded(train_df, model_type='secondary')
    primary_label_decoder = {i: label for label, i in primary_label_encoder.items()}
    secondary_label_decoder = {i: label for label, i in secondary_label_encoder.items()}
    primary_model, p_tokenizer = get_model_and_tokenizer(CFG.model_name, len(primary_label_encoder), CFG.attention_type)
    secondary_model, s_tokenizer = get_model_and_tokenizer(CFG.model_name, len(secondary_label_encoder), CFG.attention_type)

    primary_model_filepath = f"{root_path}/primary_best_model.pth"
    primary_model.load_state_dict(torch.load(primary_model_filepath, map_location=device))
    primary_model.eval()

    secondary_model_filepath = f"{root_path}/secondary_best_model.pth"
    secondary_model.load_state_dict(torch.load(secondary_model_filepath, map_location=device))
    secondary_model.eval()