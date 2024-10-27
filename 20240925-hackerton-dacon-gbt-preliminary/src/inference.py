

import torch
from tqdm import tqdm

def get_test_predictions(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    label_decoder: dict
) -> list:
    model.eval()
    test_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            test_predictions.extend(preds.cpu().tolist())

    decoded_predictions = [label_decoder[pred] for pred in test_predictions]
    return decoded_predictions