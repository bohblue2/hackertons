import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict 
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            unit="batch",
            ncols=100,
            leave=False,
            colour='blue'
        )
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar with current average loss
            current_loss = epoch_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        val_f1 = evaluate_model(model, val_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_f1_score": val_f1,
            "learning_rate": scheduler.get_last_lr()[0],
            "best_val_f1": best_val_f1
        })

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Best Validation F1 Score: {best_val_f1:.4f}")
        print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
        print("-" * 50)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), f'{config.model_type}_{config.experiment_name}.pth')
        else:
            counter += 1

        if counter >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            val_predictions.extend(preds.cpu().tolist())
            val_true_labels.extend(labels.cpu().tolist())

    return f1_score(val_true_labels, val_predictions, average='macro')
