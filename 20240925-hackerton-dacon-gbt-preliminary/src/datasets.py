import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, preprocess_datarows=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.encodings = []
        self._preprocess_datarows = preprocess_datarows
        if preprocess_datarows:
            print(f"Using {preprocess_datarows=}")
            self.encodings = self.tokenizer(
                self.texts,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self._preprocess_datarows:
            return {
                'input_ids': self.encodings['input_ids'][item],
                'attention_mask': self.encodings['attention_mask'][item],
                'labels': torch.tensor(self.labels[item], dtype=torch.long) if self.labels is not None else -1
            }
        
        text = str(self.texts[item])
        label = self.labels[item] if self.labels is not None else -1
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
