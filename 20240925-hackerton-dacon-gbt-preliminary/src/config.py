from types import SimpleNamespace

config = {
    "learning_rate": 1e-5,
    "epochs": 30,
    "batch_size": 64,
    "weight_decay": 0.01,
    "tokenizer_max_len": 256,
    "attention_type": "original_full",
    "model_name": "monologg/kobigbird-bert-base",
    "num_workers": 0,
    "pin_memory": True,
    "patience": 3,
    "warmup_ratio": 0.1,
    "expereiment_name": "Default",
}

CFG = SimpleNamespace(**config)
