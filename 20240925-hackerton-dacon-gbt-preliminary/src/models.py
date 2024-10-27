from transformers import AutoConfig, AutoTokenizer, BigBirdForSequenceClassification, AutoModelForSequenceClassification
from torch import nn

from src.roberta import RobertaForSequenceClassification 

import logging

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(
    model_name: str, 
    num_labels: int, 
    attention_type: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "monologg/kobigbird-bert-base":
        model = BigBirdForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            attention_type=attention_type
        )
        model.classifier.dense = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        model.classifier.out_proj = nn.Linear(model.config.hidden_size, num_labels)

    elif model_name == "klue/roberta-large":
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        if num_labels > 3:
            # TODO: Refactor this to be more general(configurable)
            config.focal_loss_alpha = 1
            config.focal_loss_gamma = 2
            config.focal_loss_reduction = "mean"
            config.problem_type = "sparse_multi_label_classification"
            logger.info("Using sparse multi label classification: Focal Loss")
        else:
            logger.info("Using binary classification: BCEWithLogitsLoss")
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
    elif model_name == "TfidfEnhancedClassifier":
        pass
        
    return model, tokenizer
