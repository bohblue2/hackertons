import torch
import torch.nn as nn

from transformers import AutoModel

class TfidfEnhancedClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels, tfidf_dim, num_attention_heads=8):
        super(TfidfEnhancedClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)
        self.num_attention_heads = num_attention_heads
        hidden_size = self.base_model.config.hidden_size
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=0.1)
        
        # Layer normalization and feed-forward network for attention output
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.tfidf_layer = nn.Linear(tfidf_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, attention_mask, tfidf_features):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        
        # Apply multi-head attention
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, 
                                        key_padding_mask=~attention_mask.bool())
        
        # Add & Norm
        attn_output = self.layer_norm1(hidden_states + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(attn_output)
        
        # Add & Norm
        ff_output = self.layer_norm2(attn_output + ff_output)
        
        # Global average pooling
        context_vector = torch.mean(ff_output, dim=1)
        
        tfidf_vector = self.tfidf_layer(tfidf_features)
        
        combined_vector = torch.cat([context_vector, tfidf_vector], dim=1)
        pooled_output = self.dropout(combined_vector)
        logits = self.classifier(pooled_output)
        
        return logits