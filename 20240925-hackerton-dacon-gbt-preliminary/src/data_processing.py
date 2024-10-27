import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def load_data(root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(f"{root_path}/train.csv")
    test_df = pd.read_csv(f"{root_path}/test.csv")
    return train_df, test_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    import re
    def normalize_title(text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text.strip()

    def normalize_keywords(text):
        text = re.sub(r'[^가-힣\s,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.strip()

    def clean_keywords(keywords):
        keywords = keywords.split(',')
        keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        return ' '.join(keywords)

    df['title'] = df['제목'].apply(normalize_title)
    df['keywords'] = df['키워드'].apply(lambda x: clean_keywords(normalize_keywords(x)))
    df['text'] = df['title'] + ' [SEP] ' + df['keywords']
    return df

def get_label_encoded(df: pd.DataFrame, model_type: str = "") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Encode labels based on the model type and return the encoded dataframe and label encoder.

    Args:
        df (pd.DataFrame): Input dataframe.
        model_type (str): Type of model ('primary', 'secondary', or '').

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: Encoded dataframe and label encoder dictionary.
    """
    processed_df = df.copy()

    if model_type == "primary":
        processed_df.loc[processed_df['분류'] != '지역', '분류'] = '비지역'
    elif model_type == "secondary":
        processed_df = processed_df[processed_df['분류'] != '지역']

    label_encoder = {label: i for i, label in enumerate(processed_df['분류'].unique())}
    processed_df['label'] = processed_df['분류'].map(label_encoder)

    return processed_df, label_encoder

def split_data(df: pd.DataFrame, test_size: float = 0.2, stratify_col: str='label') -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=42)