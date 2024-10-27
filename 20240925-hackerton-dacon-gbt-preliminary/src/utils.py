

from typing import List, Tuple
import pandas as pd
import os
from src import root_path

def get_sample_submission(run_name:str, decoded_predictions: List[str]) -> Tuple[pd.DataFrame, str]:
    sample_submission = pd.read_csv(os.path.join(root_path, "./sample_submission.csv"))
    sample_submission["분류"] = decoded_predictions
    submission_filepath = os.path.join(root_path, f"./{run_name}.csv")
    sample_submission.to_csv(submission_filepath, encoding='UTF-8-sig', index=False)
    return sample_submission, submission_filepath



