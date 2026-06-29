import pandas as pd
import numpy as np


def extract_case_outcomes(df, outcome_column=None):
    """
    Extract the outcome label for each case.

    If outcome_column is provided, use the first value of that column per case.
    Otherwise, use the last activity of each case as the outcome.

    Returns a Series mapping case_id -> outcome label.
    """
    if outcome_column and outcome_column in df.columns:
        return df.groupby('case_id')[outcome_column].first()

    return df.groupby('case_id')['activity'].last()


def attach_outcome_labels(df, outcomes):
    """
    Add an 'outcome' column to df by mapping each row's case_id
    to its outcome label.
    """
    df = df.copy()
    df['outcome'] = df['case_id'].map(outcomes)
    return df
