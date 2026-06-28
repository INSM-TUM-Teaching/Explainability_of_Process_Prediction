import pandas as pd
from sklearn.preprocessing import LabelEncoder

def extract_case_outcomes(df, case_id_col='CaseID', timestamp_col='Timestamp', target_col='Activity'):
    """
    Extracts the final outcome for each case in the event log.
    The outcome is defined as the value of `target_col` at the latest `timestamp_col` 
    for each `case_id_col`.

    Args:
        df (pd.DataFrame): The standardized event log dataframe.
        case_id_col (str): The column name for Case IDs.
        timestamp_col (str): The column name for Timestamps.
        target_col (str): The column name to extract the outcome from (default is Activity).
        
    Returns:
        pd.Series: A series mapping CaseID to its final outcome.
    """
    # Ensure dataframe is sorted chronologically
    df_sorted = df.sort_values(by=[case_id_col, timestamp_col])
    
    # Take the last row for each case
    last_events = df_sorted.groupby(case_id_col).tail(1)
    
    # Create a mapping from case_id to the target outcome
    outcome_mapping = last_events.set_index(case_id_col)[target_col]
    
    return outcome_mapping

def encode_outcomes(outcome_mapping):
    """
    Encodes categorical outcomes into numerical labels using LabelEncoder.
    
    Args:
        outcome_mapping (pd.Series or list): The extracted outcomes.
        
    Returns:
        tuple: (encoded_outcomes, label_encoder_instance)
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(outcome_mapping)
    return encoded, encoder
