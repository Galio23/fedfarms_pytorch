import pandas as pd
import numpy as np
from config import config

def load_data():
    """
    Loads the CSV specified by config["client_data"], extracts features and targets.
    
    Returns:
        df (DataFrame): The entire loaded dataframe (for reference).
        X (DataFrame): Feature matrix (band columns).
        y (DataFrame): Target matrix (soil properties).
    """
    # Read the CSV
    df = pd.read_csv(config["client_data"])
    
    # Determine which band columns to use based on the configured satellite type
    satellite_type = config["satellite_type"].lower()
    if satellite_type == "landsat":
        band_cols = config["landsat_bands"]
    elif satellite_type == "sentinel":
        band_cols = config["sentinel_bands"]
    else:
        raise ValueError("Invalid satellite_type in config. Must be 'landsat' or 'sentinel'.")

    # Extract target columns (soil properties), e.g. ["C_gkg_filtered", "Clay_gkg_filtered"]
    target_cols = config["soil_properties"]
    
    # Create the feature (X) and target (y) dataframes
    X = df[band_cols].copy()
    y = df[target_cols].copy()
    
    return df, X, y

def partition_by_farm(df, X, y):
    """
    Partitions the data by the 'farm' column, assigning each unique farm to a different client.
    
    Args:
        df (DataFrame): The entire loaded dataframe (including the 'farm' column).
        X (DataFrame): Feature matrix.
        y (DataFrame): Target matrix.
        
    Returns:
        dict: Mapping of client_id -> (X_client, y_client).
    """
    if "farm" not in df.columns:
        raise ValueError("The dataframe must contain a 'farm' column for partitioning.")
    
    # Sort the unique farms so that farm_1 -> cid=0, farm_2 -> cid=1, etc.
    # (Assuming your farm labels are strictly 'farm_1', 'farm_2', ... 'farm_50'.)
    unique_farms = sorted(df["farm"].unique(), key=lambda f: int(f.split("_")[-1]))
    
    clients_data_dict = {}
    for cid, farm_val in enumerate(unique_farms):
        # Get rows for this particular farm
        idx = (df["farm"] == farm_val)
        X_client = X.loc[idx].reset_index(drop=True)
        y_client = y.loc[idx].reset_index(drop=True)
        
        # Store in a dictionary keyed by client_id
        clients_data_dict[cid] = (X_client, y_client)
    
    return clients_data_dict

def create_partitioned_datasets(cid=None):
    """
    Loads the dataset, partitions it by 'farm', and returns data for either all clients
    or just the specified client.
    
    Args:
        cid (int, optional): If provided, returns data only for that client ID.
        
    Returns:
        (X_client, y_client) if cid is not None, else a dict {cid: (X_client, y_client)}.
    """
    df, X, y = load_data()
    clients_data_dict = partition_by_farm(df, X, y)
    
    if cid is not None:
        return clients_data_dict[cid]
    return clients_data_dict
