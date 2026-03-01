"""
Data loading utilities for bayesm datasets

This module provides convenient functions to load all bayesm datasets
in their appropriate formats.
"""

import pandas as pd
from pathlib import Path

# Data directory is relative to this file
DATA_DIR = Path(__file__).parent / "data"


def load_cheese():
    """
    Load cheese dataset
    
    Returns
    -------
    DataFrame
        Sliced cheese sales data with columns: RETAILER, VOLUME, DISP, PRICE
    
    Notes
    -----
    Panel data with 5555 observations across 88 retailers.
    See R documentation: ?cheese
    """
    return pd.read_parquet(DATA_DIR / "cheese.parquet")


def load_customer_sat():
    """
    Load customerSat dataset
    
    Returns
    -------
    DataFrame
        Customer satisfaction data
    """
    return pd.read_parquet(DATA_DIR / "customerSat.parquet")


def load_scotch():
    """
    Load Scotch dataset
    
    Returns
    -------
    DataFrame
        Scotch whisky brand data
    """
    return pd.read_parquet(DATA_DIR / "Scotch.parquet")


def load_tuna():
    """
    Load tuna dataset
    
    Returns
    -------
    DataFrame
        Tuna purchase data
    """
    return pd.read_parquet(DATA_DIR / "tuna.parquet")


def load_bank():
    """
    Load bank dataset
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'choiceAtt': DataFrame with choice attributes
        - 'demo': DataFrame with demographic information
    
    Notes
    -----
    Bank card conjoint data with 946 respondents.
    """
    return {
        'choiceAtt': pd.read_parquet(DATA_DIR / "bank_choiceAtt.parquet"),
        'demo': pd.read_parquet(DATA_DIR / "bank_demo.parquet")
    }


def load_detailing():
    """
    Load detailing dataset
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'counts': DataFrame with count data
        - 'demo': DataFrame with demographic information
    """
    return {
        'counts': pd.read_parquet(DATA_DIR / "detailing_counts.parquet"),
        'demo': pd.read_parquet(DATA_DIR / "detailing_demo.parquet")
    }


def load_margarine():
    """
    Load margarine dataset
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'choicePrice': DataFrame with choice and price data
        - 'demos': DataFrame with demographic information
    """
    return {
        'choicePrice': pd.read_parquet(DATA_DIR / "margarine_choicePrice.parquet"),
        'demos': pd.read_parquet(DATA_DIR / "margarine_demos.parquet")
    }


def load_orange_juice():
    """
    Load orangeJuice dataset
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'yx': DataFrame with sales and pricing data
        - 'storedemo': DataFrame with store demographics
    
    Notes
    -----
    Store-level panel data on orange juice sales at 83 stores.
    """
    return {
        'yx': pd.read_parquet(DATA_DIR / "orangeJuice_yx.parquet"),
        'storedemo': pd.read_parquet(DATA_DIR / "orangeJuice_storedemo.parquet")
    }


def load_camera(format='long'):
    """
    Load camera dataset
    
    Parameters
    ----------
    format : str, default='long'
        'long' - Returns long-format DataFrame with id column
        'lgtdata' - Returns list of dicts (hierarchical format for models)
    
    Returns
    -------
    DataFrame or list
        If format='long': DataFrame with columns:
            - id: respondent identifier (1 to 332)
            - scenario: 1 to 16 per respondent
            - option: 1 to 5 per scenario
            - y: choice indicator (1 if chosen, 0 otherwise)
            - canon, sony, nikon, panasonic: brand indicators
            - pixels, zoom, video, swivel, wifi: feature indicators
            - price: price in $100s
        
        If format='lgtdata': List of dicts, one per respondent:
            [{'y': array, 'X': matrix}, ...]
            Compatible with hierarchical model functions.
    
    Notes
    -----
    Conjoint survey data for digital cameras with 332 respondents.
    Each respondent evaluated 16 scenarios with 5 options each.
    """
    df = pd.read_parquet(DATA_DIR / "camera.parquet")
    
    if format == 'long':
        return df
    
    elif format == 'lgtdata':
        import numpy as np
        lgtdata = []
        
        # Columns to exclude from X matrix
        id_cols = ['id', 'scenario', 'option', 'y']
        x_cols = [col for col in df.columns if col not in id_cols]
        
        for resp_id in sorted(df['id'].unique()):
            resp_data = df[df['id'] == resp_id].sort_values(['scenario', 'option'])
            lgtdata.append({
                'y': resp_data['y'].values.astype(int),
                'X': resp_data[x_cols].values
            })
        
        return lgtdata
    
    else:
        raise ValueError(f"Unknown format: {format}. Use 'long' or 'lgtdata'.")


# Convenience dictionary for all loaders
LOADERS = {
    'cheese': load_cheese,
    'customerSat': load_customer_sat,
    'Scotch': load_scotch,
    'tuna': load_tuna,
    'bank': load_bank,
    'detailing': load_detailing,
    'margarine': load_margarine,
    'orangeJuice': load_orange_juice,
    'camera': load_camera,
}


def load_data(dataset_name, **kwargs):
    """
    Load any bayesm dataset by name
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset. Available datasets:
        'cheese', 'customerSat', 'Scotch', 'tuna', 'bank', 
        'detailing', 'margarine', 'orangeJuice', 'camera'
    **kwargs
        Additional arguments passed to specific loader
        (e.g., format='lgtdata' for camera)
    
    Returns
    -------
    Data in appropriate format for the dataset
    
    Examples
    --------
    >>> # Load simple dataset
    >>> cheese = load_data('cheese')
    
    >>> # Load multi-component dataset
    >>> bank = load_data('bank')
    >>> bank['choiceAtt'].head()
    
    >>> # Load hierarchical dataset
    >>> camera_lgt = load_data('camera', format='lgtdata')
    """
    if dataset_name not in LOADERS:
        available = ', '.join(sorted(LOADERS.keys()))
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                        f"Available datasets: {available}")
    
    return LOADERS[dataset_name](**kwargs)


def list_datasets():
    """
    List all available datasets
    
    Returns
    -------
    list
        List of available dataset names
    """
    return sorted(LOADERS.keys())
