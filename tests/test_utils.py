import pytest
import pandas as pd
from utils import get_agent_df

def test_get_agent_df_structure(sample_hcahps_data):
    """
    Test that get_agent_df correctly pivots the data and renames columns.
    """
    # Act
    agent_df = get_agent_df(sample_hcahps_data)
    
    # Assert
    assert 'Cleanliness' in agent_df.columns
    assert 'Quietness' in agent_df.columns
    assert 'Facility Name' in agent_df.columns
    
    # Check dimensions: Should be 2 hospitals (A and B)
    assert len(agent_df) == 2
    
    # Check values
    hosp_a = agent_df[agent_df['Facility Name'] == 'Hospital A'].iloc[0]
    assert hosp_a['Cleanliness'] == 5
    assert hosp_a['Quietness'] == 4

def test_get_agent_df_numeric_conversion(sample_hcahps_data):
    """
    Test that star ratings are converted to numbers.
    """
    agent_df = get_agent_df(sample_hcahps_data)
    assert pd.api.types.is_numeric_dtype(agent_df['Cleanliness'])
    assert pd.api.types.is_numeric_dtype(agent_df['Quietness'])
