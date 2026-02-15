import pytest
import pandas as pd
import sys
import os

# Add the parent directory to sys.path to allow imports from app, utils, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_hcahps_data():
    """
    Returns a small sample DataFrame mimicking the HCAHPS structure.
    """
    data = {
        'Facility ID': ['001', '001', '002', '002'],
        'Facility Name': ['Hospital A', 'Hospital A', 'Hospital B', 'Hospital B'],
        'State': ['TX', 'TX', 'CA', 'CA'],
        'City/Town': ['Austin', 'Austin', 'San Francisco', 'San Francisco'],
        'measure_id': ['H_CLEAN_STAR_RATING', 'H_QUIET_STAR_RATING', 'H_CLEAN_STAR_RATING', 'H_QUIET_STAR_RATING'],
        'star_rating': ['5', '4', '3', '2']
    }
    return pd.DataFrame(data)
