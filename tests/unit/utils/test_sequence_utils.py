import numpy as np
import pandas as pd
from api.utils.sequence_utils import SequenceProcessor

def test_create_artificial_sequence():
    processor = SequenceProcessor()
    df = pd.DataFrame({
        'feature1': [0.1], 
        'feature2': [0.3]
    })
    sequence = processor.create_artificial_sequence(df, ['feature1', 'feature2'], 3)
    
    assert sequence.shape == (3, 2)  

def test_compute_sequence_stats():
    processor = SequenceProcessor()
    sequence = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    stats = processor.compute_sequence_stats(sequence)
    
    assert stats.shape == (1, 10)  
    
    mean = np.array([3, 4])
    std = np.array([1.63299316, 1.63299316])
    max_vals = np.array([5, 6])
    min_vals = np.array([1, 2])
    diff_mean = np.array([2, 2])
    
    expected = np.concatenate([mean, std, max_vals, min_vals, diff_mean])
    assert np.allclose(stats[0], expected)