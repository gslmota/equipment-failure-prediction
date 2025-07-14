import numpy as np
import pandas as pd

class SequenceProcessor:
    def create_artificial_sequence(self, df: pd.DataFrame, features: list, window_size: int) -> np.ndarray:
        """Create artificial sequence for inference in real-time"""
        return np.tile(df[features].values, (window_size, 1))
    
    def compute_sequence_stats(self, sequence: np.ndarray) -> np.ndarray:
        """Compute statistics for a sequence"""
        return np.concatenate([
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.max(axis=0),
            sequence.min(axis=0),
            np.diff(sequence, axis=0).mean(axis=0)
        ]).reshape(1, -1)