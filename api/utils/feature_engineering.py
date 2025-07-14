import pandas as pd
import numpy as np

class FeatureEngineer:
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features to the input DataFrame.
        """
        df = df.copy()
        base_cols = ['Temperature','Pressure','VibrationX','VibrationY','VibrationZ','Frequency']
        
        for col in base_cols:
            df[f'{col}_gradient'] = df[col].diff().fillna(0)
        
        df['vib_magnitude'] = np.sqrt(
            df['VibrationX']**2 + df['VibrationY']**2 + df['VibrationZ']**2
        )
        df['vib_combined'] = df['VibrationY'] + 0.7 * df['VibrationZ']
        df['temp_vib_ratio'] = df['Temperature'] / (df['vib_combined'] + 1e-5)
        df['pressure_diff'] = df['Pressure'].diff(3).abs().fillna(0)
        
        df['outlier_flag'] = (
            (df['Pressure'] > 150) |
            (df['VibrationY'] > 120) |
            (df['VibrationZ'] > 100)
        ).astype(int)
        
        return df