import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict

class DataPipeline:
    def load_data(self, path: str, sheet_name: str = 'O&G Equipment Data') -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet_name)
        # padroniza cabeçalhos caso haja espaços extras
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        df['Fail'] = df['Fail'].fillna(0).astype(int)
        return df

    def split_data(self, df: pd.DataFrame, train_frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * train_frac)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    
    def add_physical_spectral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        base_cols = ['Temperature','Pressure','VibrationX','VibrationY','VibrationZ','Frequency']
        for col in base_cols:
            df[f'{col}_gradient'] = df[col].diff().fillna(0)
        df['vib_magnitude'] = np.sqrt(df['VibrationX']**2 + df['VibrationY']**2 + df['VibrationZ']**2)
        df['vib_combined']  = df['VibrationY'] + 0.7 * df['VibrationZ']
        df['temp_vib_ratio'] = df['Temperature'] / (df['vib_combined'] + 1e-5)
        df['pressure_diff'] = df['Pressure'].diff(3).abs().fillna(0)
        df['outlier_flag']  = ((df['Pressure'] > 150) |
                               (df['VibrationY'] > 120) |
                               (df['VibrationZ'] > 100)).astype(int)
        return df

    def compute_preset_risk(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[Tuple, float]:
        risk = df_train.groupby(['Preset_1','Preset_2'])['Fail'].mean()
        # aplica diretamente no train/test
        df_train['preset_risk'] = df_train.set_index(['Preset_1','Preset_2']).index.map(risk)
        df_test ['preset_risk'] = df_test .set_index(['Preset_1','Preset_2']).index.map(risk).fillna(0)
        return risk.to_dict()

    def encode_presets(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_tr = pd.get_dummies(df_train, columns=['Preset_1','Preset_2'], prefix=['p1','p2'])
        df_te = pd.get_dummies(df_test,  columns=['Preset_1','Preset_2'], prefix=['p1','p2'])
        # garante consistência de colunas
        for col in df_tr.columns:
            if col not in df_te:
                df_te[col] = 0
        df_te = df_te[df_tr.columns]
        return df_tr, df_te

    def compute_baseline_stats(self, df_train: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
        baseline = df_train[df_train['Fail']==0][numeric_cols]
        return baseline.mean(), baseline.std(ddof=0)

    def compute_anomaly_features(self, df: pd.DataFrame, numeric_cols: List[str],
                                 means: pd.Series, stds: pd.Series) -> pd.DataFrame:
        df = df.copy()
        z = df[numeric_cols].sub(means).div(stds).abs()
        df['zscore_max'] = z.max(axis=1)
        causes = ['NoFail'] + numeric_cols
        df['cause_dom'] = 'NoFail'
        idx_fail = df['Fail']==1
        df.loc[idx_fail,'cause_dom'] = z[idx_fail].idxmax(axis=1)
        df['cause_dom'] = pd.Categorical(df['cause_dom'], categories=causes)
        df['cause_dom_code'] = df['cause_dom'].cat.codes.replace({-1: len(causes)})
        df.drop(columns=['cause_dom'], inplace=True)
        return df

    def add_continuous_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx_fail = df.index[df['Fail']==1]
        df['dist_to_last_fail'] = df.index.to_series().apply(
            lambda i: i - idx_fail[idx_fail<=i].max() if any(idx_fail<=i) else 1e6
        )
        for k in (5,10,20):
            df[f'fail_roll_{k}'] = df['Fail'].rolling(k,min_periods=1).sum().shift().fillna(0).astype(int)
        num_cols = [c for c in df.columns
                    if c.endswith('_gradient')
                    or c in ['vib_magnitude','vib_combined','temp_vib_ratio','pressure_diff']]
        for col in num_cols:
            df[f'{col}_trend'] = df[col].rolling(5,min_periods=1).mean().shift().fillna(0)
        return df

    def full_preprocessing(
        self,
        df_tr: pd.DataFrame,
        df_te: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], MinMaxScaler, Dict[Tuple, float]]:
        # 1) spectral features
        df_tr = self.add_physical_spectral_features(df_tr)
        df_te = self.add_physical_spectral_features(df_te)

        # 2) preset risk (salvo em dicionário)
        preset_risk = self.compute_preset_risk(df_tr, df_te)

        # 3) codifica presets em dummies
        df_tr, df_te = self.encode_presets(df_tr, df_te)

        # 4) estatísticas para anomaly
        numeric_cols = [
            'Temperature','Pressure','VibrationX','VibrationY','VibrationZ','Frequency',
            'Temperature_gradient','Pressure_gradient','VibrationX_gradient','VibrationY_gradient',
            'VibrationZ_gradient','Frequency_gradient','vib_magnitude','vib_combined',
            'temp_vib_ratio','pressure_diff'
        ]
        means, stds = self.compute_baseline_stats(df_tr, numeric_cols)

        # 5) features de anomalia
        df_tr = self.compute_anomaly_features(df_tr, numeric_cols, means, stds)
        df_te = self.compute_anomaly_features(df_te, numeric_cols, means, stds)

        # 6) features temporais contínuas
        df_full = pd.concat([df_tr, df_te])
        df_full = self.add_continuous_temporal_features(df_full)
        df_tr = df_full.iloc[:len(df_tr)].copy()
        df_te = df_full.iloc[len(df_tr):].copy()

        # 7) normalização
        exclude  = ['Fail','Cycle']
        features = [c for c in df_tr.columns if c not in exclude]
        scaler   = MinMaxScaler()
        df_tr[features] = scaler.fit_transform(df_tr[features])
        df_te[features] = scaler.transform(df_te[features])

        return df_tr, df_te, features, scaler, preset_risk
