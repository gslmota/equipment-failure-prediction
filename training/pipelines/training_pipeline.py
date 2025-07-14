import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report
from api.domain.entities import TrainingParameters

class TrainingPipeline:
    def create_sequences(self, data: pd.DataFrame, feature_cols: list, target_col: str, window: int) -> tuple:
        Xs, ys = [], []
        for i in range(len(data) - window):
            Xs.append(data[feature_cols].iloc[i : i + window].values)
            ys.append(data[target_col].iloc[i + window])
        return np.array(Xs), np.array(ys)

    def compute_sequence_stats(self, seqs: np.ndarray) -> np.ndarray:
        stats = []
        for seq in seqs:
            stats.append(np.concatenate([
                seq.mean(axis=0),
                seq.std(axis=0),
                seq.max(axis=0),
                seq.min(axis=0),
                np.diff(seq, axis=0).mean(axis=0)
            ]))
        return np.vstack(stats)

    def train_model(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, params: dict) -> CatBoostClassifier:
        model = CatBoostClassifier(
            iterations=500,
            depth=params['depth'],
            l2_leaf_reg=params['l2_leaf_reg'],
            learning_rate=params['learning_rate'],
            random_seed=42,
            verbose=100
        )
        model.fit(
            X, 
            y, 
            sample_weight=sample_weight,
            early_stopping_rounds=50
        )
        return model

    def evaluate_model(self, model, X_test, y_test, sample_weight) -> dict:
        probs = model.predict_proba(X_test)[:,1]
        ap = average_precision_score(y_test, probs, sample_weight=sample_weight)
        prec, rec, th = precision_recall_curve(y_test, probs, sample_weight=sample_weight)
        opt_idx = np.argmax(2*prec*rec/(prec+rec+1e-8))
        opt_th = th[opt_idx] if len(th)>0 else 0.5
        
        y_pred = (probs >= opt_th).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'average_precision': ap,
            'optimal_threshold': opt_th,
            'classification_report': report
        }

    def execute_training(self, df_tr: pd.DataFrame, df_te: pd.DataFrame, features: list, params: TrainingParameters) -> tuple:
        window = params.window_size
        
        # Criar sequências
        X_tr_seq, y_tr_seq = self.create_sequences(df_tr, features, 'Fail', window)
        X_te_seq, y_te_seq = self.create_sequences(df_te, features, 'Fail', window)
        
        # Calcular estatísticas das sequências
        X_tr_stats = self.compute_sequence_stats(X_tr_seq)
        X_te_stats = self.compute_sequence_stats(X_te_seq)
        
        # Pesos das amostras
        sw_tr = df_tr['preset_risk'].iloc[window:].values
        
        # Treinar modelo
        model = self.train_model(
            X_tr_stats, 
            y_tr_seq, 
            sw_tr,
            params.dict()
        )
        
        # Avaliar
        sw_te = df_te['preset_risk'].iloc[window:].values
        metrics = self.evaluate_model(model, X_te_stats, y_te_seq, sw_te)
        
        return model, metrics