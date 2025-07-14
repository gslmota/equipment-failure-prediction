
import pandas as pd
from catboost import CatBoostClassifier
from api.domain.entities import TrainingParameters
from api.repositories.model_repository import ModelRepository
from training.pipelines.data_pipeline import DataPipeline
from training.pipelines.training_pipeline import TrainingPipeline
from logging import getLogger

logger = getLogger(__name__)

class TrainingService:
    def __init__(self, model_repo: ModelRepository):
        self.model_repo        = model_repo
        self.data_pipeline     = DataPipeline()
        self.training_pipeline = TrainingPipeline()
    
    def train_model(self, params: TrainingParameters) -> dict:
        """
        Trains a new model in the background.
        """
        df = self.data_pipeline.load_data(params.data_path, params.sheet_name)
        logger.info("Colunas carregadas:", df.columns.tolist())

        df_tr, df_te = self.data_pipeline.split_data(df)

        df_tr, df_te, features, scaler, preset_risk = \
            self.data_pipeline.full_preprocessing(df_tr, df_te)

        model, metrics = self.training_pipeline.execute_training(
            df_tr, df_te, features, params
        )
        logger.info("Treinamento concluído")
        logger.info("Metrics:", metrics)
    

        artifacts = {
            'model': model,
            'scaler': scaler,
            'metadata': {
                'features'     : features,
                'window_size'  : params.window_size,
                'preset_risk'  : preset_risk,
                'training_date': pd.Timestamp.now().isoformat(),
                'metrics'      : metrics
            }
        }
        self.model_repo.save_artifacts(artifacts)
        logger.info("Artefatos salvo com sucesso")    
        return metrics
