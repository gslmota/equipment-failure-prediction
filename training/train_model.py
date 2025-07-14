import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from training.pipelines.data_pipeline import DataPipeline
from training.pipelines.training_pipeline import TrainingPipeline
from api.domain.entities import TrainingParameters
from api.repositories.model_repository import ModelRepository

import logging

logger = logging.getLogger(__name__)

def main(data_path: str, sheet_name: str = 'O&G Equipment Data'):
    params = TrainingParameters(
        data_path=data_path,
        sheet_name=sheet_name
    )
    
    model_repo = ModelRepository()
    data_pipeline = DataPipeline()
    training_pipeline = TrainingPipeline()
    
    df = data_pipeline.load_data(params.data_path, params.sheet_name)
    df_tr, df_te = data_pipeline.split_data(df)
    df_tr, df_te, features, scaler = data_pipeline.full_preprocessing(df_tr, df_te)

    model, metrics = training_pipeline.execute_training(
        df_tr, df_te, features, params
    )
    
    artifacts = {
        'model': model,
        'scaler': scaler,
        'metadata': {
            'features': features,
            'window_size': params.window_size,
            'preset_risk': data_pipeline.compute_preset_risk(df_tr, df_te),
            'training_date': pd.Timestamp.now().isoformat(),
            'metrics': metrics
        }
    }
    model_repo.save_artifacts(artifacts)
    
    logger.info("Training completed successfully!")
    logger.info(f"Average Precision: {metrics['average_precision']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sheet_name', type=str, default='O&G Equipment Data')
    args = parser.parse_args()
    
    main(args.data_path, args.sheet_name)