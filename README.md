failure-prediction-api/
├── api
│   ├── controllers
│   │   ├── health_controller.py
│   │   ├── predict_controller.py
│   │   └── train_controller.py
│   ├── dependencies
│   │   └── model_dependencies.py
│   ├── dependencies.py
│   ├── Dockerfile
│   ├── domain
│   │   └── entities.py
│   ├── main.py
│   ├── repositories
│   │   └── model_repository.py
│   ├── routes
│   │   ├── health.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── services
│   │   ├── inference_service.py
│   │   └── training_service.py
│   └── utils
│       ├── feature_engineering.py
│       └── sequence_utils.py
├── data.xlsx
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── streamlit_app
│   ├── app.py
│   └── Dockerfile
├── tests
│   ├── conftest.py
│   ├── integration
│   │   └── test_api.py
│   └── unit
│       ├── controllers
│       │   └── test_predict_controller.py
│       ├── repositories
│       │   └── test_model_repository.py
│       ├── services
│       │   ├── test_inference_service.py
│       │   └── test_training_service.py
│       └── utils
│           ├── test_feature_engineering.py
│           └── test_sequence_utils.py
└── training
    ├── artifacts
    │   ├── metadata.pkl
    │   ├── model.cbm
    │   └── scaler.pkl
    ├── pipelines
    │   ├── data_pipeline.py
    │   └── training_pipeline.py
    └── train_model.py