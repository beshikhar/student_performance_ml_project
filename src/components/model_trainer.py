import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training started")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(objective="reg:squarederror"),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                },
                "Random Forest": {
                    "n_estimators": [64, 128],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [64, 128],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [64, 128],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [64, 128],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            best_model = models[best_model_name]

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info("Model training completed")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    train_path = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_path = os.path.join("artifacts", "data_ingestion", "test.csv")

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print(r2)
