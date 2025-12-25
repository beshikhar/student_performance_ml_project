import os
import sys
import pickle
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import r2_score

        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(
                model,
                params,
                cv=3,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
