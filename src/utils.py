import os
import sys

import numpy as np
import pandas as pd
import dill
import joblib
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

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
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, scoring="f1", n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else y_test_pred
            )

            # F1 Score - primary metric for imbalanced fraud detection
            f1 = f1_score(y_test, y_test_pred)

            # Additional metrics for comprehensive evaluation
            precision = precision_score(y_test, y_test_pred, zero_division=0)
            recall = recall_score(y_test, y_test_pred, zero_division=0)
            roc_auc = (
                roc_auc_score(y_test, y_test_proba)
                if hasattr(model, "predict_proba")
                else 0.5
            )

            # Store F1 as primary score, but also keep other metrics
            report[list(models.keys())[i]] = {
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as pickle_error:
        try:
            return joblib.load(file_path)
        except Exception as load_error:
            raise CustomException(load_error, sys) from pickle_error
