import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
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
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced", random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
                "XGBoost": XGBClassifier(
                    scale_pos_weight=10, random_state=42, eval_metric="logloss"
                ),
                "CatBoost": CatBoostClassifier(
                    verbose=False, random_state=42, auto_class_weights=True
                ),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [5, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [10, 20, 30, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "class_weight": ["balanced", "balanced_subsample"],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.8, 0.9, 1.0],
                },
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "scale_pos_weight": [10, 50, 100],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                },
                "CatBoost": {
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "iterations": [100, 200, 300],
                    "l2_leaf_reg": [1, 3, 5],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5, 1.0],
                    "algorithm": ["SAMME", "SAMME.R"],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Log all model results
            logging.info("Model Evaluation Results:")
            for model_name, metrics in model_report.items():
                logging.info(
                    f"{model_name}: F1={metrics['f1_score']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                    f"ROC-AUC={metrics['roc_auc']:.4f}"
                )

            # Get best model based on F1 score (primary metric for imbalanced classification)
            best_model_name = max(
                model_report, key=lambda x: model_report[x]["f1_score"]
            )
            best_model_score = model_report[best_model_name]["f1_score"]
            best_model = models[best_model_name]

            if best_model_score < 0.1:
                raise CustomException(
                    "No best model found - F1 score too low for fraud detection"
                )

            logging.info(
                f"Best model: {best_model_name} with F1 score: {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            # Print detailed classification report
            logging.info(f"\nClassification Report for {best_model_name}:")
            logging.info(f"\n{classification_report(y_test, predicted)}")

            f1 = f1_score(y_test, predicted)
            return f1

        except Exception as e:
            raise CustomException(e, sys)
