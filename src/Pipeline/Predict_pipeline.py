import sys
from pathlib import Path
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


BASE_DIR = Path(__file__).resolve().parents[2]


class PredictPipeline:
    def __init__(self):
        pass

    @staticmethod
    def _resolve_artifact_pair():
        candidate_pairs = [
            (Path("artifacts") / "model.pkl", Path("artifacts") / "proprocessor.pkl"),
            (Path("artifacts") / "model.pkl", Path("artifacts") / "preprocessor.pkl"),
            (Path("Notebook") / "model.pkl", Path("Notebook") / "proprocessor.pkl"),
            (Path("Notebook") / "model.pkl", Path("Notebook") / "preprocessor.pkl"),
            (Path("Notebook") / "model.pkl", Path("artifacts") / "proprocessor.pkl"),
            (Path("Notebook") / "model.pkl", Path("artifacts") / "preprocessor.pkl"),
        ]

        for model_relative, preprocessor_relative in candidate_pairs:
            model_candidate = BASE_DIR / model_relative
            preprocessor_candidate = BASE_DIR / preprocessor_relative
            if model_candidate.exists() and preprocessor_candidate.exists():
                return model_candidate, preprocessor_candidate

        searched_pairs = ", ".join(
            f"({BASE_DIR / model_path}, {BASE_DIR / preprocessor_path})"
            for model_path, preprocessor_path in candidate_pairs
        )
        raise FileNotFoundError(
            "Matching model/preprocessor artifact pair not found. Checked: "
            f"{searched_pairs}"
        )

    @staticmethod
    def _align_features(features: pd.DataFrame, expected_columns):
        if not expected_columns:
            return features

        aligned_features = features.copy()

        if (
            "balance_diff_orig" in expected_columns
            and "balance_diff_orig" not in aligned_features.columns
            and {"oldbalanceOrg", "newbalanceOrig"}.issubset(aligned_features.columns)
        ):
            aligned_features["balance_diff_orig"] = (
                aligned_features["oldbalanceOrg"] - aligned_features["newbalanceOrig"]
            )

        if (
            "balance_diff_dest" in expected_columns
            and "balance_diff_dest" not in aligned_features.columns
            and {"oldbalanceDest", "newbalanceDest"}.issubset(aligned_features.columns)
        ):
            aligned_features["balance_diff_dest"] = (
                aligned_features["newbalanceDest"] - aligned_features["oldbalanceDest"]
            )

        for column in expected_columns:
            if column in aligned_features.columns:
                continue
            if column.lower().startswith("name") or column == "type":
                aligned_features[column] = "unknown"
            else:
                aligned_features[column] = 0

        return aligned_features[expected_columns]

    def predict(self, features):
        try:
            model_path, preprocessor_path = self._resolve_artifact_pair()
            print("Before Loading")
            model = load_object(file_path=str(model_path))
            print("After Loading")

            # If model is a pipeline that already includes preprocessing, feed raw aligned data.
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                expected_columns = list(getattr(model, "feature_names_in_", []))
                aligned_features = self._align_features(features, expected_columns)
                preds = model.predict(aligned_features)
                return preds

            preprocessor = load_object(file_path=str(preprocessor_path))
            expected_columns = list(getattr(preprocessor, "feature_names_in_", []))
            aligned_features = self._align_features(features, expected_columns)
            data_scaled = preprocessor.transform(aligned_features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        type: str,
        amount: float,
        oldbalanceOrg: float,
        newbalanceOrig: float,
        oldbalanceDest: float,
        newbalanceDest: float,
    ):

        self.type = type
        self.amount = amount
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "type": [self.type],
                "amount": [self.amount],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
