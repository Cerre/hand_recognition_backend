import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import sys
import os

# Add parent directory to path to import train_xgboost_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.train_xgboost_model import extract_features

class XGBoostPredictor:
    def __init__(self):
        """Initialize the XGBoost predictor."""
        model_dir = Path("models")
        self.model = xgb.Booster()
        self.model.load_model(str(model_dir / "gesture_model.xgb"))
        self.scaler = joblib.load(model_dir / "feature_scaler.joblib")
        # Remove caching attributes
        # self._last_points = None
        # self._last_prediction = None

    def predict_count(self, points: np.ndarray) -> int:
        """Predict the total number of extended fingers for the given hand landmarks."""
        if points is None or points.shape[0] == 0:
             # Handle case with no landmarks gracefully
             return 0 
             
        try:
            # Extract features
            features = extract_features(points)
            if features is None or features.size == 0:
                # Handle potential errors in feature extraction
                return 0
                
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Make prediction
            dmatrix = xgb.DMatrix(features_scaled)
            pred_probs = self.model.predict(dmatrix)
            predicted_count = int(pred_probs.argmax()) # Get the count directly
            return predicted_count
        except Exception as e:
            # Log the error appropriately if logging is setup here
            # For now, return 0 on prediction error
            print(f"[XGBoostPredictor Error] Failed prediction: {e}") # Basic error print
            return 0

    # Remove the __call__ method entirely as it's misleading and inefficient
    # def __call__(self, points: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    #     ...

# Create a singleton instance
xgboost_method = XGBoostPredictor() 