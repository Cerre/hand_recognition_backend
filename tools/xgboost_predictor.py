import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from train_xgboost_model import extract_features

class XGBoostPredictor:
    def __init__(self):
        """Initialize the XGBoost predictor."""
        model_dir = Path("models")
        self.model = xgb.Booster()
        self.model.load_model(str(model_dir / "gesture_model.xgb"))
        self.scaler = joblib.load(model_dir / "feature_scaler.joblib")
    
    def __call__(self, points: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """Predict if a finger is extended.
        
        This interface matches the geometric methods for compatibility.
        However, the XGBoost model makes a full hand prediction and caches it
        to avoid redundant computation for the same frame.
        """
        # Extract finger index (0-4) from tip_idx
        if tip_idx == 4:
            finger_idx = 0  # thumb
        else:
            finger_idx = (tip_idx - 8) // 4 + 1  # other fingers
        
        # Check if we need to make a new prediction
        if not hasattr(self, '_last_points') or not np.array_equal(self._last_points, points):
            # Extract features
            features = extract_features(points)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            dmatrix = xgb.DMatrix(features_scaled)
            pred_probs = self.model.predict(dmatrix)
            self._last_prediction = pred_probs.argmax()
            self._last_points = points.copy()
        
        # Return True if the predicted number of extended fingers
        # is greater than the current finger index
        return finger_idx < self._last_prediction

# Create a singleton instance
xgboost_method = XGBoostPredictor() 