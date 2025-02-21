import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Add parent directory to path to import hand_recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle between three points."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def compute_palm_center(landmarks: np.ndarray) -> np.ndarray:
    """Compute palm center using wrist and MCP joints."""
    palm_points = landmarks[[0, 5, 9, 13, 17]]  # Wrist and all MCP joints
    return np.mean(palm_points, axis=0)

def compute_hand_size(landmarks: np.ndarray) -> float:
    """Compute hand size as distance between wrist and middle finger MCP."""
    return np.linalg.norm(landmarks[9] - landmarks[0])

def normalize_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute normalized distance between two points."""
    return np.linalg.norm(p1 - p2)

def compute_thumb_bend_angle(landmarks: np.ndarray) -> float:
    """Compute thumb bend angle using tip and two joints below.
    
    Uses:
    - Thumb tip (4)
    - IP joint (3)
    - MCP joint (2)
    """
    tip = landmarks[4]    # Tip
    ip = landmarks[3]     # IP joint
    mcp = landmarks[2]    # MCP joint
    
    # Calculate angle between tip-IP and IP-MCP vectors
    tip_to_ip = tip - ip
    ip_to_mcp = ip - mcp
    
    # Calculate angle in degrees
    cos_angle = np.dot(tip_to_ip, ip_to_mcp) / (np.linalg.norm(tip_to_ip) * np.linalg.norm(ip_to_mcp))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def extract_thumb_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract thumb-specific features."""
    # Thumb angles
    thumb_angles = [
        compute_thumb_bend_angle(landmarks),  # Bend angle at IP joint
        compute_angle(landmarks[3], landmarks[2], landmarks[1]),  # DIP-PIP-MCP
        compute_angle(landmarks[2], landmarks[1], landmarks[0])   # PIP-MCP-WRIST
    ]
    
    # Thumb position relative to index finger
    thumb_index_dist = normalize_distance(landmarks[4], landmarks[8])
    
    # Thumb orientation relative to palm
    palm_center = compute_palm_center(landmarks)
    thumb_palm_angle = compute_angle(landmarks[4], landmarks[1], palm_center)
    
    return np.array([*thumb_angles, thumb_index_dist, thumb_palm_angle])

def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract features from hand landmarks."""
    # Normalize landmarks by hand size and center
    hand_size = compute_hand_size(landmarks)
    palm_center = compute_palm_center(landmarks)
    landmarks_normalized = (landmarks - palm_center) / hand_size
    
    # 1. Finger angles
    finger_angles = []
    for tip, pip, mcp in [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]:
        angle = compute_angle(landmarks_normalized[tip], 
                            landmarks_normalized[pip],
                            landmarks_normalized[mcp])
        finger_angles.append(angle)
    
    # 2. Distances from palm center
    tip_distances = []
    for tip in [4, 8, 12, 16, 20]:  # all fingertips
        dist = normalize_distance(landmarks_normalized[tip], np.zeros(3))
        tip_distances.append(dist)
    
    # 3. Relative heights
    height_features = []
    for tip in [4, 8, 12, 16, 20]:
        relative_height = landmarks_normalized[tip][1]
        height_features.append(relative_height)
    
    # 4. Inter-finger relationships (between adjacent fingertips)
    finger_spreads = []
    tips = [4, 8, 12, 16, 20]  # thumb and fingertips
    for i in range(len(tips)-1):
        tip1, tip2 = landmarks_normalized[tips[i]], landmarks_normalized[tips[i+1]]
        spread = normalize_distance(tip1, tip2)
        finger_spreads.append(spread)
    
    # 5. Thumb specific features
    thumb_features = extract_thumb_features(landmarks_normalized)
    
    # Combine all features
    features = np.concatenate([
        finger_angles,      # 4 features
        tip_distances,      # 5 features
        height_features,    # 5 features
        finger_spreads,     # 4 features
        thumb_features      # 5 features
    ])
    
    return features

def load_dataset(base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset and extract features."""
    features_list = []
    labels = []
    
    # Iterate through all person directories
    for person_dir in base_dir.glob("person*"):
        # Iterate through handedness (left/right)
        for hand_dir in person_dir.glob("*"):
            # Iterate through all number directories
            for number_dir in hand_dir.glob("[0-5]"):
                number = int(number_dir.name)
                landmark_dir = number_dir / "landmarks"
                
                # Load all landmark files for this number
                for landmark_file in landmark_dir.glob("*.json"):
                    with open(landmark_file, 'r') as f:
                        data = json.load(f)
                        landmarks = np.array(data['landmarks'])
                        features = extract_features(landmarks)
                        features_list.append(features)
                        labels.append(number)
    
    return np.array(features_list), np.array(labels)

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced dataset."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weights = {int(c): total / (n_classes * count) for c, count in zip(classes, counts)}
    return weights

def train_model(X: np.ndarray, y: np.ndarray, class_weights: Dict[int, float]) -> Tuple[xgb.Booster, StandardScaler]:
    """Train XGBoost model with cross-validation."""
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': 6,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',  # CPU-based histogram method
        'device': 'cpu',  # Force CPU
        'eval_metric': ['mlogloss', 'merror']
    }
    
    # Create DMatrix with weights
    sample_weights = np.array([class_weights[label] for label in y_train])
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    
    # Train model
    print("\nTraining XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Evaluate model
    y_pred = model.predict(dtest).argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("  Actual → Predicted")
    print("    " + " ".join(f"{i:4}" for i in range(6)))
    for i, row in enumerate(cm):
        print(f"  {i}: " + " ".join(f"{n:4}" for n in row))
    
    return model, scaler

def save_model(model: xgb.Booster, scaler: StandardScaler, output_dir: Path):
    """Save model and scaler."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "gesture_model.xgb"))
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

def main():
    # Load dataset
    print("Loading dataset...")
    base_dir = Path("ml_data")
    X, y = load_dataset(base_dir)
    print(f"Loaded {len(X)} samples")
    
    # Compute class weights
    class_weights = compute_class_weights(y)
    print("\nClass weights:")
    for class_idx, weight in class_weights.items():
        count = np.sum(y == class_idx)
        print(f"Class {class_idx}: {count} samples, weight = {weight:.2f}")
    
    # Train model
    model, scaler = train_model(X, y, class_weights)
    
    # Save model
    output_dir = Path("models")
    save_model(model, scaler, output_dir)
    print(f"\nModel saved to {output_dir}")

if __name__ == "__main__":
    main() 