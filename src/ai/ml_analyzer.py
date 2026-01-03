"""
Machine Learning & AI Analysis Module

Provides AI-powered signal analysis capabilities including:
- Anomaly detection
- Signal classification
- Pattern recognition
- Predictive analysis
- Clustering analysis

Author: EduCatCode - AI Engineering Department
Version: 2.1.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import logging

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class AnomalyDetector:
    """Detect anomalies in signal data using machine learning."""

    def __init__(self, method: str = 'isolation_forest',
                logger: Optional[logging.Logger] = None):
        """
        Initialize anomaly detector.

        Args:
            method: Detection method ('isolation_forest', 'one_class_svm', 'statistical')
            logger: Optional logger instance
        """
        self.method = method
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, features: np.ndarray, contamination: float = 0.1):
        """
        Train anomaly detection model.

        Args:
            features: Feature matrix (n_samples, n_features)
            contamination: Expected proportion of outliers (0.0 to 0.5)
        """
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model.fit(features_scaled)
        self.logger.info(f"Anomaly detector trained using {self.method}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in new data.

        Args:
            features: Feature matrix

        Returns:
            Array of predictions (1 = normal, -1 = anomaly)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)

        n_anomalies = np.sum(predictions == -1)
        self.logger.info(f"Detected {n_anomalies} anomalies out of {len(predictions)} samples")

        return predictions

    def detect_signal_anomalies(self, signal_data: np.ndarray,
                                window_size: int = 100,
                                hop_length: int = 50) -> pd.DataFrame:
        """
        Detect anomalies in a signal using sliding window.

        Args:
            signal_data: Input signal array
            window_size: Size of analysis window
            hop_length: Step between windows

        Returns:
            DataFrame with anomaly scores and labels
        """
        features_list = []
        positions = []

        # Extract features from windows
        for i in range(0, len(signal_data) - window_size + 1, hop_length):
            window = signal_data[i:i + window_size]

            # Statistical features
            features = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
                np.sum(np.abs(np.diff(window)))  # Total variation
            ]

            features_list.append(features)
            positions.append(i + window_size // 2)

        features_array = np.array(features_list)

        # Train and predict
        self.fit(features_array)
        predictions = self.predict(features_array)

        # Create result DataFrame
        results = pd.DataFrame({
            'Position': positions,
            'Is_Anomaly': predictions == -1,
            'Anomaly_Score': -predictions  # Convert to positive scores
        })

        return results


class SignalClassifier:
    """Classify signals into different categories using machine learning."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize signal classifier.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, features: np.ndarray, labels: np.ndarray,
             test_size: float = 0.2) -> Dict[str, any]:
        """
        Train classification model.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            test_size: Proportion of test data

        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )

        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        y_pred = self.model.predict(X_test_scaled)

        self.logger.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels for new data.

        Args:
            features: Feature matrix

        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)

        return predictions

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: Feature matrix

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)

        return probabilities

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.feature_importances_

    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")


class PatternRecognition:
    """Recognize patterns in signals using unsupervised learning."""

    def __init__(self, n_clusters: int = 3,
                logger: Optional[logging.Logger] = None):
        """
        Initialize pattern recognition.

        Args:
            n_clusters: Number of patterns to identify
            logger: Optional logger instance
        """
        self.n_clusters = n_clusters
        self.logger = logger or logging.getLogger(__name__)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()

    def find_patterns(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify patterns in feature data using clustering.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        # Normalize
        features_scaled = self.scaler.fit_transform(features)

        # Cluster
        labels = self.kmeans.fit_predict(features_scaled)
        centers = self.kmeans.cluster_centers_

        self.logger.info(f"Found {self.n_clusters} patterns in data")

        return labels, centers

    def analyze_signal_patterns(self, signal_data: np.ndarray,
                                window_size: int = 100,
                                hop_length: int = 50) -> pd.DataFrame:
        """
        Analyze patterns in a signal using sliding window.

        Args:
            signal_data: Input signal array
            window_size: Size of analysis window
            hop_length: Step between windows

        Returns:
            DataFrame with pattern labels and statistics
        """
        features_list = []
        positions = []

        # Extract features
        for i in range(0, len(signal_data) - window_size + 1, hop_length):
            window = signal_data[i:i + window_size]

            features = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
            ]

            features_list.append(features)
            positions.append(i + window_size // 2)

        features_array = np.array(features_list)

        # Find patterns
        labels, centers = self.find_patterns(features_array)

        # Create result DataFrame
        results = pd.DataFrame({
            'Position': positions,
            'Pattern_ID': labels,
        })

        # Add pattern statistics
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            count = np.sum(mask)
            results.loc[mask, 'Pattern_Count'] = count
            results.loc[mask, 'Pattern_Percentage'] = (count / len(labels)) * 100

        return results


class DimensionalityReducer:
    """Reduce dimensionality of feature data for visualization and analysis."""

    def __init__(self, n_components: int = 2,
                logger: Optional[logging.Logger] = None):
        """
        Initialize dimensionality reducer.

        Args:
            n_components: Number of components to keep
            logger: Optional logger instance
        """
        self.n_components = n_components
        self.logger = logger or logging.getLogger(__name__)
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of features.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Reduced features (n_samples, n_components)
        """
        # Normalize
        features_scaled = self.scaler.fit_transform(features)

        # Reduce
        reduced = self.pca.fit_transform(features_scaled)

        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        self.logger.info(
            f"Reduced to {self.n_components} components "
            f"(explained variance: {explained_variance:.2%})"
        )

        return reduced

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        return self.pca.explained_variance_ratio_

    def get_components(self) -> np.ndarray:
        """Get principal components."""
        return self.pca.components_


class TimeSeriesPredictor:
    """Predict future signal values (basic implementation)."""

    def __init__(self, lookback: int = 10,
                logger: Optional[logging.Logger] = None):
        """
        Initialize time series predictor.

        Args:
            lookback: Number of past values to use for prediction
            logger: Optional logger instance
        """
        self.lookback = lookback
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            signal_data: Input signal array

        Returns:
            Tuple of (X, y) for training
        """
        X, y = [], []

        for i in range(len(signal_data) - self.lookback):
            X.append(signal_data[i:i + self.lookback])
            y.append(signal_data[i + self.lookback])

        return np.array(X), np.array(y)

    def train(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Train prediction model.

        Args:
            signal_data: Training signal array

        Returns:
            Dictionary with training metrics
        """
        X, y = self.prepare_data(signal_data)

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Simple model: Random Forest Regressor
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)

        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        self.logger.info(f"Predictor trained - R² score: {test_score:.3f}")

        return {
            'train_r2': train_score,
            'test_r2': test_score
        }

    def predict_next(self, recent_values: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict next values.

        Args:
            recent_values: Recent signal values (must be >= lookback length)
            n_steps: Number of steps to predict

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if len(recent_values) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} recent values")

        predictions = []
        current_window = recent_values[-self.lookback:].copy()

        for _ in range(n_steps):
            # Predict next value
            X = current_window.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            next_val = self.model.predict(X_scaled)[0]

            predictions.append(next_val)

            # Update window (sliding)
            current_window = np.append(current_window[1:], next_val)

        return np.array(predictions)
