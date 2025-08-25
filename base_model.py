# function 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_feature_data(npz_path):
    """
    Load training and validation feature arrays from a compressed .npz file.

    Parameters:
        npz_path (str): Path to the saved .npz file.

    Returns:
        Tuple: X_train, y_train, X_val, y_val
    """
    data = np.load(npz_path)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val']


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and labels into training and validation sets.

    Parameters:
        X (np.ndarray): Feature matrix (2D or 3D).
        y (np.ndarray): Target vector.
        test_size (float): Fraction of data to use for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_val, y_train, y_val)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_model(X_train, y_train):
    """
    Trains a Linear Regression model.

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target values.

    Returns:
        Trained LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    """
    Uses the trained model to generate predictions.

    Parameters:
        model: A trained regression model with a `.predict()` method.
        X (np.ndarray): Feature matrix to predict on.

    Returns:
        np.ndarray: Predicted RUL values.
    """
    return model.predict(X)

def run_base_model_pipeline(npz_path, strategy='last'):
    """
    Full pipeline to train and run a linear model on feature data.

    Parameters:
        npz_path (str): Path to .npz file with preprocessed data.
        strategy (str): Feature strategy used ('last', 'mean', 'flat') — placeholder for consistency.

    Returns:
        Tuple: (model, X_val, y_val, y_pred)
    """
    # Step 1: Load data
    X, y, _, _ = load_feature_data(npz_path)

    # Step 2: Split into train/val
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Step 3: Train model
    model = train_linear_model(X_train, y_train)

    # Step 4: Predict
    y_pred = predict(model, X_val)

    return model, X_val, y_val, y_pred