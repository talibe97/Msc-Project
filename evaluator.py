import numpy as np

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print RMSE and MAE for a given model.
    """
    rmse_score = rmse(y_true, y_pred)
    mae_score = mae(y_true, y_pred)
    
    print(f"{model_name} Evaluation:")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  MAE : {mae_score:.4f}")
    
    return {"model": model_name, "RMSE": rmse_score, "MAE": mae_score}