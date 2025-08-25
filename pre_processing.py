# all Functions
# pre_processing

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

def drop_flat_sensors(df):
    """
    Identifies and removes flat (no-variance) sensor columns.

    Parameters:
        df (pd.DataFrame): Engine dataset.

    Returns:
        pd.DataFrame: Cleaned DataFrame with flat sensors removed.
    """
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    flat_sensors = [col for col in sensor_cols if df[col].nunique() == 1]
    return df.drop(columns=flat_sensors)

def summarise_engine_lifespans(df, dataset_name="FD001"):
    """
    Plots and summarises the distribution of engine lifespans for a given C-MAPSS dataset.

    Parameters:
        df (pd.DataFrame): The raw or preprocessed dataset.
        dataset_name (str): Optional label to include in the plot title.

    Returns:
        Tuple: (mean, std, min, max) of engine lifespans.
    """
    max_cycles_per_unit = df.groupby("unit_number")["time_in_cycles"].max()

    # Stats
    mean_cycles = max_cycles_per_unit.mean()
    std_cycles = max_cycles_per_unit.std()
    min_cycles = max_cycles_per_unit.min()
    max_cycles = max_cycles_per_unit.max()

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(max_cycles_per_unit, bins=20, edgecolor='black')
    plt.title(f"Distribution of Engine Lifespans ({dataset_name})")
    plt.xlabel("Max Cycles Before Failure")
    plt.ylabel("Number of Engines")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Dataset: {dataset_name}")
    print(f"Mean cycles to failure: {mean_cycles:.2f}")
    print(f"Standard deviation: {std_cycles:.2f}")
    print(f"Minimum: {min_cycles}")
    print(f"Maximum: {max_cycles}")

    return mean_cycles, std_cycles, min_cycles, max_cycles


def calculate_rul(df, max_rul=130):
    """
    Calculates Remaining Useful Life (RUL) for each time step and adds it to the DataFrame.

    Parameters:
        df (pd.DataFrame): Engine dataset containing 'unit_number' and 'time_in_cycles'.
        max_rul (int): Maximum RUL value to clip at. Default is 130.

    Returns:
        pd.DataFrame: DataFrame with an additional 'RUL' column.
    """
    df = df.copy()
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = (max_cycles - df['time_in_cycles']).clip(upper=max_rul)
    return df


def standardise_per_condition(df):
    """
    Applies z-score standardisation to sensor measurements within each unique operating condition.

    Parameters:
        df (pd.DataFrame): DataFrame with sensor columns and op_setting_1/2/3.

    Returns:
        pd.DataFrame: Standardised DataFrame.
    """
    df = df.copy()
    
    setting_cols = [col for col in df.columns if "op_setting" in col]
    sensor_cols = [col for col in df.columns if "sensor_measurement" in col]

    # Group by condition
    for key, group in df.groupby(setting_cols):
        scaler = StandardScaler()
        df.loc[group.index, sensor_cols] = scaler.fit_transform(group[sensor_cols])

    return df


def generate_sliding_windows(df, seq_len=30):
    """
    Generates sliding windows of sensor data and RUL labels for model training.

    Parameters:
        df (pd.DataFrame): Must include 'unit_number', 'time_in_cycles', 'RUL', and sensor columns.
        seq_len (int): Length of each time window.

    Returns:
        Tuple (X, y): 3D array of sequences and 1D array of RUL targets.
    """
    feature_cols = [col for col in df.columns if "sensor_measurement" in col]
    X, y = [], []

    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles").reset_index(drop=True)
        data = group[feature_cols].values
        target = group["RUL"].values

        if len(group) >= seq_len:
            for i in range(len(group) - seq_len + 1):
                X.append(data[i:i+seq_len])
                y.append(target[i+seq_len-1])  # Label is RUL at end of the window

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_by_unit(df, test_size=0.3, random_state=42):
    """
    Splits the DataFrame into training and validation sets based on unique engine IDs.

    Parameters:
        df (pd.DataFrame): Dataset including 'unit_number'.
        test_size (float): Fraction of units to assign to validation set.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple (train_df, val_df): Split DataFrames by engine ID.
    """
    unit_ids = df['unit_number'].unique()
    train_ids, val_ids = train_test_split(unit_ids, test_size=test_size, random_state=random_state)

    train_df = df[df['unit_number'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['unit_number'].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df

def subset_by_unit(df, unit_ids):
    """
    Returns a subset of the DataFrame containing only the specified engine units.

    Parameters:
        df (pd.DataFrame): Original dataset.
        unit_ids (array-like): List or array of engine unit numbers.

    Returns:
        pd.DataFrame: Filtered dataset for the specified units.
    """
    return df[df['unit_number'].isin(unit_ids)].reset_index(drop=True)

def make_feature_vectors_from_windows(X, strategy='last'):
    """
    Converts 3D time-series input into 2D feature vectors for use with baseline models.

    Parameters:
        X (np.ndarray): 3D input array (samples, timesteps, features)
        strategy (str): One of ['last', 'mean', 'flat']

    Returns:
        np.ndarray: 2D array of shape (samples, features)
    """
    if strategy == 'last':
        return X[:, -1, :]  # last timestep
    elif strategy == 'mean':
        return X.mean(axis=1)  # mean over time
    elif strategy == 'flat':
        return X.reshape(X.shape[0], -1)  # flatten entire window
    else:
        raise ValueError("Invalid strategy. Choose from 'last', 'mean', or 'flat'.")

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, filename='cmaps_preprocessed.npz'):
    """
    Saves preprocessed datasets to a compressed .npz file.

    Parameters:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): Data arrays
        filename (str): Output file path

    Returns:
        None
    """
    np.savez_compressed(
        filename,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"Data saved to {filename}")


def load_preprocessed_data(filename):
    """
    Loads preprocessed datasets from a compressed .npz file.

    Parameters:
        filename (str): Path to .npz file.

    Returns:
        Tuple of arrays: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data = np.load(filename, allow_pickle=True)
    return (
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test']
    )