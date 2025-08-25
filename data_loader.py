import pandas as pd

def load_raw_data(file_path):
    """Load C-MAPSS FD001 data with appropriate column names."""
    # Columns from dataset documentation (1 engine id + 1 cycle + 3 settings + 21 sensors)
    col_names = ['unit_number', 'time_in_cycles'] + \
                [f'op_setting_{i+1}' for i in range(3)] + \
                [f'sensor_measurement_{i+1}' for i in range(21)]

    df = pd.read_csv(file_path, sep='\\s+', header=None, names=col_names)
    return df


def inspect_data(df):
    """Print high-level info about the dataset."""
    print("Shape:", df.shape)
    print("\nUnique engines:", df['unit_number'].nunique())
    print("\nMissing values:\n", df.isnull().sum().sum())
    print("\nMax cycles per engine:")
    print(df.groupby('unit_number')['time_in_cycles'].max().describe())
    print("\nFirst 5 rows:")
    display(df.head())
    
def load_test_data(test_path, rul_path):
    """Load and return test set and actual RUL labels."""
    # Same column names as training set
    col_names = ['unit_number', 'time_in_cycles'] + \
                [f'op_setting_{i+1}' for i in range(3)] + \
                [f'sensor_measurement_{i+1}' for i in range(21)]

    test_df = pd.read_csv(test_path, sep='\\s+', header=None, names=col_names)
    rul_df = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])
    
    return test_df, rul_df
def get_last_cycles(df):
    """Get the last time step for each engine unit."""
    return df.groupby('unit_number').last().reset_index()

def save_clean_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")