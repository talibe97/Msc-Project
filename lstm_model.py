# functions


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from pre_processing import load_preprocessed_data
from evaluator import evaluate_model


def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for RUL prediction.

    Parameters:
        input_shape (tuple): Shape of input data (timesteps, features)

    Returns:
        model (keras.Model): Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    """
    Trains the LSTM model with training and validation data.

    Parameters:
        model (keras.Model): Compiled LSTM model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs (int): Number of epochs
        batch_size (int): Batch size

    Returns:
        model: Trained Keras model
        history: Training history object
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history

def predict_lstm_model(model, X):
    """
    Generates RUL predictions from the trained LSTM model.

    Parameters:
        model (keras.Model): Trained LSTM model
        X (np.ndarray): Input data to predict on

    Returns:
        np.ndarray: Predicted RUL values
    """
    return model.predict(X).flatten()

def run_lstm_pipeline(npz_path="fd001_last.npz", epochs=20, batch_size=64):
    """
    Full pipeline to train, evaluate, and report LSTM model performance.

    Parameters:
        npz_path (str): Path to .npz file with preprocessed data
        epochs (int): Number of training epochs
        batch_size (int): Size of each training batch

    Returns:
        Tuple: model, y_val, y_pred, evaluation_results
    """
    print("=== Step 1: Load Preprocessed Data ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(npz_path)
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape  : {X_val.shape}, {y_val.shape}")
    print("")

    print("=== Step 2: Build Model ===")
    input_shape = X_train.shape[1:]
    model = build_lstm_model(input_shape)
    model.summary()
    print("")

    print("=== Step 3: Train Model ===")
    model, history = train_lstm_model(model, X_train, y_train, X_val, y_val,
                                      epochs=epochs, batch_size=batch_size)
    print("Training complete.\n")

    print("=== Step 4: Predict ===")
    y_pred = predict_lstm_model(model, X_val)
    print("Sample predictions:", y_pred[:5])
    print("")

    print("=== Step 5: Evaluate ===")
    evaluation_results = evaluate_model(y_val, y_pred, model_name="LSTM")
    print("")

    return model, y_val, y_pred, evaluation_results

def save_lstm_model(model, filename="lstm_model.h5"):
    """
    Saves the trained LSTM model to an HDF5 (.h5) file.

    Parameters:
        model (keras.Model): Trained model to save
        filename (str): Filename for saving
    """
    model.save(filename)
    print(f"Model saved to {filename}")


def load_lstm_model(filename="lstm_model.h5"):
    """
    Loads a saved LSTM model from an HDF5 (.h5) file.

    Parameters:
        filename (str): Filename of the saved model

    Returns:
        keras.Model: Loaded model
    """
    from keras.models import load_model
    model = load_model(filename)
    print(f"Model loaded from {filename}")
    return model