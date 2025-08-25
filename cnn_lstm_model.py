# cnn_lstm_model.py

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, LSTM, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from pre_processing import load_preprocessed_data
from evaluator import evaluate_model


def build_cnn_lstm_model(input_shape):
    """
    Build and compile a CNN→LSTM hybrid for RUL prediction.

    Args:
        input_shape (tuple): (timesteps, features), e.g. (30, 15).

    Returns:
        tf.keras.Model: compiled model ready for training.
    """
    inputs = Input(shape=input_shape)

    # Conv feature extraction
    x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # LSTM temporal modeling
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # Dense regression head
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Hybrid')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_cnn_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=40,
    batch_size=64,
    patience=4,
    verbose=1
):
    """
    Train the CNN–LSTM hybrid with early stopping.

    Args:
        model (tf.keras.Model): compiled model.
        X_train (np.ndarray): train inputs.
        y_train (np.ndarray): train targets.
        X_val, y_val: validation splits.
        epochs (int): max epochs.
        batch_size (int): batch size.
        patience (int): early-stopping patience on val_loss.
        verbose (int): verbosity level.

    Returns:
        tf.keras.callbacks.History: training history.
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    return history


def predict_cnn_lstm_model(model, X):
    """
    Run inference and return 1D RUL predictions.

    Args:
        model (tf.keras.Model): trained model.
        X (np.ndarray): input windows.

    Returns:
        np.ndarray: shape (n_samples,) predicted RUL.
    """
    preds = model.predict(X, verbose=0)
    return preds.flatten()


def save_cnn_lstm_model(model, filename):
    """
    Save the hybrid model to disk.

    Args:
        model (tf.keras.Model): model to save.
        filename (str): path, e.g. 'cnn_lstm.fd001.keras' or '.h5'.
    """
    model.save(filename)


def load_cnn_lstm_model(filename):
    """
    Load a previously saved hybrid model.

    Args:
        filename (str): path where model was saved.

    Returns:
        tf.keras.Model: reconstructed model.
    """
    return load_model(filename)

# final model is not working fully keep this in mind when working on main and come back and fix
def run_cnn_lstm_pipeline(
    npz_path="fd001_last.npz",
    epochs=40,
    batch_size=64,
    patience=4,
    verbose=1,
    model_name="CNN-LSTM"
):
    """
    Full CNN→LSTM pipeline:
      1. Load preprocessed data (.npz)
      2. Build & train the hybrid model
      3. Evaluate on validation set
      4. Attempt evaluate on test set (safely)

    Returns:
        model: the trained tf.keras.Model
        val_metrics: dict of RMSE & MAE on validation
        test_metrics: dict of RMSE & MAE on test (or {} if skipped)
    """
    # 1) Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(npz_path)

    # 2) Build & compile
    model = build_cnn_lstm_model(input_shape=X_train.shape[1:])

    # 3) Train
    train_cnn_lstm_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose
    )

    # 4) Validation evaluation
    y_val_pred = predict_cnn_lstm_model(model, X_val)
    val_metrics = evaluate_model(y_val, y_val_pred, model_name + " (val)")

    # 5) Test evaluation (wrapped to never break)
    test_metrics = {}
    try:
        if (
            isinstance(X_test, np.ndarray) and isinstance(y_test, np.ndarray)
            and X_test.size > 0 and y_test.size > 0
        ):
            y_test_pred = predict_cnn_lstm_model(model, X_test)
            test_metrics = evaluate_model(y_test, y_test_pred, model_name + " (test)")
        else:
            print("⚠️ No valid test set found – skipping test evaluation.")
    except Exception as e:
        print(f"⚠️ Skipped test evaluation due to error: {e}")

    return model, val_metrics, test_metrics
