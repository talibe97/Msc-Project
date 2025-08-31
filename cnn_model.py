import os
os.environ.setdefault("KERAS_BACKEND", "torch")
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
from pre_processing import load_preprocessed_data          # already in repo
from evaluator import evaluate_model  



# ---------- 1. Build ----------
def build_cnn_model(input_shape):
    """
    Returns a 1-D temporal CNN for RUL prediction.
    Architecture:  2×(Conv1D + MaxPool) → GAP → Dense → output
    """
    model = Sequential([
        Conv1D(64,  kernel_size=5, activation='relu', padding='same',
               input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------- 2. Train ----------
def train_cnn_model(model, X_train, y_train, X_val, y_val,
                    epochs=25, batch_size=64):
    """
    Fits the CNN, using EarlyStopping on val_loss.
    """
    es = EarlyStopping(monitor='val_loss', patience=10,
                       restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return model, history

# ---------- 3. Predict ----------
def predict_cnn_model(model, X):
    """Vectorised CNN inference."""
    return model.predict(X).flatten()



def run_cnn_pipeline(npz_path="fd001_last.npz",
                     epochs=25, batch_size=64):
    """
    Complete workflow: load → build → train → predict → evaluate.
    """
    # 1. data
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(npz_path)
    print(f"Train: {X_train.shape}   Val: {X_val.shape}")

    # 2. model
    model = build_cnn_model(X_train.shape[1:])
    model.summary()

    # 3. train
    model, hist = train_cnn_model(model, X_train, y_train,
                                  X_val, y_val,
                                  epochs=epochs, batch_size=batch_size)

    # 4. predict + evaluate
    y_pred = predict_cnn_model(model, X_val)
    metrics = evaluate_model(y_val, y_pred, model_name="CNN")

    return model, y_val, y_pred, metrics, hist

# ---------- 5. Save / Load ----------
def save_cnn_model(model, filename="cnn_model.h5"):
    model.save(filename)
    print(f"Saved → {filename}")

def load_cnn_model(filename="cnn_model.h5"):
    from keras.models import load_model
    model = load_model(filename)
    print(f"Loaded ← {filename}")
    return model
