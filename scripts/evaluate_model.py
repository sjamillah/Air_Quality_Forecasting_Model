import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

# Load preprocessed data
X_train_scaled = np.load('data/X_train_scaled.npy')
y_train_scaled = np.load('data/y_train_scaled.npy')

# Split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
)

# Define model
lookback = 72
model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(lookback, X_train_scaled.shape[2]), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(units=32, activation='relu'),
    Dropout(0.2),
    Dense(units=1)
])

# Compile model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[RootMeanSquaredError()])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train model
history = model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# Evaluate on validation set
y_pred = model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred)
y_val_inverse = scaler_y.inverse_transform(y_val.reshape(-1, 1))
val_rmse = np.sqrt(mean_squared_error(y_val_inverse, y_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Evaluate on training set
train_predictions = model.predict(X_tr)
train_predictions = scaler_y.inverse_transform(train_predictions)
train_loss = np.mean((scaler_y.inverse_transform(y_tr.reshape(-1, 1)).flatten() - train_predictions.flatten())**2)
train_rmse = np.sqrt(train_loss)
print(f"Training MSE: {train_loss:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
