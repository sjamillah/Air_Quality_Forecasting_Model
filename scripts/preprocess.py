import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load cleaned data
train_df = pd.read_csv('data/train_cleaned.csv', index_col='datetime', parse_dates=True)
test_df = pd.read_csv('data/test_cleaned.csv', index_col='datetime', parse_dates=True)

# Interpolate missing values using time-based method
train_df = train_df.interpolate(method='time')
test_df = test_df.interpolate(method='time')

# Fill remaining NaNs with column means
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Select features and target
if 'No' in train_df.columns:
    features = train_df.drop(columns=['pm2.5', 'No'])
else:
    features = train_df.drop(columns=['pm2.5'])
target = train_df['pm2.5']

# Define lookback period (72 hours as per latest conclusion)
lookback = 72

# Create sequences for time-series forecasting
def create_sequences(data, target, lookback):
    """
    Create sequences for time series forecasting
    Args:
        data: DataFrame with features
        target: Series with target values
        lookback: Number of timesteps to look back
    Returns:
        X: Array of sequences
        y: Array of target values
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data.iloc[i-lookback:i].values)
        y.append(target.iloc[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(features, target, lookback)

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
X_train_scaled = X_train_scaled.reshape(X_train.shape)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Prepare test data for prediction
if 'No' in test_df.columns:
    X_test_features = test_df.drop(['No'], axis=1)
else:
    X_test_features = test_df.copy()

padding = pd.DataFrame([X_test_features.iloc[0].values] * (lookback - 1), columns=X_test_features.columns)
X_test_features_padded = pd.concat([padding, X_test_features], ignore_index=True)
X_test = []
for i in range(len(X_test_features_padded) - lookback + 1):
    X_test.append(X_test_features_padded.iloc[i:i + lookback].values)
X_test = np.array(X_test)

X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler_X.transform(X_test_reshaped)
X_test_scaled = X_test_scaled.reshape(X_test.shape)
