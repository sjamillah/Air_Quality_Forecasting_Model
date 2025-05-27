import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Generate predictions
predictions_scaled = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)
predictions = np.nan_to_num(predictions)
predictions = np.round(predictions).astype(int)

# Create submission DataFrame
submission_indices = test_df.index[:len(predictions)]
submission = pd.DataFrame({
    'row ID': pd.to_datetime(submission_indices).strftime('%Y-%m-%d %-H:%M:%S'),
    'pm2.5': predictions.flatten()
})
submission = submission.sort_values(by='row ID')

# Save submission
submission.to_csv('/content/drive/MyDrive/Kaggle_competition_ML/air_quality_forcasting/subm_fixed.csv', index=False)
