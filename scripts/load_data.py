import pandas as pd

# Load train and test data
train_df = pd.read_csv('data/train.csv', header=0)
test_df = pd.read_csv('data/test.csv', header=0)

# Clean column names by removing whitespace
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Convert 'datetime' column to datetime and set as index
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
train_df.set_index('datetime', inplace=True)
test_df.set_index('datetime', inplace=True)

# Save cleaned data for use in other scripts
train_df.to_csv('data/train_cleaned.csv')
test_df.to_csv('data/test_cleaned.csv')
