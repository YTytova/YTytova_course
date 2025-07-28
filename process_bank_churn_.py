import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess_data(raw_df, scaler_numeric: bool = True):
    RANDOM_STATE = 42

    raw_df = raw_df.drop(columns=['id', 'CustomerId', 'Surname'])
    # Use stratify to keep the same distribution of target values in train and val sets
    train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=RANDOM_STATE, stratify=raw_df['Exited'])

    input_cols = list(train_df.columns)[:-1]

    target_col = 'Exited'
    train_inputs, train_targets = train_df[input_cols], train_df[target_col]
    val_inputs, val_targets = val_df[input_cols], val_df[target_col]

    binary_cols = ['IsActiveMember', 'HasCrCard']  # not yet used, but can be for improvement
    numeric_cols = train_inputs.drop(columns=binary_cols).select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.drop(columns=binary_cols).select_dtypes(include='object').columns.tolist()
    print(f'Numeric columns: {numeric_cols}')
    print(f'Binary columns: {binary_cols}')
    print(f'Categorical columns: {categorical_cols}')

    # One hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    train_inputs = train_inputs.drop(columns=categorical_cols)
    val_inputs = val_inputs.drop(columns=categorical_cols)

    scaler = None
    if scaler_numeric:
        # Min max scaling
        scaler = MinMaxScaler()
        train_inputs[numeric_cols] = scaler.fit_transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])


    return {
        'X_train': train_inputs,
        'train_targets': train_targets,
        'X_val': val_inputs,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(test_raw_df, scaler, encoder, scaler_numeric=False):
    test_df = test_raw_df.drop(columns=['id', 'CustomerId', 'Surname'])

    binary_cols = ['IsActiveMember', 'HasCrCard']  # not yet used, but can be for improvement
    numeric_cols = test_df.drop(columns=binary_cols).select_dtypes(include=np.number).columns.tolist()
    categorical_cols = test_df.drop(columns=binary_cols).select_dtypes(include='object').columns.tolist()
    print(f'Numeric columns: {numeric_cols}')
    print(f'Binary columns: {binary_cols}')
    print(f'Categorical columns: {categorical_cols}')

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    test_df[encoded_cols] = encoder.transform(test_df[categorical_cols])
    test_df = test_df.drop(columns=categorical_cols)

    if scaler_numeric:
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return test_df