import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_clean_data(data_path):
    """
    Load data from a CSV file and perform basic cleaning.
    
    Parameters:
        data_path (str): Path to the CSV data file.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Basic cleaning (e.g., drop rows with missing target or input features)
    df.dropna(inplace=True)
    
    return df

def preprocess_data(df):
    # Define the target and features
    X = df.drop(columns=['Outcome'])  # Assuming 'Outcome' is the target variable
    y = df['Outcome']
    
    # Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform the data
    X = preprocessor.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor
