import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ingest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    return df

def ordinal_encoding(df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
    encoder = OrdinalEncoder(categories=[categories], dtype=int)
    df['target_encoded'] = encoder.fit_transform(df[['y']])
    df.drop(column, axis=1, inplace=True)
    return df

def nominal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    nominal_columns = df.select_dtypes(include=['object']).columns
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    a = ohe.fit_transform(df[nominal_columns])
    a = pd.DataFrame(a, columns=ohe.get_feature_names_out(nominal_columns))
    df = df.drop(nominal_columns, axis=1)
    df = pd.concat([df, a], axis=1)
    return df


def outlier_detection(df):
    X = df.drop('target_encoded', axis=1)
    isof = IsolationForest(n_estimators=100, contamination=0.1)
    isof.fit(X)
    outliers = isof.predict(X)
    df['is_inliner'] = outliers
    df = df[df['is_inliner'] == 1]
    df.drop('is_inliner', axis=1, inplace=True)
    return df

def dim_reduce(df):
    X = df.drop('target_encoded', axis=1)
    y = df['target_encoded']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled
    pca = PCA(n_components=0.85)
    X_pca = pca.fit_transform(X_scaled)
    df = X
    return X_pca, y

def split_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_test, y_train, y_test

def split_data2(X, y) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_test, y_train, y_test
