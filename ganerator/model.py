import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RiskModel:
    def __init__(self, data, target_column, categorical_columns, model_type):
        self.df = data
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.model = None
        self.df_encoded = None
        self.model_type = model_type

    def encode_data(self, df):
        return pd.get_dummies(df, columns=self.categorical_columns)

    def preprocess_data(self):
        # Encoding categorical variables
        self.df_encoded = self.encode_data(self.df)

        # Splitting the data into train and test sets
        X = self.df_encoded.drop(columns=[self.target_column])
        y = self.df_encoded[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        self.preprocess_data()
        # Training a Random Forest model
        if self.model_type == "logistic":
            self.model = LogisticRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise Exception("Select a valid model: 'logistic' or 'random_forest'.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Making predictions
        y_pred = self.model.predict(self.X_test)

        # Evaluating the model
        self.accuracy = accuracy_score(self.y_test, y_pred)
        return self.accuracy

    def predict(self, new_df):
        new_df_encoded = self.encode_data(new_df)

        missing_columns = set(self.df_encoded.columns) - set(new_df_encoded.columns)
        for column in missing_columns:
            if column != self.target_column:
                new_df_encoded[column] = 0

        column_order = self.df_encoded.columns.tolist()
        column_order.remove("TARGET")

        predictions = self.model.predict(new_df_encoded[column_order])
        return predictions
