import numpy as np
import pandas as pd
import streamlit as st


def is_binary(column_data):
    return np.issubdtype(column_data.dtype, np.integer) and column_data.isin([0, 1]).all()


def is_numerical(column_data):
    return np.issubdtype(column_data.dtype, np.number)


def is_categorical(column_data):
    return isinstance(column_data.dtype, pd.CategoricalDtype) or len(
        column_data.unique()
    ) <= 0.5 * len(column_data)


def classify_data_types(df):
    data_types = {}

    for col in df.columns:
        column_data = df[col]

        if is_binary(column_data):
            data_info = column_data.unique().tolist()
            data_types[col] = {"dtype": "binary", "data_info": data_info}
        elif is_numerical(column_data):
            min_number = int(column_data.min())
            max_number = int(column_data.max())
            data_types[col] = {
                "dtype": "numerical",
                "data_info": [min_number, max_number],
            }
        elif column_data.dtype == "object":
            if is_categorical(column_data):
                categorie_count = len(column_data.unique().tolist())
                data_types[col] = {
                    "dtype": "categorical",
                    "data_info": categorie_count,
                }
            else:
                data_types[col] = {"dtype": "text", "data_info": np.nan}
        else:
            data_types[col] = {"dtype": "other", "data_info": np.nan}

    return data_types


def main():

    st.title("Model Trainer")

    tab1, tab2 = st.tabs(["Data", "Prediction"])
    target_col = False

    with st.sidebar:
        file = st.file_uploader("File uploader")
        if file:
            my_dataframe = pd.read_csv(file)
            target_col = st.selectbox("Select Target Column", my_dataframe.columns.tolist())

    with tab1:
        if file:
            st.dataframe(my_dataframe)
            data_types = classify_data_types(my_dataframe)
            data_types = pd.DataFrame(data_types).T
            st.dataframe(data_types)

            coded = st.radio("", ["Encoded", "Decoded"])

            categorial_cols = data_types.query("dtype=='categorical'").index.tolist()
            encoded_dataframe = pd.get_dummies(
                my_dataframe[categorial_cols], categorial_cols, prefix_sep="__"
            )

            if coded == "Encoded":
                st.dataframe(encoded_dataframe)
            else:
                decoded_dataframe = pd.from_dummies(encoded_dataframe, sep="__")
                st.dataframe(decoded_dataframe)

    with tab2:
        if target_col:
            st.button(f"Train model to predict column '{target_col}'")
        else:
            st.text("Upload data first to unlock this section.")


if __name__ == "__main__":
    main()
