import streamlit as st
import pandas as pd
import pickle
import data_processing as dp
from model import metrics
from io import StringIO

st.title('Bank Marketing')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    df = dp.ingest(uploaded_file)
    df = dp.ordinal_encoding(df, 'y', ['no', 'yes'])
    df = dp.nominal_encoding(df)
    df = dp.outlier_detection(df)
    X, y = dp.dim_reduce(df)
    with open('model_pickle', 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    acc, prec, rec, f1 = metrics(y, y_pred)
    st.write('Accuracy: ', acc)
    st.write('Precision: ', prec)
    st.write('Recall: ', rec)
    st.write('F1 Score: ', f1)

    
    

    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)

