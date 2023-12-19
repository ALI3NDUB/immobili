
import numpy as np
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#Import warnings for sklearn
import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.londonart.it/public/images/2020/restanti-varianti/20043-01.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.markdown("""
    <style>
        [data-testid=stSidebar] {
        background-color: #0000001C;
    }
        </style>
        """, 
        unsafe_allow_html=True
    )

st.title('NLP Sentiment Classification')

uploaded_model = joblib.load('NLPEs2.pkl')

add_bg_from_url()

def user_input_features():
    user_input_text = st.text_input("Enter your text here")
    return user_input_text

df_user_input = user_input_features()

st.subheader('Input')
st.write(df_user_input)

prediction = uploaded_model.predict([df_user_input])

st.subheader('Classification')
st.write(prediction[0])



