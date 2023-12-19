import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px
import joblib
import io

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

def main(): 
    new_model = joblib.load('algo.pkl')
      
    add_bg_from_url()
 
    st.title('Data Explore Immobiliare')
    st.markdown (
        """ 
        Tramite un algoritmo di Machine Learning ho addestrato una semplice AI che fa predizioni immobiliari sul prezzo, 
        plottando i relativi grafici che poi possono esser anche scaricati in locale.
        \n
        Tramite il Drag&Drop è possibile caricare dei file CSV precedentemente creati, è importante che tutti i file 
        abbiano la stessa struttura e nome colonne.
        """)
    
    dropzone = st.file_uploader("Caricare il file CSV", type=["csv"], key="fileUploader")
    if dropzone is not None:
        df = pd.read_csv(dropzone)
        st.markdown(""" ### Il programma mostra il file CSV, anche full screen, e ci permette di ordinare le colonne in modo ascendente e discendente.""")
        st.dataframe(df)

        st.markdown(""" ### Avviene un 'Describe' che mostra valori utili allo studio statistico, in maniera rapida e istantanea""")
        dfdesc = df.iloc[1:]
        dfdesc = dfdesc.astype(float)
        descrittore = dfdesc.describe()
        st.dataframe(descrittore)

        y = df["medv"]
        X = df.drop(columns="medv")

        st.subheader('Gli slider permettono di variare dei valori della tabella, su cui è stata addestro il modello permettendo di desumerne un quarto che sarebbe il prezzo.')
        
        sl1 = st.slider('lstat', min_value=0, max_value=50, value=15)
        sl2 = st.slider('rm', min_value=0, max_value=50, value=5)
        sl3 = st.slider('ptratio', min_value=0, max_value=50, value=16)

        input_data = {'lstat': sl1,
                    'rm': sl2,
                    'ptratio': sl3}

        input_df = pd.DataFrame(input_data, index=[0])
        prediction = new_model.predict([[sl1,sl2,sl3]])
        
        st.subheader (
        """ 
        Questo plot mostra dove il nostro modello si trova rispetto al resto degli immobili, permettendo piu facilmente di comprenderne i risultati, che possono essere anche scaricati in locale.


        """)
        
        fig = px.scatter_3d(df, x='lstat', y='rm', z='ptratio', color_discrete_sequence=['#ffa500'])
        fig.add_trace(px.scatter_3d(input_df, x='lstat', y='rm', z='ptratio', color_discrete_sequence=['#FF1493']).data[0])
        fig.update_layout(
            title="",
            scene=dict(
                xaxis_title='lstat',
                yaxis_title='rm',
                zaxis_title='ptradio'))
        
        fig.update_layout(scene=dict(bgcolor='teal')) 
        st.plotly_chart(fig, use_container_width=True)

        
        if st.button('Scarica Plot come PNG', key='download_plot_button'):
            output = io.BytesIO()
            fig.write_image(output, format='png')
            output.seek(0)
            st.download_button(label='Download', data=output.getvalue(), file_name='plot.png', mime='image/png')

        st.subheader('Risultato numerico del modello, rappresenta il prezzo degli immobili. ')
        
        custom_predictions = new_model.predict([[sl1, sl2, sl3]])[0]
        st.write(custom_predictions)

        if st.button('Scarica output'):
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            pd.DataFrame([custom_predictions], columns=['Predict']).to_excel(writer, index=False, sheet_name='Output')  
            writer.save()
            output.seek(0)
            st.download_button(label="Download", data=output, file_name='out.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                               
        
        df = df.drop(index=df.index[-1])
        df = df.astype(float)
        df= df.assign(Prezzo=df.medv.apply(np.log))
        
        fig = px.scatter_3d(df, x='lstat', y='rm', z='ptratio', color='Prezzo',color_continuous_scale='reds', title='Stima Prezzo')       
        fig.update_layout(
                scene=dict(
                xaxis_title='lstat',
                yaxis_title='rm',
                zaxis_title='ptratio'))
        st.plotly_chart(fig, use_container_width=True)        
        
        if st.button('Scarica Plot come PNG', key='download_plot_button2'):
            output = io.BytesIO()
            fig.write_image(output, format='png')
            output.seek(0)
            st.download_button(label='Download', data=output.getvalue(), file_name='plot.png', mime='image/png')

            
    else:
         st.warning("Carica un file CSV per iniziare l'analisi.")   

if __name__ == '__main__':
    main()
    
# streamlit run app.py        