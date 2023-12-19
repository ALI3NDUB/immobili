import streamlit as st

st.set_page_config(
    page_title="ML&AI ANDREA",
    page_icon="👋",
)

percorso_locale_immagine = "/Users/user/Desktop/demo/01.png"
st.image(percorso_locale_immagine, use_column_width=True)

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

st.write("## Welcome & About 👋")

st.sidebar.success("Seleziona un Modello")

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #0000001C;
    }
</style>
""", unsafe_allow_html=True)
          
st.markdown(
    """
    Ho utilizzato Streamlit per costruire questa "pagina-raccoglitore" con 
    i miei migliori progetti del corso.
    
    Questo pacchetto con annesso FrameWork permette la creazione di 
    WebApp basate su Python, grazie all'integrazione con Github, 
    in maniera specifica di Machine Learning e Data Science.  
    Ormai divenuto uno standard, è stato creato con l'idea di semplificare lo sviluppo e il deploy della app, 
    senza avre necessariamente molte conoscenze di html5, CSS e JavaScript.  
    L'integrazione con Pandas e i pacchetti di visualizzazione lo hanno reso ormai uno standard, 
    grazie anche a features come la possibilita di aggiornare in tempo reale 
    l'interfaccia utente rispetto al codice e le librerie oggetti integrate che permettono 
    la visualizzazione interattive dei grafici.
    
    **👈 Selezionate una progetto dalla barra laterale per vederne le potenzialita** 
    
    
    
    #### Creato da Andrea Giorgio con [GitHub](https://github.com/ALI3NDUB) e [Streamlit](https://streamlit.io)
        
  
"""
)

percorso_locale_immagine = "/Users/user/Desktop/demo/ifoa.png"
st.image(percorso_locale_immagine, use_column_width=False, width = 200)

