import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


    #############
    ## sidebar ##
    #############

st.sidebar.title('Navigation')

categorie = st.sidebar.radio("Categorie",('Analyse comparative','Corrélation', 'Retrospective', 'Machin Learning'))

expander = st.sidebar.beta_expander("Sources")
expander.markdown(
"""
[Base de donnée imdb](https://www.imdb.com/interfaces/) : 
N'ont été retenus que les films ayant plus de 1000 votes.

[Base de donnée Netflix](https://en.wikipedia.org/wiki/Lists_of_Netflix_original_films) : 
Les films exclusifs à la plaforme Netflix ont été retirés.
"""
)


    ##########
    ## DATA ##
    ##########

REPRO_DB = ('https://raw.githubusercontent.com/MickaelKohler/imbd_recommandation/main/repro.csv') # modifier selon la localisation de la BD 
FR_MOV_DB = ('https://raw.githubusercontent.com/MickaelKohler/imbd_recommandation/main/fr_mov.csv') # modifier selon la localisation de la BD 

@st.cache
def load_data(url):
    return pd.read_csv(url)

    ###############
    ## MAIN PAGE ##
    ###############

if categorie == 'Analyse comparative':
    st.title('Analyse comparative')

    #data
    data = load_data(FR_MOV_DB)

    st.write('Comparaison de différents indices, en fonction des filtres selectionnés. \n ___')
    
    rating_filter = st.slider('Selectionner une plage de notes', 0, 10, (7, 10))
    tick_max = data['Votes'].max().item()
    tick_min = data['Votes'].min().item()
    pop_filter = st.slider('Selectionner une plage de nombre de votes', tick_min, tick_max, (1500, tick_max))
    min_rate, max_rate = zip(rating_filter)
    min_pop, max_pop = zip(pop_filter)
    st.write('___')

    #Analyse de la durée
    temp_tab = data[data['Durée'] != '\\N'][['Année','Durée', 'Note', 'Votes']]
    temp_tab['Durée'] = pd.to_numeric(temp_tab['Durée'])
    temp_tab = temp_tab[temp_tab['Durée'] < 500]
    mov_runtime = temp_tab.groupby('Année').mean()[['Durée']]
    best_mov = temp_tab[(temp_tab['Votes'] > min_pop) & (temp_tab['Note'] > min_rate)]
    mov_runtime['Filtrés'] = best_mov.groupby('Année').mean()[['Durée']]
    st.line_chart(mov_runtime.loc[1920:2020])

    #charts
    #fig, ax = plt.subplots()
    #plt.title('Répartition des notes sur la période sélectionnée')
    #ax.hist(data['Note'], bins=10, color='orangered')
    #st.pyplot(fig)


elif categorie == 'Corrélation':
    st.title('Corrélation')
    st.markdown("""
    ## Work in progress
    La progression du travail a une forte corrélation avec votre temps de travail.
    """)

elif categorie == 'Retrospective':    
    st.title('Retrospective par décennie')

    st.write("""
    Classement selon les données présentes sur la platforme **imbd**
    """)
    
    data = load_data(REPRO_DB)

    # create a filter
    start_year = st.slider('Décennie', 1920, 2020, 1990, 10)
    stop_year = start_year+10

    # display best all times
    if st.button('Toutes les années confondues'):
        start_year = 1920
        stop_year = 2020

    #filtre tables
    filtre = st.selectbox(
    'Vous souhaitez filtrer par :',
    ['Votes', 'Total', 'Note'])

    filtered_data = data[(data['Année'] >= start_year) & (data['Année'] < stop_year)]

    #diplay the data
    st.subheader('Top 10 des films')
    temp_tab = filtered_data[['Titre', 'Année', 'Genres', 'Note', 'Votes']]
    top_film = temp_tab.groupby(['Titre', 'Année']).max().sort_values(['Note', 'Votes'], ascending=False)
    top_film.reset_index(inplace=True)
    top_film.index = top_film.index + 1
    st.table(top_film.iloc[0:10])

    st.subheader('Top 10 des réalisateurs')
    temp_tab = filtered_data[filtered_data['category']=='director'][['Nom', 'Naissance', 'Décès', 'Total', 'Note', 'Votes']]
    top_real = temp_tab.groupby(['Nom', 'Naissance']).agg({'Note':'mean', 'Votes':'sum', 'Total':'count'}).sort_values(filtre, ascending=False)
    top_real.reset_index(inplace=True)
    top_real.index = top_real.index + 1
    st.table(top_real.iloc[0:10])

    st.subheader('Top 10 des acteurs')
    temp_tab = filtered_data[filtered_data['category'].isin(['actor', 'actress'])][['Nom', 'Naissance', 'Décès', 'Total', 'Note', 'Votes']]
    top_act = temp_tab.groupby(['Nom', 'Naissance']).agg({'Note':'mean', 'Votes':'sum', 'Total':'count'}).sort_values(filtre, ascending=False)
    top_act.reset_index(inplace=True)
    top_act.index = top_act.index + 1
    st.table(top_act.iloc[0:10])

elif categorie == 'Machin Learning':
    st.title('Machin Learning')
    st.write('## Work in progress')
    st.image('https://assets.moncoachdata.com/v7/moncoachdata.com/wp-content/uploads/2019/07/machine_learning.png?func=crop&w=880&h=360', width=700)

# lien tuto streamlit
# https://docs.streamlit.io/en/stable/getting_started.html