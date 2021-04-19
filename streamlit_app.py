import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache
def load_data(url):
    return pd.read_csv(url)


def fav_filter(dataframe):
    """Return a data filtered"""
    return dataframe[(dataframe['Votes'] > min_pop) & (dataframe['Votes'] < max_pop)
                     & (dataframe['Note'] > min_rate) & (dataframe['Note'] < max_rate)]


#############
## sidebar ##
#############

st.sidebar.title('Navigation')

categorie = st.sidebar.radio("Categorie", ('Analyse comparative', 'Corrélation', 'Retrospective', 'Machin Learning'))

expander = st.sidebar.beta_expander("Sources")
expander.markdown(
    """
    [Base de donnée imdb](https://www.imdb.com/interfaces/) : 
    N'ont été retenus que les films ayant plus de 1000 votes.
    
    [Base de donnée Netflix](https://en.wikipedia.org/wiki/Lists_of_Netflix_original_films) : 
    Les films exclusifs à la plaforme Netflix ont été retirés.
    """)


##########
## DATA ##
##########

# modifier selon la localisation de la BD
REPRO_DB = 'https://raw.githubusercontent.com/MickaelKohler/imbd_recommandation/main/repro.csv'
FR_MOV_DB = 'https://raw.githubusercontent.com/MickaelKohler/imbd_recommandation/main/fr_mov.csv'


###############
## MAIN PAGE ##
###############

col1, col2, col3, col4, col5, col6, col7 = st.beta_columns(7)
if col1.button('Précédent'):
    categorie = 'Machin Learning'
if col7.button('Suivant'):
    categorie = 'Corrélation'


if categorie == 'Analyse comparative':
    st.title('Analyse comparative')

    col1, col2 = st.beta_columns(2)
    with col1: st.markdown(
        """
        La *courbe bleue* montre différentes caractéristiques des films selon les décénnies.
        Elles sont extraites de la base de données du site **imbd**.
        """)
    with col2: st.markdown(
        """
        La *courbe orange* montre les mêmes indices selon le filtre selectionné.
        Il suffit de modifier les curseurs pour modifier la plage de notes et de votes appliquée.
        Par défaut, on retient les films ayant une **note supérieure à 7 pour plus de 1500 votes**.
        """)

    # data
    data = load_data(FR_MOV_DB)
    data_crew = load_data(REPRO_DB)

    # filtres
    filter_expand = st.beta_expander('Filtres')
    with filter_expand:
        rating_filter = st.slider('Selectionnez une plage de notes', 0, 10, (7, 10))
        tick_max = data['Votes'].max().item()
        tick_min = data['Votes'].min().item()
        pop_filter = st.slider('Selectionnez le nombre de votes', tick_min, tick_max, (1500, tick_max))
        min_rate, max_rate = zip(rating_filter)
        min_pop, max_pop = zip(pop_filter)
        show = st.checkbox('Montre moi la data')

    # Runtime
    temp_tab = data[data['Durée'] != '\\N'][['Année', 'Durée', 'Note', 'Votes']]
    temp_tab['Durée'] = pd.to_numeric(temp_tab['Durée'])
    temp_tab = temp_tab[temp_tab['Durée'] < 500]
    mov_runtime = temp_tab.groupby('Année').mean()[['Durée']].loc['1920':'2020']
    best_mov = temp_tab[(temp_tab['Votes'] > min_pop) & (temp_tab['Votes'] < max_pop)
                        & (temp_tab['Note'] > min_rate) & (temp_tab['Note'] < max_rate)]
    mov_runtime['Filtrés'] = best_mov.groupby('Année').mean()[['Durée']]
    mov_runtime.fillna(mov_runtime['Filtrés'].median(), inplace=True)

    fig = px.line(mov_runtime, x=mov_runtime.index, y=["Durée", "Filtrés"],
                  title='<b>Durée moyenne</b> (en mintues)',
                  color_discrete_map={
                     'Durée': 'steelblue',
                     'Filtrés': 'darkorange'})
    fig.update_yaxes(title=None, tick0=True, nticks=12)
    fig.update_xaxes(title=None, nticks=12)
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans')
    st.plotly_chart(fig, use_container_width=True)
    if show:
        mov_runtime

    # Age
    age = data_crew[(data_crew['category'].isin(['actor', 'actress'])) & (data_crew['Naissance'] != '\\N')]
    age['Naissance'] = pd.to_numeric(age['Naissance'])
    age['Age'] = age['Année'] - age['Naissance']
    age_chart = age.groupby((age['Année']//10)*10).median()[['Age']].loc[1920:2020]
    best_age = age[(age['Votes'] > min_pop) & (age['Votes'] < max_pop)
                   & (age['Note'] > min_rate) & (age['Note'] < max_rate)]
    age_chart['Filtrés'] = best_age.groupby((best_age['Année']//10)*10).median()[['Age']]

    fig = px.bar(age_chart, x=age_chart.index, y=["Age", "Filtrés"], barmode='group',
                 title='<b>Age moyen des acteurs</b> (en années)',
                 color_discrete_map={
                     'Age': 'steelblue',
                     'Filtrés': 'darkorange'})
    fig.update_yaxes(title=None, nticks=10)
    fig.update_xaxes(title=None, nticks=12)
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans')
    st.plotly_chart(fig, use_container_width=True)
    if show:
        age_chart

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

    # filtre tables
    filtre = st.selectbox(
        'Vous souhaitez filtrer par :',
        ['Votes', 'Total', 'Note'])

    filtered_data = data[(data['Année'] >= start_year) & (data['Année'] < stop_year)]

    # diplay the data
    st.subheader('Top 10 des films')
    temp_tab = filtered_data[['Titre', 'Année', 'Genres', 'Note', 'Votes']]
    top_film = temp_tab.groupby(['Titre', 'Année']).max().sort_values(['Note', 'Votes'], ascending=False)
    top_film.reset_index(inplace=True)
    top_film.index = top_film.index + 1
    st.table(top_film.iloc[0:10])

    st.subheader('Top 10 des réalisateurs')
    temp_tab = filtered_data[filtered_data['category'] == 'director'][['Nom', 'Naissance', 'Décès', 'Total', 'Note', 'Votes']]
    top_real = temp_tab.groupby(['Nom', 'Naissance']).agg({'Note': 'mean', 'Votes': 'sum', 'Total': 'count'}).sort_values(filtre, ascending=False)
    top_real.reset_index(inplace=True)
    top_real.index = top_real.index + 1
    st.table(top_real.iloc[0:10])

    st.subheader('Top 10 des acteurs')
    temp_tab = filtered_data[filtered_data['category'].isin(['actor', 'actress'])][['Nom', 'Naissance', 'Décès', 'Total', 'Note', 'Votes']]
    top_act = temp_tab.groupby(['Nom', 'Naissance']).agg({'Note': 'mean', 'Votes': 'sum', 'Total': 'count'}).sort_values(filtre, ascending=False)
    top_act.reset_index(inplace=True)
    top_act.index = top_act.index + 1
    st.table(top_act.iloc[0:10])

elif categorie == 'Machin Learning':
    st.title('Machin Learning')
    st.write('## Work in progress')
    st.image('https://assets.moncoachdata.com/v7/moncoachdata.com/wp-content/uploads/2019/07/machine_learning.png?func=crop&w=880&h=360', width=700)

# lien tuto streamlit
# https://docs.streamlit.io/en/stable/getting_started.html

# Galerie 
# https://streamlit.io/gallery?type=apps&category=data-visualization

# Doc
# https://awesome-streamlit.org
