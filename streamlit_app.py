import math
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

import urllib.request
from gazpacho import Soup


@st.cache
def load_data(url):
    data = pd.read_csv(url)
    data.rename(columns={'title': 'Titre', 'startYear': 'Année', 'genres': 'Genres', 'averageRating': 'Note',
                         'numVotes': 'Votes', 'primaryName': 'Nom', 'ordering': 'Ordre', 'birthYear': 'Naissance',
                         'deathYear': 'Décès'}, inplace=True)
    data['indice_MORE'] = ((data['Note'] * data['Votes']) / (data['Votes'].sum()) * 1000000000).apply(math.sqrt).apply(
        math.sqrt).apply(lambda x: round(x, 2))
    return data


def fav_filter(dataframe):
    """Return a data filtered"""
    return dataframe[(dataframe['Votes'].between(min_pop, max_pop)) &
                     (dataframe['Note'].between(min_rate, max_rate))]


st.set_page_config(page_title="Projet MORE",
                   page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )

# style
st.markdown("""
    <style>
    .important_info {
        font-size:30px;
        font-weight:bold;
        color:steelblue;
    }
    .actress_stat {
        font-size:30px;
        color:darkorange;
    }
    .sub_title {
        font-size:25px;
        font-weight:bold;
    }        
    </style>
    """, unsafe_allow_html=True)

#############
## sidebar ##
#############

st.sidebar.title('Navigation')

categorie = st.sidebar.radio("Categorie", ('La Base de Données', 'Analyse Comparative', 'Femme et Cinéma',
                                           'Retrospective', 'Machine Learning'))

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
REPRO_DB = 'https://github.com/MickaelKohler/Projet_MORE/raw/5229e2c46ed10881eb3b9e372cd4c0198c4b15d5/repro.zip'
FR_MOV_DB = 'https://github.com/MickaelKohler/Projet_MORE/raw/main/fr_mov.csv'
ML_DB = 'https://github.com/MickaelKohler/Projet_MORE/raw/5229e2c46ed10881eb3b9e372cd4c0198c4b15d5/mldb.csv'

data = load_data(FR_MOV_DB)
data_crew = load_data(REPRO_DB)
tick_max = data['Votes'].max().item()
tick_min = data['Votes'].min().item()

###############
## MAIN PAGE ##
###############

if categorie == 'La Base de Données':

    st.title('La Base de Données')


elif categorie == 'Analyse Comparative':

    st.title('Analyse comparative')

    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown(
            """
        La *courbe bleue* montre différentes caractéristiques des films selon les décénnies.
        Elles sont extraites de la base de données du site **imbd**.
        """)
    with col2:
        st.markdown(
            """
        La *courbe orange* montre les mêmes indices selon le filtre selectionné.
        Il suffit de modifier les curseurs pour modifier la plage de notes et de votes appliquée.
        Par défaut, on retient les films ayant une **note supérieure à 7 pour plus de 1500 votes**.
        """)

    # filters
    filter_expand = st.beta_expander('Filtres')
    with filter_expand:
        defaults = st.selectbox("Filtres predéfinis",
                                [
                                    {"name": "Par défaut",
                                     "min_rate": 7,
                                     "max_rate": 10,
                                     "min_vote": 1500,
                                     "max_vote": 150000
                                     },
                                    {"name": "Les 1000 meilleurs films",
                                     "min_rate": 7,
                                     "max_rate": 10,
                                     "min_vote": 170000,
                                     "max_vote": 150000
                                     },
                                    {"name": "Les 100 meilleurs films",
                                     "min_rate": 8,
                                     "max_rate": 10,
                                     "min_vote": 740000,
                                     "max_vote": 150000
                                     },
                                    {"name": "Fin du classement",
                                     "min_rate": 0,
                                     "max_rate": 7,
                                     "min_vote": tick_min,
                                     "max_vote": 1500
                                     },
                                    {"name": "Tout inclus",
                                     "min_rate": 0,
                                     "max_rate": 10,
                                     "min_vote": tick_min,
                                     "max_vote": 150000
                                     },
                                ],
                                format_func=lambda option: option["name"]
                                )
        rating_filter = st.slider('Selectionnez une plage de notes', 0, 10, (defaults['min_rate'],
                                                                             defaults['max_rate']))

        vote_filter = st.slider('Selectionnez le nombre de votes '
                                '(150.000 renvoie au maximum des votes)', tick_min, 150000, (defaults['min_vote'],
                                                                                             defaults['max_vote']))
        show = st.checkbox('Montre moi la data')

    min_rate, max_rate = zip(rating_filter)
    min_pop = int(vote_filter[0])
    max_pop = int(vote_filter[1])
    if max_pop < 150000:
        min_pop, max_pop = zip(vote_filter)
    else:
        max_pop = tick_max

    # percent of best
    total_decade = data.groupby((data['Année'] // 10) * 10).count()
    best_mov = fav_filter(data)
    best_decade = best_mov.groupby((data['Année'] // 10) * 10).count()

    percent_best = best_decade[['Titre']].rename(columns={'Titre': 'Meilleurs films'}).loc[1920:2020]
    percent_best['Total'] = total_decade[['Titre']].rename(columns={'Titre': 'Total'}).loc[1920:2020]
    percent_best['Pourcentage'] = (percent_best['Meilleurs films'] * 100) / percent_best['Total']

    fig = px.bar(percent_best, x=percent_best.index, y='Pourcentage',
                 title='<b>Proportion des films selon le filtre</b> (en pourcents)',
                 color_discrete_sequence=['darkorange'])
    fig.update_yaxes(title=None, tick0=True, nticks=12)
    fig.update_xaxes(title=None, nticks=12)
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                      margin=dict(l=30, r=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    if show:
        percent_best

    st.markdown(
        """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus.
        Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor.
        Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi.
        Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat.
        """
    )

    # Runtime
    temp_tab = data[data['Durée'] != '\\N'][['Année', 'Durée', 'Note', 'Votes']]
    temp_tab['Durée'] = pd.to_numeric(temp_tab['Durée'])
    temp_tab = temp_tab[temp_tab['Durée'] < 500]
    mov_runtime = temp_tab.groupby('Année').mean()[['Durée']].loc['1920':'2020']
    best_mov = fav_filter(temp_tab)
    mov_runtime['Filtrés'] = best_mov.groupby('Année').mean()[['Durée']]
    mov_runtime.fillna(mov_runtime['Filtrés'].median(), inplace=True)

    fig = px.line(mov_runtime, x=mov_runtime.index, y=["Durée", "Filtrés"],
                  title='<b>Durée moyenne</b> (en mintues)',
                  color_discrete_map={
                      'Durée': 'steelblue',
                      'Filtrés': 'darkorange'})
    fig.update_yaxes(title=None, tick0=True, nticks=12)
    fig.update_xaxes(title=None, nticks=12)
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                      margin=dict(l=30, r=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    if show:
        mov_runtime

    # Age
    age = data_crew[(data_crew['category'].isin(['actor', 'actress'])) & (data_crew['Naissance'] != '\\N')]
    age['Naissance'] = pd.to_numeric(age['Naissance'])
    age['Age'] = age['Année'] - age['Naissance']
    age_chart = age.groupby((age['Année'] // 10) * 10).median()[['Age']].loc[1920:2020]
    best_age = fav_filter(age)
    age_chart['Filtrés'] = best_age.groupby((best_age['Année'] // 10) * 10).median()[['Age']]

    fig = px.bar(age_chart, x=age_chart.index, y=["Age", "Filtrés"], barmode='group',
                 title='<b>Age moyen des acteurs</b> (en années)',
                 color_discrete_map={
                     'Age': 'steelblue',
                     'Filtrés': 'darkorange'})
    fig.update_yaxes(title=None, nticks=10)
    fig.update_xaxes(title=None, nticks=12)
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                      margin=dict(l=30, r=30, b=30))
    st.plotly_chart(fig, use_container_width=True)
    if show:
        age_chart


elif categorie == 'Femme et Cinéma':

    data_crew = load_data(REPRO_DB)
    actors = data_crew[data_crew['category'].isin(['actor', 'actress'])]

    col1, col2 = st.beta_columns(2)
    with col1:
        st.title('La place des Femmes dans le Cinéma')
        st.title(' ')
        st.markdown(
            """
            *Je ne suis pas une femme qui fait du cinéma, mais quelqu’un qui fait du cinéma*, Coline Serreau
    
            Au-delà des tapis rouges et de la grande famille du cinéma,
            l’envers du décor montre une situation plus inégale qu’il n’y parait.
    
            Cette rubrique est l’occasion de donner un coup de projecteur sur les inégalités
            portant sur la représentation de la femme au cinéma et le rôle qu’elle occupe dans les films. 
            """)
    with col2:
        prop_total = actors.value_counts('category', normalize=True)
        fig = go.Figure(data=[go.Pie(labels=prop_total.index, values=prop_total, pull=[0.2, 0])])
        fig.update_layout(showlegend=False,
                          font=dict(
                              family="IBM Plex Sans",
                              size=10,
                              color="darkgray"),
                          title={
                              'text': "Proportion d'actrices dans la base de données",
                              'y': 0.05,
                              'x': 0.52,
                              'xanchor': 'center',
                              'yanchor': 'middle'},
                          margin=dict(l=30, r=30, b=30, t=70))
        fig.update_traces(textfont_size=20, marker=dict(colors=['steelblue', 'darkorange'],
                                                        line=dict(color='#000000', width=1)))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        Si la proportion d'actrices dans la base de données est inférieure à 40%, la situation est évidement la même
        sur dans la distribution des roles. **Les femmes représent généralement moins de 35% des personnes à l'écran**,
        même si on note une légère tendence vers l'équilibre à partir des années 90.
    """)

    # proportion d'actrices dans
    actor = actors[actors['category'].isin(['actor'])]
    nb_act = actor.groupby('Titre').count()['Nom']

    actress = actors[actors['category'].isin(['actress'])]
    nb_acts = actress.groupby('Titre').count()['Total']

    total = actors.groupby('Titre')['Année'].max()
    total_bis = pd.concat([total, nb_acts, nb_act], axis=1).rename(
        columns={'Nom': 'nb_Acteur', 'Total': 'nb_Actrice'}).fillna(0)

    percent = total_bis.groupby((total_bis['Année'] // 10) * 10).sum()[['nb_Acteur', 'nb_Actrice']]
    percent['Acteur'] = (percent['nb_Acteur'] * 100) / (percent['nb_Acteur'] + percent['nb_Actrice'])
    percent['Actrice'] = (percent['nb_Actrice'] * 100) / (percent['nb_Acteur'] + percent['nb_Actrice'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=percent["Acteur"],
        x=percent.index,
        name="Acteur",
        marker=dict(
            color='steelblue',
            line=dict(color='steelblue', width=0.05)
        )
    ))
    fig.add_trace(go.Bar(
        y=percent["Actrice"],
        x=percent.index,
        name="Actrice",
        marker=dict(
            color='darkorange',
            line=dict(color='darkorange', width=0.05)
        )
    ))
    fig.update_layout(
        yaxis=dict(
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            tickvals=[0, 20, 40, 60, 80, 100],
            tickmode="array",
        ),
        xaxis=dict(
            nticks=20
        ),
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=30, r=30, b=30, t=70),
        autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',
        title="<b>Proportion d'actrices présente au grand écran par décennies</b>",
        barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
        On peut constater que l'actrice est avant tout une femme jeune.
        La différence d'age moyenne entre acteur et actrice se maintient à 10 ans environ,
        même si on peut constater un léger fléchissement depuis les années 80. 
    """)

    # Age comparaison
    age = actors[actors['Naissance'] != '\\N']
    age['Naissance'] = pd.to_numeric(age['Naissance'])
    age['Age'] = age['Année'] - age['Naissance']

    actor_age = age[age['category'].isin(['actor'])][['Année', 'Age']].groupby('Année').median().loc[1920:2020]
    actor_age['category'] = 'acteur'
    actress_age = age[age['category'].isin(['actress'])][['Année', 'Age']].groupby('Année').median().loc[1920:2020]
    actress_age['category'] = 'actrice'

    total = pd.concat([actor_age, actress_age], axis=1)
    total['Age'] = total.iloc[:, 0] - total.iloc[:, 2]
    diff_age = total.iloc[:, -2:]
    diff_age['category'] = 'Difference'

    final = pd.concat([actor_age, actress_age, diff_age]).reset_index()

    layout = go.Layout(
        title="<b>Age moyen des acteurs et des actrices lors du tournage d'un film</b>",
        font=dict(family="IBM Plex Sans"),
        hovermode="x",
        hoverdistance=100,
        spikedistance=1000,
        margin=dict(l=30, r=30, b=30, t=70),
        colorway=['steelblue', 'darkorange', 'tomato'],
        xaxis=dict(
            title=None,
            nticks=12,
            linecolor="#BCCCDC",
            showspikes=True,
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title=None,
            nticks=12
        )
    )

    data = []
    for role in ["acteur", "actrice", 'Difference']:
        time = final.loc[final.category == role, "Année"]
        price = final.loc[final.category == role, "Age"]
        line_chart = go.Scatter(
            x=time,
            y=price,
            name=role
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    st.title(' ')
    st.markdown('<p class="sub_title"> Comparaison entre les films et le TOP des films</p>', unsafe_allow_html=True)

    # filters
    defaults = st.selectbox("Filtres predéfinis",
                            [
                                {"name": "Par défaut",
                                 "min_rate": 7,
                                 "min_vote": 1500,
                                 },
                                {"name": "Les 1000 meilleurs films",
                                 "min_rate": 7,
                                 "min_vote": 170000,
                                 },
                                {"name": "Les 100 meilleurs films",
                                 "min_rate": 8,
                                 "min_vote": 740000,
                                 },
                            ],
                            format_func=lambda option: option["name"]
                            )
    filter_expand = st.beta_expander('Détails du filtre')
    with filter_expand:
        min_rate = st.slider('Selectionnez une notes minimale', 0, 10, defaults['min_rate'])
        min_pop = st.slider('Selectionnez un nombre de votes minimal ', tick_min, tick_max, defaults['min_vote'])
        max_rate, max_pop = 10, tick_max
        show = st.checkbox('Montre moi la data')

    # proportion per movies
    col1, col2 = st.beta_columns(2)
    with col1:

        # Mean percent actress in each movie
        per_mov = actors.groupby(['Titre', 'category']).count()['characters'].unstack().fillna(0.0)
        per_mov['Proportion Actrices'] = (per_mov['actress'] * 100) / (per_mov['actress'] + per_mov['actor'])
        per_mov['Proportion Acteurs'] = (per_mov['actor'] * 100) / (per_mov['actress'] + per_mov['actor'])

        top_per_mov = fav_filter(actors).groupby(['Titre', 'category']).count()['characters'].unstack().fillna(0.0)
        top_per_mov['Proportion Actrices'] = (top_per_mov['actress'] * 100) / (
                top_per_mov['actress'] + top_per_mov['actor'])
        top_per_mov['Proportion Acteurs'] = (top_per_mov['actor'] * 100) / (
                top_per_mov['actress'] + top_per_mov['actor'])

        final = pd.concat([pd.DataFrame(per_mov.mean()).T, pd.DataFrame(top_per_mov.mean()).T]).reset_index().iloc[:,
                1:].rename(index={0: 'Global', 1: 'Filtré'})

        fig = px.bar(final, x=final.index, y=["Proportion Actrices", "Proportion Acteurs"], barmode='group',
                     title="<b>Proportion moyenne d'actrices dans les films</b>",
                     color_discrete_map={'Proportion Acteurs': 'steelblue', 'Proportion Actrices': 'darkorange'})
        fig.update_yaxes(title=None, showticklabels=False, range=[0, 85], showgrid=False)
        fig.update_xaxes(title=None, nticks=12)
        fig.update_traces(texttemplate='%{text:.2s}%', textposition='outside')
        fig.update_layout(showlegend=True, font_family='IBM Plex Sans',
                          uniformtext_minsize=14, uniformtext_mode='hide',
                          margin=dict(l=10, r=10, b=10),
                          plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(
                              x=0,
                              y=1,
                              traceorder="normal",
                              bgcolor='rgba(0,0,0,0)',
                              font=dict(
                                  size=12)))
        texts = [final["Proportion Actrices"], final["Proportion Acteurs"]]
        for i, t in enumerate(texts):
            fig.data[i].text = t
            fig.data[i].textposition = 'outside'
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        best_actors = fav_filter(actors)
        nb_vote_W = best_actors[(best_actors['category'] == 'actress') &
                                (best_actors['Ordre'] == 1)].mean().iloc[2]
        nb_votes_M = best_actors[(best_actors['category'] == 'actor') &
                                 (best_actors['Ordre'] == 1)].mean().iloc[2]

        st.title(" ")
        st.markdown(
            f"""
        Si peu de films n'ont aucune femme dans leur casting, 
        on peut constater que les actrices sont généralement placées en retrait
        et interviennent surtout pour des seconds rôles.

        Un **filtre**, ci dessus, permet de changer la popularité des films étudiés,
        et de mettre en avant l’aggravation des inégalités dès lors qu’on se rapproche des films les plus populaires.

        La popularité se ressent aussi au niveau du nombre de votes 
        qui est inférieur dès lors que l'actrice tient le role principal.

        Pour un premier role, on constate en moyenne **{round(nb_vote_W)} votes pour une actrice**,
        contre **{round(nb_votes_M)} votes pour un acteur**, soit un **nombre de votre inférieur de
        {round(100 - ((nb_vote_W * 100) / nb_votes_M))} %**.
        """)

    col1, col2 = st.beta_columns(2)
    with col1:
        best_actors = fav_filter(actors)
        percent = best_actors[best_actors['Ordre'] == 1].value_counts('category', normalize=True)

        fig = go.Figure(data=[go.Pie(labels=percent.index, values=percent, pull=[0.2, 0])])
        fig.update_layout(showlegend=False,
                          font=dict(
                              family="IBM Plex Sans",
                              size=10),
                          title={
                              'text': "Proportion d'actrices ayant le premier rôle",
                              'y': 0.02,
                              'x': 0.48,
                              'xanchor': 'center',
                              'yanchor': 'bottom'},
                          margin=dict(l=20, r=60, b=60, t=60))
        fig.update_traces(textfont_size=20, marker=dict(colors=['steelblue', 'darkorange'],
                                                        line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        raw_order = pd.DataFrame(best_actors[(best_actors['category'].isin(['actress'])) &
                                             (best_actors['Ordre'] < 5)].value_counts('Ordre',
                                                                                      normalize=True).sort_index()).rename(
            columns={0: 'Actrices'})
        raw_order['Acteurs'] = best_actors[(best_actors['category'].isin(['actor'])) &
                                           (best_actors['Ordre'] < 5)].value_counts('Ordre',
                                                                                    normalize=True).sort_index()
        raw_order = raw_order.apply(lambda x: x * 100).round(2)

        fig = px.bar(raw_order, x=raw_order.index, y=["Actrices", "Acteurs"], barmode='group',
                     title="<b>A quel rang se trouve les actrices et acteurs ?</b>",
                     color_discrete_map={'Acteurs': 'steelblue', 'Actrices': 'darkorange'})
        fig.update_yaxes(title='pourcents')
        fig.update_xaxes(title='Du rôle principal au second rôle (une couleur est égale à 100%)')
        fig.update_traces(texttemplate='%{text:.2s}%', textposition='outside')
        fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                          margin=dict(l=0, r=0, b=0))
        texts = [raw_order["Actrices"], raw_order["Acteurs"]]
        for i, t in enumerate(texts):
            fig.data[i].text = t
            fig.data[i].textposition = 'outside'
        st.plotly_chart(fig, use_container_width=True)

    if show:
        st.title('')
        st.markdown('**Détail des films de la selection**')
        st.markdown('Le role principal est tenu par une actrice : ')
        best_actors[(best_actors['category'] == 'actress') & (best_actors['Ordre'] == 1)]
        st.markdown('Le role principal est tenu par un acteur : ')
        best_actors[(best_actors['category'] == 'actor') & (best_actors['Ordre'] == 1)]

elif categorie == 'Retrospective':
    st.title('Retrospective par décennie')

    st.write("""
    Classement selon les données présentes sur la platforme **imbd**
    """)

    data = load_data(REPRO_DB)
    # create a filter
    start_year = st.slider('Décennie', 1920, 2020, 1990, 10)
    stop_year = start_year + 10

    # display best all times
    if st.button('Toutes les années confondues'):
        start_year = 1920
        stop_year = 2020

    # filter tables
    filtre = st.selectbox(
        'Vous souhaitez filtrer par :',
        ['indice MORE', 'Votes', 'Total', 'Note'])
    expander = st.beta_expander("indice MORE ?")
    expander.markdown("**L'indice MORE** permet de lier le nombre de votes et la note d'un film "
                      "pour en retirer un indice de popularité. Plus l'indice est élevée (de 1 à 40) "
                      "plus le film est populaire. "
                      "Pour les autres catégories que les films, c'est la moyenne des indices sur la décennie "
                      "selectionnée qui est calculé."
                      )

    filtered_data = data[(data['Année'] >= start_year) & (data['Année'] < stop_year)]

    # diplay the data
    st.subheader('Top 10 des films')
    temp_tab = filtered_data[['Titre', 'Année', 'Genres', 'Note', 'Votes', 'indice MORE']]
    top_film = temp_tab.groupby(['Titre', 'Année']).max().sort_values(
        'indice MORE' if filtre == 'Total' else filtre, ascending=False)
    top_film.reset_index(inplace=True)
    top_film.index = top_film.index + 1
    st.table(top_film.iloc[0:10])

    st.subheader('Top 10 des réalisateurs')
    temp_tab = filtered_data[filtered_data['category'] == 'director'][['Nom', 'Naissance', 'Décès',
                                                                       'Total', 'Note', 'Votes', 'indice MORE']]
    top_real = temp_tab.groupby(['Nom', 'Naissance']) \
        .agg({'Note': 'mean', 'Votes': 'sum', 'Total': 'count', 'indice MORE': 'mean'}) \
        .sort_values(filtre, ascending=False)
    top_real.reset_index(inplace=True)
    top_real.index = top_real.index + 1
    st.table(top_real.iloc[0:10])

    st.subheader('Top 10 des acteurs')
    temp_tab = filtered_data[filtered_data['category'].isin(['actor'])][['Nom', 'Naissance', 'Décès',
                                                                         'Total', 'Note', 'Votes', 'indice MORE']]
    top_act = temp_tab.groupby(['Nom', 'Naissance']) \
        .agg({'Note': 'mean', 'Votes': 'sum', 'Total': 'count', 'indice MORE': 'mean'}) \
        .sort_values(filtre, ascending=False)
    top_act.reset_index(inplace=True)
    top_act.index = top_act.index + 1
    st.table(top_act.iloc[0:10])

    st.subheader('Top 10 des actrices')
    temp_tab = filtered_data[filtered_data['category'].isin(['actress'])][['Nom', 'Naissance', 'Décès',
                                                                           'Total', 'Note', 'Votes', 'indice MORE']]
    top_act = temp_tab.groupby(['Nom', 'Naissance']) \
        .agg({'Note': 'mean', 'Votes': 'sum', 'Total': 'count', 'indice MORE': 'mean'}) \
        .sort_values(filtre, ascending=False)
    top_act.reset_index(inplace=True)
    top_act.index = top_act.index + 1
    st.table(top_act.iloc[0:10])

    st.subheader('Top 10 des compositeurs')
    temp_tab = filtered_data[filtered_data['category'].isin(['composer'])][['Nom', 'Naissance', 'Décès',
                                                                            'Total', 'Note', 'Votes', 'indice MORE']]
    top_act = temp_tab.groupby(['Nom', 'Naissance']) \
        .agg({'Note': 'mean', 'Votes': 'sum', 'Total': 'count', 'indice MORE': 'mean'}) \
        .sort_values(filtre, ascending=False)
    top_act.reset_index(inplace=True)
    top_act.index = top_act.index + 1
    st.table(top_act.iloc[0:10])


elif categorie == 'Machine Learning':
    st.title('Machine Learning')
    st.subheader('Recommandation de films')

    # Model ML preparation
    mov_db = load_data(ML_DB)

    temp_db = mov_db.copy()
    temp_db['Note'] = temp_db['Note'] * 10

    temp_db['director'] = temp_db['director'].factorize()[0]
    temp_db['main_role'] = temp_db['main_role'].factorize()[0]
    temp_db['second_role'] = temp_db['second_role'].factorize()[0]
    temp_db['third_role'] = temp_db['third_role'].factorize()[0]
    temp_db['Genres'] = temp_db['Genres'].apply(lambda x: x.split(','))

    temp_tab = temp_db.explode('Genres')['Genres'].str.get_dummies()
    db = pd.concat([temp_db, temp_tab.groupby(temp_tab.index).agg('sum')], axis=1).drop(columns=['Genres'])

    mldb = db[['director', 'main_role', 'second_role', 'third_role', 'indice_MORE', 'Année', 'Votes', 'Drama',
               'Comedy', 'Action', 'Thriller', 'Adventure', 'Animation', 'Biography', 'Crime', 'Documentary', 'Romance',
               'Family', 'Sci-Fi', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
               'Mystery', 'News', 'Reality-TV', 'Short', 'Sport', 'War', 'Western', 'Adult', '\\N']]

    mldb.iloc[:, :7] = preprocessing.normalize(mldb.iloc[:, :7])
    mldb.iloc[:, 7:] = preprocessing.normalize(mldb.iloc[:, 7:])

    mldb['Note'] = db['Note']

    # Train model
    X = mldb.drop(columns=['Note'])
    y = mldb['Note']
    modelMORE = KNeighborsClassifier(weights='distance', n_neighbors=5).fit(X, y)

    st.markdown(
        f"""
        Cet outil permet d'utiliser toute la puissance du *Machine Learning* pour vous proposer des films qui sont
        le plus en proches de vos goûts.
        
        Afin de de vous faire une proposition, nous vous invitons à selectionner un de vos films favoris pour que 
        l'outil MORE puisse l'analyser et vous proposer des films proches
        parmis **une selection de {mov_db.index[-1]} films**.
        """
    )

    movie_selected = st.selectbox('Choisissez votre film :', mov_db['Titre'])

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.3)
        my_bar.progress(percent_complete + 1)

    #Show result
    reco = pd.DataFrame(data=modelMORE.kneighbors(X, return_distance=False))
    reco = reco.loc[mov_db[mov_db['Titre'] == movie_selected].index]

    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        reco_one = mov_db[mov_db.index == reco.iloc[0, 1]][['tconst', 'Titre']]
        st.subheader(reco_one.iloc[0, 1])

        page = urllib.request.urlopen('https://www.imdb.com/title/' + reco_one.iloc[0, 0] + '/?ref_=adv_li_i%27')
        htmlCode = page.read().decode('UTF-8')
        soup = Soup(htmlCode)
        tds = soup.find("div", {"class": "poster"})
        img = tds[0].find("img")
        st.image(img.attrs['src'])

    with col2:
        reco_two = mov_db[mov_db.index == reco.iloc[0, 2]][['tconst', 'Titre']]
        st.subheader(reco_two.iloc[0, 1])

        page = urllib.request.urlopen('https://www.imdb.com/title/' + reco_two.iloc[0, 0] + '/?ref_=adv_li_i%27')
        htmlCode = page.read().decode('UTF-8')
        soup = Soup(htmlCode)
        tds = soup.find("div", {"class": "poster"})
        img = tds[0].find("img")
        st.image(img.attrs['src'])

    with col3:
        reco_three = mov_db[mov_db.index == reco.iloc[0, 3]][['tconst', 'Titre']]
        st.subheader(reco_three.iloc[0, 1])

        page = urllib.request.urlopen('https://www.imdb.com/title/' + reco_three.iloc[0, 0] + '/?ref_=adv_li_i%27')
        htmlCode = page.read().decode('UTF-8')
        soup = Soup(htmlCode)
        tds = soup.find("div", {"class": "poster"})
        img = tds[0].find("img")
        st.image(img.attrs['src'])

    with col4:
        reco_four = mov_db[mov_db.index == reco.iloc[0, 4]][['tconst', 'Titre']]
        st.subheader(reco_four.iloc[0, 1])

        page = urllib.request.urlopen('https://www.imdb.com/title/' + reco_four.iloc[0, 0] + '/?ref_=adv_li_i%27')
        htmlCode = page.read().decode('UTF-8')
        soup = Soup(htmlCode)
        tds = soup.find("div", {"class": "poster"})
        img = tds[0].find("img")
        st.image(img.attrs['src'])



# lien tuto streamlit
# https://docs.streamlit.io/en/stable/getting_started.html

# Galerie 
# https://streamlit.io/gallery?type=apps&category=data-visualization

# Doc
# https://awesome-streamlit.org
