import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import urllib.request
from gazpacho import Soup


###############
## Fonctions ##
###############

@st.cache
def load_data(url):
    db = pd.read_csv(url)
    db.rename(columns={'title': 'Titre', 'startYear': 'Année', 'genres': 'Genres', 'averageRating': 'Note',
                       'numVotes': 'Votes', 'primaryName': 'Nom', 'ordering': 'Ordre', 'birthYear': 'Naissance',
                       'deathYear': 'Décès'}, inplace=True)
    db['indice MORE'] = ((db['Note'] * db['Votes']) / (db['Votes'].sum()) * 1000000000).apply(math.sqrt).apply(
        math.sqrt).apply(lambda x: round(x, 2))
    return db


@st.cache
def load_df(url):
    df = pd.read_csv(url)
    df.set_index(df.iloc[:,0], inplace=True)
    df = df.iloc[:, 1:]
    return df


def fav_filter(dataframe):
    """Return a data filtered"""
    return dataframe[(dataframe['Votes'].between(min_pop, max_pop)) &
                     (dataframe['Note'].between(min_rate, max_rate))]


def ml_data(df):
    data = df.copy()
    data['director'] = data['director'].factorize()[0]
    data['main_role'] = data['main_role'].factorize()[0]
    data['second_role'] = data['second_role'].factorize()[0]
    data['third_role'] = data['third_role'].factorize()[0]
    data['Genres'] = data['Genres'].apply(lambda x: x.split(','))
    data = data[['Année', 'indice MORE', 'Votes', 'director',
                 'main_role', 'second_role', 'third_role', 'Genres']]
    temp = data.explode('Genres')['Genres'].str.get_dummies()
    mldb = pd.concat([data, temp.groupby(temp.index).agg('sum')], axis=1).drop(columns=['Genres'])
    power = PowerTransformer().fit(mldb)
    return power.transform(mldb)


def ml_rating(df):
    data = df.copy()
    data['director'] = data['director'].factorize()[0]
    data['main_role'] = data['main_role'].factorize()[0]
    data['second_role'] = data['second_role'].factorize()[0]
    data['third_role'] = data['third_role'].factorize()[0]
    data['Genres'] = data['Genres'].apply(lambda x: x.split(','))
    data = data[['Année', 'indice MORE', 'Votes', 'Note', 'director',
                 'main_role', 'second_role', 'third_role', 'Genres']]
    temp = data.explode('Genres')['Genres'].str.get_dummies()
    mldb = pd.concat([data, temp.groupby(temp.index).agg('sum')], axis=1).drop(columns=['Genres'])
    power = PowerTransformer().fit(mldb)
    return power.transform(mldb)


def picture(index):
    page = urllib.request.urlopen('https://www.imdb.com/title/' +
                                  index.iloc[0, 0] +
                                  '/?ref_=adv_li_i%27')
    htmlCode = page.read().decode('UTF-8')
    soup = Soup(htmlCode)
    tds = soup.find("div", {"class": "poster"})
    img = tds[0].find("img")
    return img.attrs['src']


class Retrospective:
    def __init__(self, film, type_retro):
        """
            A l'initialisation l'utilisateur choisit un film et un type de rétro, auxquels on ajoute
            un réalisateur, une année de sortie, et un (ou des) genres
        """
        self.film = film
        self.tr = type_retro
        self.real = mov_RL[mov_RL["Titre"] == film]["director"].values[0]
        self.annee = mov_RL[mov_RL["Titre"] == film]["Année"].values[0]
        self.genres = mov_RL[mov_RL["Titre"] == film]["Genres"].values[0]

    def __repr__(self):
        """
           Méthode cosmétique pour afficher tous les attributs d'une rétro de façon propre
        """
        return (f'Film : {self.film} - Réalisateur : {self.real} - Sortie : {self.annee} - '
                f'Genre(s) : {self.genres} - Rétro souhaitée : {self.tr}')

    def propo_retro(self):
        """
            Cette fonction va proposer des rétro en fonction du type choisie à l'initiatlisation de la classe.
        """
        DF_retro = self.filtre_retro()
        if self.tr == "réalisateur":
            DF_retro = self.quatre_tests(DF_retro)
        elif self.tr == "décennie":
            t_genres = self.genres.split(',')
            if len(t_genres) > 1:
                # s'il y a plus d'un genre, on filtre les films qui partagent les deux derniers genres
                DF_retro = DF_retro[(DF_retro["Genres"].str.contains(t_genres[-1])) &
                                    (DF_retro["Genres"].str.contains(t_genres[-2]))].reset_index(drop=True)
                DF_retro = self.quatre_tests(DF_retro)
            # sinon il n'y a qu'un genre, et on scrute les films proches qui ne partagent QUE ce genre
            else:
                print('test2')
                DF_retro = DF_retro[DF_retro["Genres"] == t_genres[0]].reset_index(drop=True)
                DF_retro = self.quatre_tests(DF_retro)
        return DF_retro

    def quatre_tests(self, DF):
        """
            Cette fonction sert à résumer les tests rébartatifs que l'on fait avant de lâcher une rétro
        """
        index_ref = int(DF[DF["Titre"] == self.film].index[0])
        # d'abord, si la DF filtrée a au maximum cinq films, on la retourne sans le film considéré
        if len(DF) <= 5:
            DF.drop(labels=index_ref, inplace=True)
        # sinon, si le film cherché a un index trop proche du début de DF, on affiche
        # les quatre films suivants
        elif index_ref < 2:
            DF = DF.iloc[index_ref:index_ref + 4]
            DF.drop(labels=index_ref, axis=0, inplace=True)
        # même chose en symétrique si le film est cette fois trop proche de la fin de DF
        elif index_ref > (len(DF) - 2):
            DF = DF.iloc[index_ref - 4:index_ref]
            DF.drop(labels=index_ref, axis=0, inplace=True)
        # sinon, on prend deux films avant et deux films après
        else:
            DF = DF.iloc[index_ref - 2:index_ref + 3]
            DF.drop(labels=index_ref, axis=0, inplace=True)
        DF = DF.copy().reset_index()
        DF.drop(["index"], axis=1, inplace=True)
        return DF

    def filtre_retro(self):
        """
            Cette fonction applique un filtre simple (décennie ou réalisateur).
        """
        DF_retro = mov_RL.copy()
        if self.tr == "décennie":
            DF_retro = DF_retro[DF_retro["Année"].astype(str).str[2] == str(self.annee)[2]].reset_index(drop=True)
        elif self.tr == "réalisateur":
            DF_retro = DF_retro[DF_retro["director"] == self.real].reset_index(drop=True)
        return DF_retro


# Option
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
        color:deepskyblue;
    }
    .actress_stat {
        font-size:30px;
        color:coral;
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

st.sidebar.title('Projet MORE')
st.sidebar.subheader('Navigation')

categorie = st.sidebar.radio("Categorie", ("Qu'est-ce que le projet MORE ?", 'Presentation de la Base de données',
                                           'Femme et Cinéma', 'Les TOP par décennies', 'Quoi voir ?'))
if categorie == 'Quoi voir ?':
    sub_categorie = st.sidebar.radio("Machine Learning", ('Recommandation de films',
                                                          'Restrospectives',
                                                          'Probabilité que vous aimiez ce film'))

st.sidebar.title(' ')
option = st.sidebar.beta_expander("Options")
option.markdown(
    """
    L'option _Montre moi la data_ affichera les données 
    qui ont permis de réaliser les graphiques, sous forme de tableaux. 
    """)
show = option.checkbox('Montre moi la data')

expander = st.sidebar.beta_expander("Sources")
expander.markdown(
    """
    [Base de donnée imdb](https://www.imdb.com/interfaces/) : N'ont été retenus que les films 
    ayant été distribués en France.

    [Base de donnée Netflix](https://en.wikipedia.org/wiki/Lists_of_Netflix_original_films) : 
    Les films exclusifs à la plaforme Netflix ont été retirés.
    """)
expander.info('Résiliation de la **Team MORE** : _Alhem, Fanyme, Michael, Raphael, Soufiane_')
expander.info('Projet de la **WildCodeSchool** livré le 07/05/2021')


##########
## DATA ##
##########

# modifier selon la localisation de la BD
REPRO_DB = 'https://github.com/MickaelKohler/Projet_MORE/raw/5229e2c46ed10881eb3b9e372cd4c0198c4b15d5/repro.zip'
FR_ml_db = 'https://github.com/MickaelKohler/Projet_MORE/raw/main/fr_mov.csv'
ML_DB = 'https://github.com/MickaelKohler/Projet_MORE/raw/main/mldb.csv'
country = 'https://raw.githubusercontent.com/MickaelKohler/Projet_MORE/main/country.csv'
prod = 'https://raw.githubusercontent.com/MickaelKohler/Projet_MORE/main/cum_prod.csv'

data = load_data(FR_ml_db)
data_crew = load_data(REPRO_DB)
ml_db = load_data(ML_DB).sort_values('indice MORE', ascending=False).reset_index(drop=True)
tick_max = data['Votes'].max().item()
tick_min = data['Votes'].min().item()


###############
## MAIN PAGE ##
###############

if categorie == "Qu'est-ce que le projet MORE ?":

    st.title('Projet MORE ')
    st.subheader('Movie Recommandation programme')
    st.title(" ")

    st.markdown("""
    Bienvenu dans le Projet MORE.
    
    Ce projet a pour objectif de fournir les outils d’analyse d’une base de données de films, 
    afin d'en comprendre les **indices clés**. 
    
    Les deux premières parties mettent à disposition des **données comparatives sur le monde du cinéma** 
    en se concentrant sur la comparaison entre les meilleurs films et la moyenne des films 
    ou sur la place des femmes dans le cinéma. 
    
    La section suivante génère des **TOP 10**, de films, d’acteurs et de réalisateurs, 
    compte tenu de critères que vous aurez déterminés (une décennie ou un genre). 
    
    Enfin, la dernière section met à profit la puissance du **Machine Learning** 
    pour apporter les meilleures recommandations de films, 
    que ce soit la recommandation de films proches qu’un film selectionné, 
    la proposition de films dans le cadre d’une rétrospective et enfin, 
    la probabilité d'aimé un films compte tenu d’une sélection de films.
    
    Il ne vous reste plus qu’à explorer les sections !   
    """)

    st.title(" ")
    st.subheader('I need MORE !')
    col1, col2, col3 = st.beta_columns(3)
    with col2 :
        st.image('https://github.com/MickaelKohler/Projet_MORE/raw/main/Ressources/sub.png')


elif categorie == 'Presentation de la Base de données':
    st.title('Presentation de la Base de données')
    st.subheader("La face cachée des chiffres")

    st.markdown(
        """
        Afin d’avoir une vision générale des productions cinématographique, nous avons utilisé la base de données 
        mise à disposition par le **site de IMDb**, acronyme de _Internet Movie Database_.
        
        La base de donnée est mise à jour quotidiennement, mais pour les besoins de la présentation, 
        **les données ont été arrêtées à début mai 2021**.  

        Puisque le site est américain (propriété d’Amazon), le public est aussi à majorité anglo-saxonne, 
        ce qui va se ressentir dans le nombre de votes des films français. 
        Il ne faudra pas oublier cette donnée lors des différentes analyses. 

        Après cette première présentation, nous pouvons commencer par quelques observations générales. 
        """
    )

    prod = load_df(prod)
    col1, col2 = st.beta_columns([2, 1])
    with col2:
        st.title(' ')
        st.markdown(
            """
            Un premier point intéressant à regarder est la **diversité des oeuvres répertoriées dans IMDB**. 
            
            Une très large majorité de contenus est étiquetée _tvEpisode_, avec près de 6 millions d'entrées, 
            et **les films n'arrivent qu'en troisième position**, avec moins d'un million d'unités.
            """
        )
        filter_type = st.multiselect(label='Selectionnez les Types ?', options=list(prod.columns),
                                     default=['movie', 'short', 'tvEpisode'],
                                     help='Seuls les types principaux ont été retenus par défaut',)
    prod_fil = pd.DataFrame(index=prod.index)
    for col in filter_type:
        if col in prod.columns:
            prod_fil[col] = prod[col]
    with col1:
        fig = go.Figure()
        for i in range(0, len(prod_fil.columns)):
            fig.add_trace(go.Scatter(
                x=prod_fil.index,
                y=prod_fil.iloc[:, i],
                name=prod_fil.columns[i],
                hoverinfo='name+y',
                mode='lines',
                stackgroup='one'
            ))
        fig.update_layout(
            hovermode="closest",
            hoverdistance=100,
            spikedistance=1000,
            xaxis=dict(
                nticks=20,
                linecolor="#BCCCDC",
                showspikes=True,
                spikethickness=1,
                spikedash="dot",
                spikecolor="#999999",
                spikemode="across",
            ),
            yaxis=dict(
                title="Production (en millions d'unités)"),
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)),
            font=dict(family="IBM Plex Sans"),
            title="<b>Proportion des types de productions au fil du temps</b>",
            margin = dict(l=40, r=40, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    if show:
        st.dataframe(prod)

    st.markdown(
        """
        La production de contenu audiovisuel à commencer à prendre son essor à partir de la moitié des années 1910. 
        La montée en puissance de la **production de séries TV commencer à partir du début des années 60** et 
        explose à partir des années 2000 sans que cette dynamique semble fléchir. 
        
        La production cinématographique elle connait une croissance constante jusqu’à 2007 où 
        les nouvelles sorties s’accélèrent.
                
         **Mais qui regarde tous ces films ?** La distribution générale des films confirme une prédominance 
         du marché Nord-Américain. Comparativement, la France est à la **4eme position**, derrière l’Angleterre et 
         le Japon, confirmant sa place de pays cinéphile. 
        """
    )

    country = load_df(country)
    df = px.data.gapminder().query("year==2007")
    fig = go.Figure(data=go.Choropleth(
        locations=country.index,
        z=country['nb_mov'],
        text=country['country'],
        colorscale=px.colors.sequential.Sunsetdark,
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title='Nombre de films',
    ))
    fig.update_layout(
        dragmode=False,
        title_text='<b> Où les films sont ils le plus distribués ? </b>',
        margin=dict(l=40, r=40, b=40, t=40),
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        )
    )
    fig.update_geos(bgcolor='rgba(0,0,0,0)')

    if show:
        col1, col2 = st.beta_columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.title(' ')
            st.dataframe(country, height=360)
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    st.subheader("Qu'est ce qui caractérise un bon film ?")

    st.markdown(
        """
        Après ces présentations générales, nous pouvons analyser plus précisément la base de données 
        des films, avec l’objectif de déterminer les critères qui font un _bon film_.
        
        Mais d'ailleur, **qu'est-ce qu'un bon film ?**
        
        Regardons la répartition des notes moyennes et du nombre de votes de tous les films.
        """)

    col1, col2 = st.beta_columns(2)
    with col1:
        fig = px.box(data, y="Note", hover_data=["Titre"], notched=True, color_discrete_sequence=['deepskyblue'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(data, y="Votes", hover_data=["Titre"], notched=True, color_discrete_sequence=['coral'])
        fig.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        Les boites à moustaches mettent bien en avant la présence de nombreux outliers, 
        surtout dans le nombre de votes. On va arbitrairement choisir les **troisièmes quartiles** des deux variables 
        pour fixer notre filtre par défaut, soit **une moyenne de 7 pour un nombre de votes supérieur à 3000**. 

        Il est bien évidement possible de personaliser ce filtre.  
        """)

    # filters
    defaults = st.selectbox("Filtres predéfinis",
                            [
                                {"name": "Par défaut",
                                 "min_rate": 7,
                                 "max_rate": 10,
                                 "min_vote": 3000,
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
    filter_expand = st.beta_expander('Personnaliser le filtre')
    with filter_expand:
        rating_filter = st.slider('Selectionnez une plage de notes', 0, 10, (defaults['min_rate'],
                                                                             defaults['max_rate']))
        vote_filter = st.slider('Selectionnez le nombre de votes '
                                '(150.000 renvoie au maximum des votes)', tick_min, 150000, (defaults['min_vote'],
                                                                                             defaults['max_vote']))
    min_rate, max_rate = zip(rating_filter)
    min_pop = int(vote_filter[0])
    max_pop = int(vote_filter[1])
    if max_pop < 150000:
        min_pop, max_pop = zip(vote_filter)
    else:
        max_pop = tick_max

    col1, col2 = st.beta_columns(2)
    with col1:
        st.title("")
        st.markdown(
        """
        **Est-ce que c’était vraiment mieux avant ?** Globalement **non**.
        Le pourcentage de bon film par rapport au nombre total de films se situe en moyenne entre 8% et 10%. 
        
        Petite exception pour les années 90 et 2000 qui bénéficient de 2 points supplémentaires. 
        
        Pour la suite, :
        - la **courbe bleue** représentera le données moyennes pour **tous les films**, 
        - la **courbe orange** représentera les données moyennes pour les **_bons films_**.
        """)
    with col2:
        # percent of best
        total_decade = data.groupby((data['Année'] // 10) * 10).count()
        best_mov = fav_filter(data)
        best_decade = best_mov.groupby((data['Année'] // 10) * 10).count()

        percent_best = best_decade[['Titre']].rename(columns={'Titre': 'Meilleurs films'}).loc[1920:2020]
        percent_best['Total'] = total_decade[['Titre']].rename(columns={'Titre': 'Total'}).loc[1920:2020]
        percent_best['Pourcentage'] = (percent_best['Meilleurs films'] * 100) / percent_best['Total']

        fig = px.bar(percent_best, x=percent_best.index, y='Pourcentage',
                     title='<b>Proportion des films selon le filtre</b> (en pourcents)',
                     color_discrete_sequence=['coral'], text=percent_best['Pourcentage'].round(2))
        fig.update_yaxes(title=None, tick0=True, nticks=12, ticksuffix="%",
                         range=[0, percent_best['Pourcentage'].max()*1.1])
        fig.update_xaxes(title=None, nticks=12)
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                          margin=dict(l=30, r=30, b=30),
                          height=350)
        st.plotly_chart(fig, use_container_width=True)
    if show:
        col1, col2 = st.beta_columns(2)
        col2.dataframe(percent_best)

    # Runtime
    temp_tab = data[data['Durée'] != '\\N'][['Année', 'Durée', 'Note', 'Votes']]
    temp_tab['Durée'] = pd.to_numeric(temp_tab['Durée'])
    temp_tab = temp_tab[temp_tab['Durée'] < 500]
    mov_runtime = temp_tab.groupby('Année').mean()[['Durée']].loc['1920':'2020']
    best_mov = fav_filter(temp_tab)
    mov_runtime['Filtrés'] = best_mov.groupby('Année').mean()[['Durée']]
    mov_runtime.fillna(mov_runtime['Filtrés'].median(), inplace=True)

    fig = px.line(mov_runtime, x=mov_runtime.index, y=["Durée", "Filtrés"],
                  title='<b>Evolution de la durée des films en 100 ans</b> (en mintues)',
                  color_discrete_map={
                      'Durée': 'deepskyblue',
                      'Filtrés': 'coral'})
    fig.update_yaxes(title=None, tick0=True, nticks=12)
    fig.update_xaxes(title=None, nticks=12,
                     linecolor="#BCCCDC",
                     showspikes=True,
                     spikethickness=2,
                     spikedash="dot",
                     spikecolor="#999999",
                     spikemode="across")
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                      hovermode="x",
                      hoverdistance=100,
                      spikedistance=1000,
                      margin=dict(l=30, r=30, b=30))
    if show:
        col1, col2 = st.beta_columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.title(' ')
            st.dataframe(mov_runtime, height=380)
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        Première analyse comparative permet de constater que depuis les années 1920, 
        **les films ont gagné environ une demi-heure**. Cette tendance semble assez stable 
        depuis le milieu le début des années 60.

        Les bons films sont quant à eux plus longs de vingt minutes supplémentaires en moyenne. 
        Un bon film serait, en moyenne, plutot un film long. 
        """
    )

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
                     'Age': 'deepskyblue',
                     'Filtrés': 'coral'})
    fig.update_yaxes(title=None, nticks=10)
    fig.update_xaxes(title=None, nticks=12)
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                      margin=dict(l=30, r=30, b=30))

    if show:
        col1, col2 = st.beta_columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.title(' ')
            st.title(' ')
            st.dataframe(age_chart)
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        Enfin, **l’âge des acteurs ne semble pas être un critère discriminant entre les bons et les mauvais films**.

        En revanche, on peut noter une augmentation de la moyenne d’âge des acteurs à partir des années 2010, 
        qui **passe de 37 ans à 44 ans, soit 7 ans en plus**. 
        """
    )

    st.markdown('---')

    st.subheader('Une étude des genres : un _Drama_ en 2 actes.')
    st.markdown(
        """
        En conclusion, on va faire un focus sur les genres des films tels qu’ils sont défini par IMDb. 
        Le site a donné un descriptif de chaque genre sur [ce lien]
        (https://help.imdb.com/article/contribution/titles/genres/GZDRMS6R742JRGAG#).

        IMDb se limite à **attribuer de 0 à 3 genres par films** 
        pour les définir sur un **total de 27 genres différents**.
        """
    )

    temp_tab = data.copy()
    temp_tab['Genres'] = temp_tab['Genres'].apply(lambda x: x.split(','))
    temp_tab = temp_tab.explode('Genres')
    Best_genres = temp_tab.pivot_table(values='Votes', index=['Genres'], aggfunc='count')
    Best_genres = Best_genres.sort_values('Votes', ascending=False).drop(index=['\\N', 'Adult', 'Reality-TV', 'Short'])

    fig = go.Figure(data=[go.Scatter(
        x=list(Best_genres.index), y=[1] * 27,
        text=Best_genres['Votes'],
        mode='markers',
        marker=dict(
            size=Best_genres['Votes'],
            color=Best_genres['Votes'],
            sizemode='area',
            sizeref=2,
            showscale=True,
            colorscale='Peach'
        )
    )])
    fig.update_layout(showlegend=False, font_family='IBM Plex Sans',
                      title='<b>Quels sont les genres dominants ?</b> Entre rire et pleurer, ils ont choisi.',
                      uniformtext_minsize=14, uniformtext_mode='hide',
                      margin=dict(l=10, r=10, b=10),
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(title=None, showticklabels=False)

    if show:
        col1, col2 = st.beta_columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.title(' ')
            st.dataframe(Best_genres)
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        On note tout de suite que le genre le plus présent dans la base de données est **_Drama_, 
        qui ressort dans un peu moins de 30% des films**, suivi de _Comédie_.

        Il est toutefois victime de son succès, puisque selon que l’on cherche par vote moyen ou par nombre 
        de votre, le genre _Drama_ tombe largement das le classement.
        
        A ce jeu-là, le genre _Sci_Fi_ devient le grand vainqueur de la popularité alors que les genres 
        qui se collent à la réalité (_Documentaires_ et _Biographies_) reçoivent  la meilleure moyenne. 
        """
    )

    rate_genres = temp_tab.pivot_table(values='Note', index=['Genres'], aggfunc='mean')
    rate_genres = rate_genres.sort_values('Note', ascending = False).drop(index=['\\N', 'Adult', 'Reality-TV', 'Short'])

    pop_genres = temp_tab.pivot_table(values='Votes', index=['Genres'], aggfunc='mean')
    pop_genres = pop_genres.sort_values('Votes',ascending=False).drop(index=['\\N', 'Adult', 'Reality-TV', 'Short'])

    color_rate = ['deepskyblue']*15
    color_rate[9] = 'coral'
    color_pop = ['deepskyblue']*15
    color_pop[13] = 'coral'
    fig_rate = go.Figure(go.Bar(
        x=rate_genres['Note'][:15],
        y=rate_genres.index[:15],
        orientation='h',
        marker_color=color_rate,
        text=rate_genres['Note'][:15].round(2),
        textposition='auto'))
    fig_rate.update_layout(margin=dict(l=30, r=10, b=30, t=30))
    fig_rate.update_yaxes(autorange="reversed", side='left')
    fig_rate.update_xaxes(title='Note moyenne par catégorie de films')

    fig_pop = go.Figure(go.Bar(
        x=pop_genres['Votes'][:15],
        y=pop_genres.index[:15],
        orientation='h',
        marker_color=color_pop,
        text=pop_genres['Votes'][:15].round(0),
        textposition='auto'))
    fig_pop.update_layout(margin=dict(l=10, r=30, b=30, t=30))
    fig_pop.update_yaxes(autorange="reversed", side='right')
    fig_pop.update_xaxes(autorange="reversed",
                         title='Nombre de votes moyen par catégorie de films')

    col1, col2 = st.beta_columns(2)
    with col1:
        st.plotly_chart(fig_rate, use_container_width=True)
        if show:
            st.dataframe(rate_genres)
    with col2:
        st.plotly_chart(fig_pop, use_container_width=True)
        if show:
            st.dataframe(pop_genres)

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
        fig.update_traces(textfont_size=20, marker=dict(colors=['deepskyblue', 'coral'],
                                                        line=dict(color='#000000', width=1)))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        Si la proportion d'actrices dans la base de données est inférieure à 40%, la situation est évidement la même
        sur dans la distribution des roles. **Les femmes représent généralement moins de 35% des personnes à l'écran**,
        même si on note une légère tendence vers l'équilibre à partir des années 90.
    """)

    # proportion d'actrices
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
            color='deepskyblue',
            line=dict(color='deepskyblue', width=0.05)
        )
    ))
    fig.add_trace(go.Bar(
        y=percent["Actrice"],
        x=percent.index,
        name="Actrice",
        marker=dict(
            color='coral',
            line=dict(color='coral', width=0.05)
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
        title="<b>Proportion d'actrices présente sur le grand écran par décennies</b>",
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
        colorway=['deepskyblue', 'coral', 'lightgreen'],
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
                                 "min_vote": 3000,
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

    try:
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

            final = pd.concat([pd.DataFrame(per_mov.mean()).T, pd.DataFrame(top_per_mov.mean()).T])
            final = final.reset_index().iloc[:, 1:].rename(index={0: 'Global', 1: 'Filtré'})

            fig = px.bar(final, x=final.index, y=["Proportion Actrices", "Proportion Acteurs"], barmode='group',
                         title="<b>Proportion moyenne d'actrices dans les films</b>",
                         color_discrete_map={'Proportion Acteurs': 'deepskyblue', 'Proportion Actrices': 'coral'})
            fig.update_yaxes(title=None, showticklabels=False, range=[0, 90], showgrid=False)
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
                et de mettre en avant l’aggravation des inégalités 
                dès lors qu’on se rapproche des films les plus populaires.
        
                La popularité se ressent aussi au niveau du nombre de votes 
                qui est inférieur dès lors que l'actrice tient le role principal.
        
                Pour un premier role, on constate en moyenne **{round(nb_vote_W) if nb_vote_W > 0 else 0 } votes 
                pour une actrice**, contre **{round(nb_votes_M)} votes pour un acteur**, 
                soit un **nombre de votre inférieur de 
                {round(100 - ((nb_vote_W * 100) / nb_votes_M)) if nb_vote_W > 0 else 100 } %**.
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
            fig.update_traces(textfont_size=20, marker=dict(colors=['deepskyblue', 'coral'],
                                                            line=dict(color='#000000', width=1)))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            raw_order = pd.DataFrame(best_actors[(best_actors['category'].isin(['actress'])) &
                                                 (best_actors['Ordre'] < 5)].value_counts('Ordre',
                                                                                          normalize=True).sort_index())
            raw_order.rename(columns={0: 'Actrices'}, inplace=True)
            raw_order['Acteurs'] = best_actors[(best_actors['category'].isin(['actor'])) &
                                               (best_actors['Ordre'] < 5)].value_counts('Ordre',
                                                                                        normalize=True).sort_index()
            raw_order = raw_order.apply(lambda x: x * 100).round(2)

            fig = px.bar(raw_order, x=raw_order.index, y=["Actrices", "Acteurs"], barmode='group',
                         title="<b>A quel rang se trouve les actrices et acteurs ?</b>",
                         color_discrete_map={'Acteurs': 'deepskyblue', 'Actrices': 'coral'})
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
            st.dataframe(best_actors[(best_actors['category'] == 'actress') & (best_actors['Ordre'] == 1)])
            st.markdown('Le role principal est tenu par un acteur : ')
            st.dataframe(best_actors[(best_actors['category'] == 'actor') & (best_actors['Ordre'] == 1)])
    except KeyError:
        st.error("Cette configuration ne fait plus apparaitre d'actrices dans le classement. "
                 "Les rôles ne sont tenus que par des hommes.")


elif categorie == 'Les TOP par décennies':
    st.title('Les TOP par décennies')

    st.write("""
    Classement selon les données présentes sur la platforme **imbd**
    """)

    data = load_data(REPRO_DB)

    st.title(' ')
    col1, col2 = st.beta_columns([2, 1])
    with col2:
        TOP_choice = st.radio("Type de TOP",
                                     ('Par décennies', 'Par Année', 'Toutes années confondues'))
        select_genres = st.checkbox('Selectionner un genre de film')
        expander = st.beta_expander("indice MORE ?")
        expander.markdown("**L'indice MORE** permet de lier le nombre de votes et la note d'un film "
                          "pour en retirer un indice de popularité. Plus l'indice est élevée (de 1 à 40) "
                          "plus le film est populaire. "
                          "Pour les autres catégories que les films, c'est la moyenne des indices sur la décennie "
                          "selectionnée qui est calculée."
                          )

    with col1:
        if TOP_choice == 'Par décennies':
            start_year = st.slider('Décennie', 1920, 2020, 1990, 10)
            stop_year = start_year + 10
        if TOP_choice == 'Par Année':
            start_year = st.slider('Décennie', 1920, 2020, 1990, 1)
            stop_year = start_year + 1
        if TOP_choice == 'Toutes années confondues':
            start_year = 1920
            stop_year = 2020
        if select_genres:
            list_selected = st.multiselect('Quel genre de films souhaitez-vous selectionner ?',
                                           ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                                            'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
                                            'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi',
                                            'Sport', 'Thriller', 'War', 'Western'],
                                           help='Il faut selectionner 1 à 3 genres différents maximum')
            list_selected.sort()
            genre_selected = ','.join(list_selected)
        else:
            st.title(' ')
        filtre = st.selectbox(
            'Vous souhaitez filtrer par :',
            ['indice MORE', 'Votes', 'Total', 'Note'])

    filtered_data = data[(data['Année'] >= start_year) & (data['Année'] < stop_year)]
    if select_genres:
        filtered_data = filtered_data[filtered_data['Genres'] == genre_selected]

    if not select_genres or 0 < len(list_selected) < 4:
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
    else:
        st.warning('Merci de selectionner le genre  de film de votre selection. '
                   'Il faut selectionner au **minimum 1** genre et au **maximum 3**')


elif categorie == 'Quoi voir ?':
    if sub_categorie == 'Recommandation de films':
        st.title('Recommandation de films')
        st.subheader('Laissez vous seduire par la magie du Machine Learning')

        # data
        ml_db = load_data(ML_DB).sort_values('indice MORE', ascending=False).reset_index(drop=True)

        st.markdown(
            f"""
            Cet outil permet d'utiliser toute la puissance du *Machine Learning* pour vous proposer des films qui sont
            le plus en proches de vos goûts.
    
            Afin de de vous faire une proposition, nous vous invitons à selectionner un de vos films favoris pour que 
            l'outil MORE puisse l'analyser et vous proposer des films proches
            parmis **une selection de {ml_db.index[-1]} films**.
            """
        )

        with st.form(key='my_form'):
            movie_selected = st.selectbox('Choisissez votre film :', ml_db['Titre'])
            speed = st.checkbox('Activer le FastML')
            st.markdown('Le mode _Fast Machine Learning_ permet de réduire fortement le temps de calcul, '
                        'mais impacte la précision des recommandations.')
            submit = st.form_submit_button(label='Rechercher')

        if submit:
            # recommandation genre
            X = ml_data(ml_db)
            y = ml_db['Note'].round(0)

            if speed:
                pca = PCA(n_components=0.60).fit(X)
                X = pca.transform(X)

            modelMORE = KNeighborsClassifier(weights='distance', n_neighbors=5).fit(X, y)
            reco = pd.DataFrame(data=modelMORE.kneighbors(X, return_distance=False))
            reco = reco.loc[ml_db[ml_db['Titre'] == movie_selected].index]

            st.subheader(f'_Parce que vous appreciez **{movie_selected}**_')
            cols = st.beta_columns(4)
            for i, col in enumerate(cols):
                index_mov = ml_db[ml_db.index == reco.iloc[0, i+1]][['tconst', 'Titre']]
                col.subheader(index_mov.iloc[0, 1])
                col.image(picture(index_mov))

            st.markdown('---')

            # recommendation cast
            for cast_rang in range(8, 11):
                search = ml_db[ml_db['Titre'] == movie_selected]
                fil_data = ml_db[(ml_db['main_role'] == search.iloc[0, cast_rang]) |
                                  (ml_db['second_role'] == search.iloc[0, cast_rang]) |
                                  (ml_db['third_role'] == search.iloc[0, cast_rang])].sort_values('indice MORE').reset_index()

                if fil_data.shape[0] > 2:
                    X = ml_data(fil_data)
                    y = fil_data['Note'].round(0)

                    modelMORE = KNeighborsClassifier(weights='distance',
                                                     n_neighbors=fil_data.shape[0] if fil_data.shape[0] < 5else 5).fit(X, y)

                    new_row = fil_data[fil_data['Titre'] == movie_selected]
                    reco = pd.DataFrame(data=modelMORE.kneighbors(X, return_distance=False)).iloc[new_row.index[0]]

                    st.subheader(f'_Parce que vous appreciez **{search.iloc[0, cast_rang]}**_')
                    cols = st.beta_columns(fil_data.shape[0] - 1 if fil_data.shape[0] < 5 else 4)
                    for i, col in enumerate(cols):
                        index_mov = fil_data[fil_data.index == reco.iloc[i+1]][['tconst', 'Titre']]
                        col.subheader(index_mov.iloc[0, 1])
                        col.image(picture(index_mov))

                    st.markdown('---')

            # recommendation director
            search = ml_db[ml_db['Titre'] == movie_selected]
            fil_data = ml_db[ml_db['director'] == search.iloc[0, 7]].sort_values('indice MORE').reset_index(drop=True)

            if fil_data.shape[0] > 2:
                X = ml_data(fil_data)
                y = fil_data['Note'].round(0)
                modelMORE = KNeighborsClassifier(weights='distance',
                                                 n_neighbors=fil_data.shape[0] if fil_data.shape[0] < 5 else 5).fit(X, y)

                new_row = fil_data[fil_data['Titre'] == movie_selected]
                reco = pd.DataFrame(data=modelMORE.kneighbors(X, return_distance=False)).iloc[new_row.index[0]]

                st.subheader(f'_Parce que vous appreciez **{search.iloc[0, 7]}** à la réalisation_')
                cols = st.beta_columns(fil_data.shape[0]-1 if fil_data.shape[0] < 5 else 4)
                for i, col in enumerate(cols):
                    index_mov = fil_data[fil_data.reset_index().index == reco.iloc[i+1]][['tconst', 'Titre']]
                    col.subheader(index_mov.iloc[0, 1])
                    col.image(picture(index_mov))


    if sub_categorie == 'Restrospectives':
        st.title('Restrospectives')
        st.subheader("Organisez des rétrospectives à l'aide de l'**Intelligence Artificielle**.")
        st.markdown(
            f"""
            Idée de rétro ? 
            
            Au boulot !
            """
        )

        with st.form(key='retro'):
            movie = st.selectbox('Choisissez votre film :', ml_db['Titre'])
            retro = st.selectbox('Choisissez le type de rétrospective :', ['réalisateur', 'décennie'])
            submit = st.form_submit_button(label='Rechercher')

        if submit:
            min_pop, max_pop = (3000, tick_max)
            min_rate, max_rate = (7, 10)

            retro_db = ml_db.copy()
            best_mov = fav_filter(ml_db)
            best_mov['bm'] = 1
            total_mov = pd.merge(retro_db, best_mov[['tconst', 'bm']], how="left", on=["tconst"]).fillna(0)

            X = ml_rating(total_mov)
            y = total_mov["bm"]
            modeleLR = LogisticRegression(class_weight="balanced").fit(X, y)

            Xpred = modeleLR.predict_proba(X)
            retro_db["probm"] = Xpred[:, 1]
            mov_RL = retro_db.sort_values("probm", ascending=False).reset_index(drop=True)

            # class retrospective
            movies_selection = Retrospective(film=movie, type_retro=retro).propo_retro()
            st.subheader('Pour votre rétospective, nous vous proposons les films suivants : ')
            cols = st.beta_columns(len(movies_selection))
            for i, col in enumerate(cols):
                index_mov = ml_db[ml_db['tconst'] == movies_selection.iloc[i, 0]][['tconst', 'Titre']]
                col.subheader(index_mov.iloc[0, 1])
                col.image(picture(index_mov))

            if show:
                st.title(' ')
                st.dataframe(movies_selection)


    if sub_categorie == 'Probabilité que vous aimiez ce film':
        st.title('Est-ce que ce film va me plaire ?')
        st.subheader('Ne perdez plus votre temps avec des films qui ne vous correspondent pas')
        st.markdown(
            f"""
            Vous hésitez entre deux films ? Ne laissez plus rien au hasard et laissez-vous conseiller par 
            l'_Intelligence Artificielle_ et vous n’aurez plus qu’à chercher le popcorn. 

            Afin d’initialiser l’outil, nous vous invitons à créer votre profil en **indiquant les films 
            que vous avez appréciez**. Il est conseillé d’indiquer 3 films minimum. 

            Il vous suffit ensuite d’**indiquer le film que vous souhaitez voir** et l’algorithme vous indiquera alors 
            quel est votre **degré de comptabilité**.
            """
        )

        with st.form(key='proba'):
            user_profil = st.multiselect('Constituer votre profil : Quels sont les films que vous aimez ?',
                                         ml_db['Titre'],
                                         help='Il est conseillé de selectionner au moins 3 films. '
                                              'Plus vous selectionnez de films plus la recommandation sera pertinante')
            movie_selected = st.selectbox('Choisissez le film que vous souhaitez voir :', ml_db['Titre'])
            submit = st.form_submit_button(label='Rechercher')

        if submit:
            predict_data = ml_db.copy()
            predict_data['user_choice'] = 0
            for movie in user_profil:
                id_mov = predict_data[predict_data['Titre'] == movie].index
                predict_data.iloc[id_mov, 12] = 1

            # data
            temp = predict_data.copy()
            temp['director'] = temp['director'].factorize()[0]
            temp['main_role'] = temp['main_role'].factorize()[0]
            temp['second_role'] = temp['second_role'].factorize()[0]
            temp['third_role'] = temp['third_role'].factorize()[0]
            temp['Genres'] = temp['Genres'].apply(lambda x: x.split(','))
            temp['indice_MORE'] = ((temp['Note'] * temp['Votes']) / (temp['Votes'].sum()) * 1000000000).apply(math.sqrt).apply(math.sqrt).apply(lambda x: round(x, 2))
            temp = temp[['user_choice', 'director', 'main_role', 'second_role', 'third_role',
                         'Note', 'Votes', 'Année', 'Genres']]
            genres = temp.explode('Genres')['Genres'].str.get_dummies()
            db = pd.concat([temp, genres.groupby(genres.index).agg('sum')], axis=1).drop(columns=['Genres'])
            power = PowerTransformer().fit(db.drop(columns='user_choice'))

            #model
            X = power.transform(db.drop(columns='user_choice'))
            y = db['user_choice']
            model = LogisticRegression(max_iter=1000, class_weight={1: 10 ** 17, 0: 1}).fit(X, y)

            prediction = model.predict_proba(X)
            predict_data["proba"] = (prediction[:, 1] * 100).round(2)

            st.title(' ')
            col1, col2 = st.beta_columns([2, 1])
            with col1:
                col1.title(' ')
                col1.subheader(f"Vous avez **{predict_data[predict_data['Titre'] == movie_selected].iloc[0,13]} %** "
                               f"de chances d'apprecier le film **{movie_selected}**")
            with col2:
                col2.image(picture(predict_data[predict_data['Titre'] == movie_selected]))

            if show :
                st.title(' ')
                st.dataframe(predict_data.sort_values('proba', ascending=False).reset_index())
