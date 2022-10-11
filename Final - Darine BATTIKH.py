#!/usr/bin/env python
# coding: utf-8

# 
# # Projet: Investiguer la Base de Données des Films "The Movies Database"
# 
# ## Table des Matières
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Traitement des Données</a></li>
# <li><a href="#eda">Analyse Exploratoire des Données</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# L'ensemble de données sur les films contient des informations sur 10 000 films recueillies auprès de The Movie Database (TMDb). Cet ensemble de données comprend les avis des utilisateurs, les revenus, les acteurs, le genre, le directeur et la boîte de production comme principales variables à étudier.
# 
# Compte tenu de tout cela, la principale question à se poser est la suivante : **Quels sont les critères de succès d'un film?**
# 
# Comme il n'existe pas de définition précise d'un film à succès, nous pouvons citer deux variables cruciales à cet égard : 
# 1. Le film est un succès commercial en termes de profits.
# 2. le film est hautement noté par le public - Populairité. 
# 
# Pour cela, nous allons étudier les différents critères qui font le succès d'un film du point de vue commercial et du point de vue du public. 
# 
# Nous pouvons décomposer cette question générale en questions spécifiques afin d'orienter notre analyse et d'obtenir la réponse précise:
# 1. Success Commerciale: Quels critères sont associés aux films qui ont des revenus élevés? 
# 2. Popularité: Quels critères sont associés aux films qui ont des votes élevés? 
# 3. Un film populaire signifie-t-il directement un film commercialement réussi?
# 
# Enfin, il est particulièrement intéressant d'étudier l'évolution des bénéfices au fil des ans.
# 
# La conclusion de toutes les analyses sera détaillée à la fin de ce rapport et répond à la question principale.

# ## Importation des Paquets et La Base de Données
# Dans cette premiere section du rapport, nous allons charger les pqauets necessaires aisni que la base de données des films.

# In[1]:


# Importing necessary packages
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


#Importing the database we will work on
df = pd.read_csv('tmdb-movies.csv')


# <a id='wrangling'></a>
# ## Traitement des Données
# Dans cette section du rapport, nous allons charger les données, vérifier leur propreté, puis découper et nettoyer notre ensemble de données pour l'analyse. 
# ### Propriétés Générales

# In[2]:


# Load the database
df.head()


# In[3]:


print(df.shape)
df.info()


# #### Insight
# Les informations sur les données indiquent que certaines colonnes ne seront pas utilisées dans notre analyse car elles ne sont pas directement liées à notre problématique, et que certaines valeurs manquantes doivent être corrigées pour garantir une analyse précise. 
# 
# Nous vérifierons également les lignes dupliquées et les types de données inexactes pour les corriger.

# ### Data Cleaning

# In[4]:


# Dropping unecessary columns
df.drop(['id', 'imdb_id','homepage','tagline', 'overview',  'keywords'], axis=1, inplace=True)
df.info()


# In[5]:


# Checking and removing duplicated rows
sum(df.duplicated())


# In[6]:


df.drop_duplicates(inplace=True)


# In[7]:


sum(df.duplicated())


# In[8]:


# Checking and fixing missing values
df.isnull().sum()


# In[9]:


df.fillna(0, inplace = True)


# In[10]:


df.isnull().sum()


# In[11]:


# Correcting the release date data type
df['release_date'] = pd.to_datetime(df['release_date'])


# Maintenant que nous avons nettoyé notre base de données des colonnes non essentielles, des valeurs manquantes, des lignes redondantes et des types de données erronés, nous allons créer trois nouvelles colonnes:
# 
# 
# 1. **profit**: ou la colonne de bénéfices: cette colonne est nécessaire pour comparer les bénéfices de chaque film afin de pouvoir extraire le critère du succès commercial (bénéfices élevés)
# 
# 
# 2. **profit_rate**: ou le taux_de_profits : Même si les bénéfices peuvent donner un aperçu des films qui font des profits, le taux de profits sera un indicateur plus précis car il souligne à quel point un film multiplie ses revenus par rapport au budget investi. 
# 
# 
# 3. **release_month**: dans cette colonne, nous ne considérerons que le mois car le mois de sortie peut affecter le succès du film. Pour cela, nous allons supprimer la colonne release_date. 

# In[12]:


df['profits'] = df['revenue_adj'] - df['budget_adj']
df['profits_rate'] = df['revenue_adj'] / df['budget_adj']
df.drop(['revenue_adj', 'budget_adj'], axis=1, inplace=True)


# In[13]:


df['release_month'] = df['release_date'].dt.month
df.drop(['release_date'], axis=1, inplace=True)


# In[14]:


print(df.shape)
df.head()


# <a id='eda'></a>
# ## Analyse Exploratoire des Données
# Dans cette section, nous analyserons nos données pour répondre à nos principales questions en détail. À la fin de cette section, nous comprendrons les facteurs critiques d'un succès commercial et d'un film populaire (point de vue du public). 
# 
# Une analyse de corrélation sera effectuée pour découvrir la dépendance entre les deux principales variables en question : le succès commercial et la popularité d'un film. En d'autres termes, nous aimerions savoir si un film très bien noté fera plus de bénéfices. 
# 
# 
# ### Question de Recherche 1: Success Commerciale: Quels critères sont associés aux films qui ont des revenus élevés?
# 
# #### Critère 1: Genres
# Nous allons étudier si le genre d'un film est un facteur de succès commercial. En d'autres termes, nous chercherons à savoir s'il existe un ou plusieurs genres spécifiques d'un film qui contribuent à faire plus de profits. Si ce n'est pas le cas, alors il existe d'autres facteurs de succès commercial différents des genres.
# Nous étudierons deux échantillons : un échantillon de top 50 films et un second échantillon de top 1000 films. 
# 

# In[48]:


def stat(sortBy, headCount, column):
    df_stat = df.sort_values(by=sortBy, ascending = False).head(headCount)
    df_stat.loc[:, column].head()
    df_stat[column] = df_stat[column].astype(str)
    df_stat_sep = df_stat[column].str.cat(sep='|')
    df_stat_sep = pd.Series(df_stat_sep.split('|'))
    df_stat_sep = df_stat_sep.value_counts(ascending = False).to_frame()
    return df_stat_sep

data = stat('profits', 50, 'genres')
data.head()


# In[49]:


data = stat('profits', 1000, 'genres')
data.head()


# #### Insight
# Sur la base des bénéfices et de deux échantillons différents, nous n'avons pas pu prendre une décision claire sur les genres qui contribuent à des bénéfices élevés, car lorsque la taille de l'échantillon a changé, les valeurs ont changé, ce qui est logique. Pour cela, nous utiliserons le taux de profits pour une étude plus approfondie. 

# In[50]:


data = stat('profits_rate', 50, 'genres')
data.head()


# In[53]:


data = stat('profits_rate', 1000, 'genres')
data.head()


# #### Insight
# L'utilisation du taux de profits donne de meilleurs résultats car il met l'accent sur les profits, c'est-à-dire sur les revenus qu'un film génère par rapport au budget investi. 
# 
# Cette analyse montre que les genres Drame et Comédie contribuent le plus à la réalisation de bénéfices, et donc à un succès commercial. 

# #### Critère 2: Cast
# Nous allons étudier si le cast d'un film est un facteur de succès commercial. En d'autres termes, nous chercherons à savoir s'il un acteur ou plusieurs acteurs spécifiques contribuent à faire plus de profits. 
# data = stat('profits_rate', 1000, 'genres')
# data.head()
# 
# Nous étudierons deux échantillons : un échantillon de top 50 films et un second échantillon de top 1000 films. 

# In[56]:


data = stat('profits', 50, 'cast')
data.head()


# In[59]:


data = stat('profits', 1000, 'cast')
data.head()


# In[61]:


data = stat('profits_rate', 50, 'cast')
data.head()


# In[63]:


data = stat('profits_rate', 1000, 'cast')
data.head()


# #### Insight
# Sur la base de cette analyse, nous pouvons conclure deux points majeurs: 
# 1. Si l'on considère quels sont les acteurs qui contribuent à faire plus de recettes, il s'agirait de Tom Cruise et Tom Hanks, ce qui est logique puisqu'ils sont principalement choisis pour leur notoriété. 
# 
# 2. Si l'on considère les films qui font plus de recettes tout en investissant moins de budget, ce qui signifie un taux de profit plus élevé, ce sont d'autres acteurs comme Clint Eastwood et John Cusack. Dans le même temps, Tom Cruise et Tom Hanks n'apparaissent pas dans la liste des 10 acteurs ayant les taux de profit les plus élevés. Cela peut refléter le fait que ces deux acteurs sont payés très cher et qu'ils contribuent effectivement à la réalisation de bénéfices, mais pas au taux de profit.

# #### Critère 3: Production Company
# Nous chercherons à savoir si la société de production peut être un facteur de succès commercial pour les réalisateurs de films. Y a-t-il vraiment une différence dans les revenus lorsqu'on travaille avec une société de production et pas une autre? 

# In[65]:


data = stat('profits', 50, 'production_companies')
data.head()


# In[66]:


data = stat('profits', 1000, 'production_companies')
data.head()


# In[68]:


data = stat('profits_rate', 50, 'production_companies')
data.head()


# In[69]:


data = stat('profits_rate', 1000, 'production_companies')
data.head()


# #### Insight
# Qu'il s'agisse de profits purs ou de taux de profits, il y a trois grandes sociétés de production qui contribuent le plus aux revenus élevés des films : Warner Bros, Universal Pictures et Paramount Pictures. 
# Par conséquent, travailler avec l'une de ces sociétés garantira les profits les plus élevés.

# #### Critère 4: Release Month
# Nous chercherons à savoir si le mois de diffusion de film peut être un facteur de succès commercial pour les réalisateurs de films. Y a-t-il vraiment une différence dans les revenus lorsqu'on diffuse le film dans un mois et pas un autre?

# In[70]:


data = stat('profits', 50, 'release_month')
data.head()


# In[71]:


data = stat('profits_rate', 100, 'release_month')
data.head()


# #### Insight
# Le 6e mois, qui est Juin, s'est avéré être le mois le plus rentable pour la diffusion des films, ce qui est logique puisque c'est le début des vacances d'été où les gens souhaitent profiter au maximum de leur temps.

# ### Question de Recherche 2: Popularité: Quels critères sont associés aux films qui ont des votes élevés?
# Dans cette section, nous allons d'abord étudier la corrélation entre la popularité et les films bien notés. En d'autres termes, nous aimerions savoir si un film populaire est nécessairement bien noté par le public.
# 
# Ensuite, nous étudierons les facteurs qui motivent le public à donner de bonnes critiques à un film. Cela aidera les réalisateurs de films à se concentrer sur ces facteurs pour obtenir l'appréciation du public.

# In[29]:


df_loved_movies = df
df_loved_movies.drop(df_loved_movies[df_loved_movies['vote_count'] < 1000].index, inplace=True)
df_loved_movies = df_loved_movies.sort_values(by='vote_average', ascending = False).head(1000)
df_loved_movies.drop(df_loved_movies[df_loved_movies['production_companies'] == 'nan'].index, inplace=True)


# In[30]:


print(df_loved_movies.shape)
df_loved_movies.head()


# #### Correlation between Popularity and Reviews

# In[72]:


# Defining our plotting function that will be used for all plots
def plotting (title, x, y, x_label, y_label):
    plt.scatter(x, y, alpha = 0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    
plotting ('Correlation Between Average Vote and Popularity',df_loved_movies['vote_average'], df_loved_movies['popularity'],"Average Vote","Popularity")


# #### Insight
# Sur la base du Scratter plot ci-dessus, nous pouvons conclure qu'il n'y a pas de corrélation entre la popularité et les notes moyennes. Cela nous amène à conclure que le vote moyen du public n'affecte pas la popularité du film et que ce sont des variables indépendantes.
# 
# Ceci étant dit, nous pouvons travailler sur les avis des utilisateurs et détecter les facteurs qui conduisent à un plus grand nombre d'avis des utilisateurs.

# #### Criteria 1: Production Companies

# In[32]:


df_loved_movies['production_companies'] = df_loved_movies['production_companies'].astype(str)
df_loved_movies_companies = df_loved_movies['production_companies'].str.cat(sep='|')
companies = pd.Series(df_loved_movies_companies.split('|'))
count_loved_movies_campanies = companies.value_counts(ascending = False).to_frame()
count_loved_movies_campanies.head(10)


# #### Insight
# Les films réalisés par Warner Bros. sont remarquablement mieux notés par le public que les autres films.

# #### Criteria 2: Cast

# In[33]:


df_loved_movies['cast'] = df_loved_movies['cast'].astype(str)
df_loved_movies_companies = df_loved_movies['cast'].str.cat(sep='|')
companies = pd.Series(df_loved_movies_companies.split('|'))
count_loved_movies_campanies = companies.value_counts(ascending = False).to_frame()
count_loved_movies_campanies.head(50)


# #### Insight
# Sur la base de cette analyse, nous pouvons conclure qu'il n'y a pas de grande différence dans le score en fonction des acteurs. Par conséquent, le casting n'est pas un facteur de popularité du film du point de vue du public.

# #### Criteria 3: Genres

# In[34]:


df_loved_movies['genres'] = df_loved_movies['genres'].astype(str)
df_loved_movies_companies = df_loved_movies['genres'].str.cat(sep='|')
companies = pd.Series(df_loved_movies_companies.split('|'))
count_loved_movies_campanies = companies.value_counts(ascending = False).to_frame()
count_loved_movies_campanies.head(10)


# #### Insight
# Sur la base de cet aperçu, nous pouvons conclure que les films d'action et d'aventure sont les plus appréciés par le public.

# ### Research Question 3: Un film populaire signifie-t-il directement un film commercialement réussi?

# In[35]:


df.corr()


# In[44]:


plotting ('Correlation Between Profits Rate and Popularity',df_loved_movies['profits'], df_loved_movies['popularity'],"Average Vote","Popularity")


# 

# In[73]:


plotting ('Correlation Between Profits Rate and Popularity',df_loved_movies['profits_rate'], df_loved_movies['popularity'],"Profits","Popularity")


# In[74]:


plotting ('Correlation Between Profits and Average Vote',df_loved_movies['vote_average'], df_loved_movies['profits'],"Average Vote","Profits")


# In[75]:


plotting ('Correlation Between Profits Rate and Average Vote',df_loved_movies['vote_average'], df_loved_movies['profits_rate'],"Average Vote","Profits Rate")


# #### Insight
# Le succès commercial ( plus de bénéfices) s'est avéré indépendant de la moyenne des votes et de la popularité du film.

# ### Evolution du Bénéfice par Année

# In[40]:


df['prft'] = df['revenue'] - df['budget']
plt.figure(figsize=(10,8))
sns.lineplot(x = df['release_year'], y = df['prft'])
plt.title('Evolution du Bénéfice par Année')
plt.xlabel("Year")
plt.ylabel("Bénéfice")
plt.xlim(1955,2025)
plt.show()


# #### Insight
# L'évolution des bénéfices au fil des ans ne suit pas un schéma clair : elle est marquée par de nombreux hauts et bas, avec un pic significatif dans les années 70. Pour comprendre cette tendance, nous devons soit examiner en profondeur l'histoire de l'industrie cinématographique, soit étendre nos recherches et essayer de comprendre quels sont les facteurs qui ont contribué à ce pic à cette période.

# <a id='conclusions'></a>
# ## Conclusions
# 
# Tout au long de notre travail, nous avons conservé deux définitions principales du **succès**: Le succès commercial (bénéfices) et la popularité ( avis élevés des utilisateurs). Sur la base de ces deux définitions, nous avons travaillé à la détection des facteurs contribuant au succès global d'un film. 
# 
# L'analyse a montré que :
# 
# 1. **Succès commercial** : Les critères affectant le succès commercial d'un film sont principalement le genre du film et la société de production. Par exemple, les genres **Drame et Comédie** ont prouvé qu'ils généraient des taux de profit significativement plus élevés que les autres genres. De même, le bénéfice dépend de la société de production, puisque **Warner Bros., Universal Pictures et Paramount Pictures** sont les sociétés de production qui produisent les films les plus rentables. 
# En plus, le 6ème mois s'est avéré être le mois le plus rentable pour la sortie des films, les producteurs doivent donc en tenir compte pour viser des bénéfices plus substantiels.
# En ce qui concerne le facteur "cast", il n'y a pas eu d'acteur ou de casting particulier pour contribuer à faire plus de profits. Pourtant, gagner plus d'argent dans un film dépend du salaire de chaque acteur. En d'autres termes, l'intégration d'une figure célèbre dans un film coûtera plus cher que celle d'une figure moins célèbre, et rapportera sûrement plus de recettes, mais en ce qui concerne les bénéfices, il est plus efficace de travailler avec une figure moins célèbre pour que le taux de profit soit plus élevé. 
# 
# 
# 2. **Popularité** : La popularité et les avis des utilisateurs moyens se sont avérés être indépendants. Pour cela, nous reflétons la popularité par des films très bien notés par le public. Les facteurs qui affectent les avis des utilisateurs sont les genres des films et les sociétés de production. Sur la base de notre analyse, nous concluons que le fait de travailler avec **Warner Bros** contribuera de manière significative à l'obtention de meilleures critiques. De même, les genres **Action et Aventure** sont les plus appréciés par le public. Le casting n'a pas été présenté comme un facteur important contribuant à des critiques plus élevées.
# Sur cette base, nous pouvons conclure que pour faire un film plaisant pour le public, nous devons nous concentrer sur la société de production et les genres.
# 
# 
# 3. Comme nous voulions démontrer une **relation entre la popularité (les critiques des utilisateurs) et le succès commercial d'un film**, la corrélation était proche de zéro, ce qui prouve qu'un film très bien noté ne signifie pas nécessairement un succès commercial et vice versa. 
# 
# 
# 4. **Evolution des bénéfice au fil des ans**:  L'évolution du bénéfice au cours de l'année ne présente pas un schéma clair : elle comporte de nombreux hauts et bas avec quelques pics. Nous pouvons étendre notre recherche pour étudier en profondeur les facteurs qui contribuent à ce modèle instable et déterminer comment optimiser les profits.
# 
# En conclusion, dans l'industrie cinématographique, les réalisateurs s'efforcent de faire le plus de profits possible et d'attirer le public. Pour cela, afin de réaliser un film populaire et à succès sur le plan commercial, ils doivent se concentrer sur deux facteurs principaux : **les genres et les sociétés de production**. 
# 
# Les acteurs peuvent être un facteur important pour le film et peuvent générer plus de revenus, mais ils ne contribuent pas à générer plus de bénéfices (car pour générer plus de revenus, il faut plus de budget dans le cas des acteurs). Pour cela, le choix du cast dépend vraiment de la perspective du décideur.
# 
# ### Limitations
# Bien que ces données aient contribué à la compréhension de nombreux facteurs et relations, il leur manque encore certaines données qui pourraient rendre notre analyse et notre interprétation plus précises, comme l'importante quantité de données manquantes trouvées dans cette base de données. Plus important encore, il n'y avait pas de données claires pouvant être utilisées pour recueillir des informations sur l'évolution instable des bénéfices au fil des ans. Ce type de données aurait beaucoup aidé à conclure les facteurs et critères importants sur lesquels se concentrer dans l'industrie cinématographique et contribuerait énormément à sa croissance. 
# 
# En outre, il serait plus efficace de construire un algorithme d'apprentissage automatique qui automatisera le processus de prédiction des films susceptibles d'avoir du succès commerciale et/ou popularité.

# In[ ]:




