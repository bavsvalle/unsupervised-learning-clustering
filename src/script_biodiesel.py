# language: python

# Importação pacotes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.stats import zscore
from scipy.spatial.distance import pdist 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

## Leitura arquivo csv
df_mp_original = pd.read_csv('data/biodiesel-materia-prima.csv')

# Verificando as primeiras 5 linhas do dataframe
df_mp_original.head()

# Verificando as info do dataframe
df_mp_original.info()

#Verificando se as variaveis estao na mesma escala 
df_mp_original.describe()

# Verificando quais as colunas do dataframe
df_mp_original.columns

# Verificando quant. de linhas e colunas
df_mp_original.shape

# Verificando quantos dados vazios em cada coluna
df_mp_original.isnull().sum()

# Padronizando str em minúscula
df_mp_original.columns = df_mp_original.columns.str.lower()
df_mp_original.head()


df_mp_original['região'] = df_mp_original['região'].str.lower()
df_mp_original['produto'] = df_mp_original['produto'].str.lower()
df_mp_original['estado'] = df_mp_original['estado'].str.lower()
df_mp_original.head()

# Removendo acentuação
from unidecode import unidecode
df_mp_original = df_mp_original.map(lambda x: unidecode(x) if isinstance(x, str) else x)
df_mp_original.head()


df_mp_original.columns = [unidecode(str(col)) for col in df_mp_original.columns]
df_mp_original.head()


df_mp_original.rename(columns={"quantidade (m3)": "quantidade_m3"}, inplace=True)

df_mp_original.head()

# Separando mês e ano em duas colunas
df_mp_original['mes'] = pd.to_datetime(df_mp_original['mes/ano'], format='%m/%Y').dt.month
df_mp_original['ano'] = pd.to_datetime(df_mp_original['mes/ano'], format='%m/%Y').dt.year
df_mp_original.head()

## Dispersao dos dados

# Histograma
fig_hist = px.histogram(
    df_mp_original,
    x='quantidade_m3',
    nbins=30
)

fig_hist.write_html('hist_quantidade_m3_30.html')

# Boxplot
fig_box = px.box(
    df_mp_original,
    y='quantidade_m3',
    title='Boxplot da Quantidade (m³)'
)

fig_box.write_html('boxplot_quantidade_m3.html')
# Obs: Tanto pelo hisograma quanto pelo boxplot, verifiquei escalas muito divergentes entre os dados


## Padronização pelo Z score para 
col_quant = df_mp_original[['quantidade_m3']] # Selecionei apenas coluna quantitativa e mantive como df [[]] 

type(col_quant)

df_mp_pad = pd.DataFrame(
            zscore(col_quant, ddof=1),
            columns = ['quantidade_m3'],
            index = df_mp_original.index
)

df_mp_pad.head()

print(round(df_mp_pad.mean(), 2))
print(round(df_mp_pad.std(), 2))

##  Cluster Hierárquico Aglomerativo: distância euclidiana + single linkage

# Visualizando distâncias
df_mp_pad.head()
dist_euclidiana = pdist(df_mp_pad, metric='euclidean')

# Gerando Dendograma

plt.figure(figsize=(12,6))

Z = sch.linkage(df_mp_pad, method='single', metric='euclidean')

sch.dendrogram(Z)

plt.axhline(y=4.5, color='red', linestyle='--')  # ponto de corte
plt.title('Dendrograma - Método Single Linkage')
plt.xlabel('Observações')
plt.ylabel('Distância Euclidiana')

plt.show()

# Dendograma com truncamento, selecionando os ultimos (lastp) 30 clusters formados (p=30)
# Obs: Foi necessário o truncamento para visualizar melhor o dendograma e onde realizar a linha de corte
plt.figure(figsize=(12,6))

Z = sch.linkage(df_mp_pad, method='single', metric='euclidean')

sch.dendrogram(Z, truncate_mode='lastp', p=30)

plt.title('Dendrograma - Single Linkage (Truncado)')
plt.xlabel('Clusters')
plt.ylabel('Distância')
plt.show()

# Configurando modelo de clusterização
cluster_sing = AgglomerativeClustering(
               n_clusters = 2,
               metric='euclidean',
               linkage = 'single'
               )
# Indica qual cluster cada linha pertence no df_mp_pad e no df_mp_original
indica_cluster_sing = cluster_sing.fit_predict(df_mp_pad)
df_mp_original['cluster_sing'] = indica_cluster_sing
df_mp_original.head()

# Faz a contagem de numeros por cluster
df_mp_original['cluster_sing'].value_counts()
# Obs: cluster 1 apresenta somente 7 valores, enquanto 0 contém + de 4k, indicando possivelmente outliers

## Comparação entre clusters

df_mp_original.groupby('cluster_sing')['quantidade_m3'].describe()
df_mp_original[df_mp_original['cluster_sing'] == 1]
df_mp_original.columns
df_mp_original.loc[
    df_mp_original['cluster_sing'] == 1,
    ['ano', 'produto', 'estado', 'quantidade_m3']
]
# Obs: Dados do cluster 1 apresentam alta escala, são outliers globais.





