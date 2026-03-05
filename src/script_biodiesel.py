#%% Importação pacotes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.stats import zscore
from scipy.spatial.distance import pdist 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Análise e padronização do arquivo

# Leitura arquivo csv
df_mp_original = pd.read_csv("../data/biodiesel-materia-prima.csv")

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

#%% Dispersao dos dados
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
# Obs: Tanto pelo hisograma quanto pelo boxplot, verifiquei escala divergente entre os dados

# Padronização pelo Z score para adequação de escala
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

#%% Modelo 1 - Cluster Hierárquico Aglomerativo: métrica similaridade Dist Euclidiana e método encadeamento Single Linkage

# Gerando Dendograma
dist_euclidiana = pdist(df_mp_pad, metric='euclidean')

plt.figure(figsize=(12,6))

Z_simple = sch.linkage(df_mp_pad, method='single', metric='euclidean') # Matriz de aglomeracao hierarquica
print(Z_simple[:,2].max()) # Maior distancia que ocorreu a fusão de clusters
print(sorted(Z_simple[:,2])[-10:]) # 10 maiores distancias de fusão, sendo o maior salto: 0,20 a 0,32
sch.dendrogram(Z_simple)

plt.axhline(y=0.25, color='red', linestyle='--') # Utilizando valor 0,25 como linha de corte
plt.title('Dendrograma - Método Single Linkage')
plt.xlabel('Observações')
plt.ylabel('Distância Euclidiana')

plt.show()

# Dendograma com truncamento, selecionando os ultimos (lastp) 30 clusters formados (p=30)
# Obs: Foi necessário o truncamento para visualizar melhor o dendograma e onde realizar a linha de corte
plt.figure(figsize=(12,6))

sch.dendrogram(Z_simple, truncate_mode='lastp', p=30)

plt.axhline(y=0.25, color='red', linestyle='--') # Utilizando valor 0,25 como linha de corte
plt.title('Dendrograma - Single Linkage (Truncado)')
plt.xlabel('Clusters')
plt.ylabel('Distância')
plt.show()
# Realizado corte horizontal no dendrograma na altura 0,25, correspondente à formação de 2 clusters

# Configurando cluster Single
cluster_sing = AgglomerativeClustering(
               n_clusters=2,
               metric='euclidean',
               linkage='single'
               )

# Indica qual cluster cada linha pertence no df_mp_pad e no df_mp_original
indica_cluster_sing = cluster_sing.fit_predict(df_mp_pad)
df_mp_original['cluster_sing'] = indica_cluster_sing
df_mp_original.head()

# Faz a contagem de numeros por cluster
df_mp_original['cluster_sing'].value_counts()
# Obs: Cluster 1 é composto por 7 obervações, enquanto Cluster 0 concentra 4.738 registros.
# Indicando possivelmente outliers - a ser realizada análise na sequência.

# Análise Cluster 1 (7 valores)
df_mp_original.groupby('cluster_sing')['quantidade_m3'].describe()
df_mp_original.columns
df_mp_original.loc[
    df_mp_original['cluster_sing'] == 1,
    ['ano', 'produto', 'estado', 'quantidade_m3']
]
# Produto 'oleo de soja (glycine max)' teve destaque com a presença de 6 observações do total de 7.

# Avaliando 'oleo de soja (glycine max)' separadamente
df_soja_cluster1_sing = df_mp_original [
    (df_mp_original['cluster_sing'] == 1) &
    (df_mp_original['produto'] == 'oleo de soja (glycine max)')
]

df_soja_cluster1_sing['quantidade_m3'].describe()

df_mp_original['quantidade_m3'].describe(percentiles=[0.90,0.95,0.99])

df_soja_cluster1_sing.groupby('ano')['quantidade_m3'].agg(['count', 'mean', 'max', 'std'])

# Conclusão: O Cluster 1 reúne os maiores volumes de produção do conjunto de dados. Todos os valores desse grupo estão acima do percentil 99% da base geral, ou seja, fazem parte da faixa mais alta de produção observada.
# O óleo de soja representa aproximadamente 85% dos produtos presentes no cluster, indicando forte predominância desse produto entre os maiores volumes.
# Além disso, o óleo de milho corresponde ao maior valor registrado tanto dentro do cluster quanto em toda a base de dados.
# A diferença entre a mediana do cluster (143.904. m³) e a mediana geral (1.849 m³) — cerca de 77 vezes maior — mostra que os dados são bastante heterogêneos, com a produção concentrada em poucas observações de grande escala.

#%% Modelo 2 - Cluster Hierárquico Aglomerativo: métrica similaridade Dist Euclidiana e método encadeamento Complete Linkage

# Gerando Dendograma
Z_comp = sch.linkage(df_mp_pad, method='complete', metric='euclidean') # Matriz de aglomeracao hierarquica
print(Z_comp[:,2].max()) # Maior distancia que ocorreu a fusão de clusters
print(sorted(Z_comp[:,2])[-10:]) # 10 maiores distancias de fusão, sendo o maior salto: 4,05 a 7,90

sch.dendrogram(Z_comp)

plt.axhline(y=6, color='red', linestyle='--') # Utilizando valor 6 como linha de corte
plt.title('Dendrograma - Método Complete Linkage')
plt.xlabel('Observações')
plt.ylabel('Distância Euclidiana')

plt.show()

# Dendograma com truncamento, selecionando os ultimos (lastp) 30 clusters formados (p=30)
# Obs: Foi necessário o truncamento para visualizar melhor o dendograma e onde realizar a linha de corte
plt.figure(figsize=(12,6))

sch.dendrogram(Z_comp, truncate_mode='lastp', p=30)

plt.axhline(y=6, color='red', linestyle='--') # Utilizando valor 6 como linha de corte
plt.title('Dendrograma - Complete Linkage (Truncado)')
plt.xlabel('Clusters')
plt.ylabel('Distância')
plt.show()
# Realizado corte horizontal no dendrograma na altura 6, correspondente à formação de 2 clusters

# Configurando cluster Complete
cluster_compl = AgglomerativeClustering(
                n_clusters=2,
                metric='euclidean',
                linkage='complete'
                )

indica_cluster_comp = cluster_compl.fit_predict(df_mp_pad)
df_mp_original['cluster_compl'] = indica_cluster_comp
df_mp_original.head()
df_mp_original['cluster_compl'].value_counts()
# Obs: Cluster 1 é composto por 118 observações, enquanto Cluster 0 concentra 4.627 registros.
# Houve aumento do n de observações no Cluster 1 em relação ao método simple linkage, porém permanece sendo um grupo minoritário, já que abrange 2,49% de observações da base de dados.

# Análise Cluster 1 (118 observacoes)
df_mp_original.groupby('cluster_compl')['quantidade_m3'].describe()
df_mp_original.loc[
    df_mp_original['cluster_compl'] == 1,
    ['ano', 'produto', 'estado', 'quantidade_m3']
]
# Produto 'oleo de soja (glycine max)' parece manter destaque no Cluster 1, assim como no simple likage

# Avaliando 'oleo de soja (glycine max)' separadamente
df_soja_cluster1_comp = df_mp_original[
    (df_mp_original['cluster_compl'] == 1) &
    (df_mp_original['produto'] == 'oleo de soja (glycine max)')
]

df_soja_cluster1_comp.shape[0]
# Produto 'oleo de soja (glycine max)' teve destaque com a presença de 117 observações do total de 118.

df_soja_cluster1_comp.groupby('ano')['quantidade_m3'].agg(['count','mean','max', 'std'])
df_soja_cluster1_comp['quantidade_m3'].describe()
df_mp_original['quantidade_m3'].describe(percentiles=[0.90,0.95,0.99])


# Conclusão: O Cluster 1 reúne os maiores volumes de produção do conjunto de dados. Os valores desse grupo estão acima do percentil 95% da base geral, pertencendo a faixa mais elevada de produçao observada.
# O óleo de soja representa aproximadamente 99% (117 de 118 registros) dos produtos presentes no cluster, indicando forte predominância desse produto entre os maiores volumes.
# Além disso, o óleo de milho corresponde ao maior valor registrado tanto dentro do cluster quanto em toda a base de dados.
# A diferença entre a mediana do cluster (96.327 m³) e a mediana geral (1.849 m³) — cerca de 50 vezes maior — mostra que os dados são bastante heterogêneos, com a produção concentrada em poucas observações de grande escala.

#%% Modelo 3 - Cluster Hierárquico Aglomerativo: métrica similaridade Dist Euclidiana e método encadeamento Average Linkage

# Gerando Dendograma
Z_aver = sch.linkage(df_mp_pad, method='average', metric='euclidean') # Matriz de aglomeracao hierarquica
print(Z_aver[:,2].max()) # Maior distancia que ocorreu a fusão de clusters
print(sorted(Z_aver[:,2])[-10:]) # 10 maiores distancias de fusão, sendo o maior salto: 2.38 a 3.80

sch.dendrogram(Z_aver)

plt.axhline(y=3, color='red', linestyle='--') # Utilizando valor 6 como linha de corte
plt.title('Dendrograma - Método Average Linkage')
plt.xlabel('Observações')
plt.ylabel('Distância Euclidiana')

plt.show()

# Dendograma com truncamento, selecionando os ultimos (lastp) 30 clusters formados (p=30)
# Obs: Foi necessário o truncamento para visualizar melhor o dendograma e onde realizar a linha de corte
plt.figure(figsize=(12,6))

sch.dendrogram(Z_aver, truncate_mode='lastp', p=30)

plt.axhline(y=3, color='red', linestyle='--') # Utilizando valor 3 como linha de corte
plt.title('Dendrograma - Average Linkage (Truncado)')
plt.xlabel('Clusters')
plt.ylabel('Distância')
plt.show()
# Realizado corte horizontal no dendrograma na altura 3, correspondente à formação de 2 clusters

# Configurando cluster Average
cluster_aver = AgglomerativeClustering(
                n_clusters=2,
                metric='euclidean',
                linkage='average'
)

indica_cluster_aver = cluster_aver.fit_predict(df_mp_pad)
df_mp_original['cluster_aver'] = indica_cluster_aver
df_mp_original.head()
df_mp_original['cluster_aver'].value_counts()
# Obs: Cluster 0 é composto por 276 observações, enquanto Cluster 1 concentra 4.469 registros.
# Houve aumento do n de observações no Cluster 0 em relação ao método simple linkage e ao complete linkage, porém permanece sendo um grupo minoritário, já que abrange 5,81% de observações da base de dados.

# Análise Cluster 0 (276 observacoes)
df_mp_original.groupby('cluster_aver')['quantidade_m3'].describe()
df_mp_original.loc[
    (df_mp_original['cluster_aver'] == 0),
     ['ano','produto','estado','quantidade_m3']
]
# Produto 'oleo de soja (glycine max)' parece manter destaque no Cluster 1, assim como nos outros metodos linkage

# Avaliando 'oleo de soja (glycine max)' separadamente
df_soja_cluster0_aver = df_mp_original[
    (df_mp_original['cluster_aver'] == 0) &
    (df_mp_original['produto'] == 'oleo de soja (glycine max)')
]

df_soja_cluster0_aver.shape[0]
# Produto 'oleo de soja (glycine max)' teve destaque com a presença de 275 observações do total de 276.

df_soja_cluster0_aver.groupby('ano')['quantidade_m3'].agg(['count','mean','max', 'std'])
df_soja_cluster0_aver['quantidade_m3'].describe()
df_mp_original['quantidade_m3'].describe(percentiles=[0.90,0.95,0.99])


# Conclusão: O Cluster 0 reúne os maiores volumes de produção do conjunto de dados. Os valores desse grupo estão acima do percentil 95% da base geral, com forte presença acima do percentil 99%, caracterizado como a faixa mais elevada de produçao observada.
# A análise temporal indica crescimento contínuo da produção média de óleo de soja entre 2017 e 2021. 
# Em 2022 há uma redução pontual na média, e em 2023, apesar do menor número de observações (29), houve retomada dos níveis de produção.
# O óleo de soja representa aproximadamente 99% (275 de 276 registros) dos produtos presentes no cluster, indicando forte predominância desse produto entre os maiores volumes.
# Além disso, o óleo de milho corresponde ao maior valor registrado tanto dentro do cluster quanto em toda a base de dados.
# A diferença entre a mediana do cluster (96.327 m³) e a mediana geral (1.849 m³) — cerca de 40 vezes maior — mostra que os dados são bastante heterogêneos, com a produção concentrada em poucas observações de grande escala.

#%% Modelo 4 - Cluster Hierárquico Não Aglomerativo K-means

# Identificação da quantidade de clusters

# Método Elbow para identificação do nº de clusters
# Observar o "cotovelo" para verificar n de clusters. Quanto menor o WCSS, mais prox as obs estao do seu proprio centroide
elbow = []
K = range(1,6)  # você pode aumentar se quiser

for k in K:
    kmeansElbow = KMeans(n_clusters=k, init='k-means++', random_state=100)
    kmeansElbow.fit(df_mp_pad[['quantidade_m3']])
    elbow.append(kmeansElbow.inertia_)

# Plotando
plt.figure(figsize=(8,5), dpi=600)
plt.plot(K, elbow, marker='o')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('WCSS')
plt.title('Método do Elbow')
plt.xticks(K)
plt.show()

# Método Silhueta para identificação do nº de clusters
# Coeficiente silhueta - valores próximos de 1 indicativos de melhor separação e coesão entre os grupos. Maior = melhor.
silhouette_scores = []
K = range(2,6)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=100)
    labels = kmeans.fit_predict(df_mp_pad[['quantidade_m3']])
    score = silhouette_score(df_mp_pad[['quantidade_m3']], labels)
    silhouette_scores.append(score)

# Plot
plt.figure(figsize=(8,5), dpi=600)
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Coeficiente Médio de Silhueta')
plt.xticks(K)
plt.show()

# Obs: Resultados convergem o numero de clusters apresentado no metodo aglomerativo, cluster = 2.

# Criando modelo Kmeans com 2 clusters
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=100) # Escolhi 2 clusters, k-means++ já pula para resultados iniciais melhores, 100 garante reprodutibilidade
clusters_kmeans = kmeans.fit_predict(df_mp_pad[['quantidade_m3']]) # Calcula o centroide e atribui obs ao centroide mais proximo. fit ajusta o modelo aos dados e predict retorna o cluster de cada observação

# Gerando variável para identificar clusters gerados
df_mp_original['cluster_k'] = clusters_kmeans
df_mp_original['cluster_kmeans'] = df_mp_original['cluster_k'].astype('category')

df_mp_original['cluster_kmeans'].value_counts()

# Obs: Cluster 1 é composto por 266 observações, enquanto cluster 0 contém 4.479 observações

# Identificando Centróides em Z-score (uso interno do modelo)
cent_kmeans = pd.DataFrame(
    kmeans.cluster_centers_,
    columns = ['quantidade_m3']
 )

cent_kmeans.index.name = 'cluster'
cent_kmeans

# Centróides nos valores originais (para interpretação)
cent_kmeans_original = df_mp_original.groupby('cluster_k')['quantidade_m3'].mean().reset_index()
cent_kmeans_original.columns = ['cluster', 'quantidade_m3']
cent_kmeans_original

# Plotando Scatter em linha horizontal (apenas 1 variável quantidade_m3)
plt.figure(figsize=(8,4), dpi=600)
sns.scatterplot(
    data=df_mp_original,
    x='quantidade_m3',
    y=[0]*len(df_mp_original),
    hue='cluster_kmeans',
    palette='viridis',
    s=80
)
sns.scatterplot(
    data=cent_kmeans_original,
    x='quantidade_m3',
    y=[0]*len(cent_kmeans_original),
    color='red',
    marker='X',
    s=200,
    label='Centróides'
)
plt.yticks([])
plt.xlabel('Quantidade (m³)')
plt.title('Clusters e Centróides - KMeans')
plt.legend()
plt.show()

#       Análise Cluster 1 (266 observacoes) 
df_soja_cluster1_k = df_mp_original [
    (df_mp_original['cluster_kmeans'] == 1) &
    (df_mp_original['produto'] == 'oleo de soja (glycine max)')
]

df_soja_cluster1_k.shape[0]

df_soja_cluster1_k['quantidade_m3'].describe()
df_mp_original['quantidade_m3'].describe(percentiles=[0.90,0.95,0.99])

# Obs: Produto oleo de soja é predominante no cluster 1, com 265 observações, representando 99,62% dos dados.
# O metodo K means apresenta mesmo comportamento na criação dos clusters que o metodo Hierarquico Aglomerativo.
# Visto que, o cluster 1 apresenta os valores extremos de produção.

