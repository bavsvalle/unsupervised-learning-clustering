![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-FF0000?logo=spyderide&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)

# Unsupervised Machine Learning - Clusterização da Produção de Biodiesel

Este projeto tem como objetivo identificar padrões e concentração na produção de matérias-primas do biodiesel no Brasil utilizando clusterização hierárquica, a partir de dados abertos do [Governo Federal](https://dados.gov.br/home)

## 📊 Etapas do Projeto
1. **Coleta dos Dados**  
   - Fonte: [Painéis de Produção de Etanol e de Biodiesel](https://dados.gov.br/dados/conjuntos-dados/paineis-de-producao-de-etanol-e-de-biodiesel)
      - Arquivo: Matéria-Prima utilizadas na Produção de Biodiesel (CSV)
   
2. **Metodologia**  
   - Limpeza e padronização dos dados (Python + Pandas)
   - Padronização da variável quantidade_m3 (Z-score)
   - Cluster Hierárquico Aglomerativo
      - Single Linkage
      - Complete Linkage
      - Average Linkage
   - Análise por dendrogramas e estatísticas por clusters
   - Cluster Hierárquico Não Aglomerativo (K-means)
      - Definição do número de clusters pelo Método Elbow (WCSS)
      - Validação pelo Coeficiente de Silhueta
      - Modelo final com 2 clusters (k-means++)

3. **Conclusão comparativa entre métodos**
   - Single Linkage
      - Cluster 1 reúne os maiores volumes de produção
      - Todos os valores estão acima do percentil 99% da base geral
      - Óleo de soja representa aproximadamente 85% dos registros
      - Mediana do cluster (143.904 m³) é ~77x maior que a mediana geral (1.849 m³)
   - Complete Linkage
      - Cluster 1 concentra os maiores volumes, acima do percentil 95% da base
      - Forte predominância do óleo de soja (99% dos registros)
      - Mediana ~50x maior que a mediana geral.
   - Average Linkage
      - Cluster 0 reúne volumes acima do percentil 95%, com forte presença acima do 99%
      - Predominância do óleo de soja (99% dos registros)
      - Mediana ~40x maior que a mediana geral
      - A análise temporal indica:
         - Crescimento da produção média entre 2017 e 2021.
         - Queda pontual em 2022.
         - Retomada em 2023, mesmo com menor número de observações.
   - K-means
      - Cluster 1 concentra os maiores volumes de produção, acima do percentil 90% da base geral
      - Forte predominância do óleo de soja (99,62% dos registros)
      - Média do cluster (~77.845 m³) é ~9x maior que a média geral (~8.411 m³)
      - Resultados convergem com os métodos hierárquicos aglomerativos, confirmando a separação entre observações extremas e o restante da base
     
4. **Síntese Geral**
   - Em todos os métodos, os maiores volumes estão concentrados em poucos registros
   - O óleo de soja domina os clusters de alta escala em todos os modelos
   - Os métodos hierárquicos e o K-means convergiram para a mesma estrutura de 2 clusters
   - A diferença expressiva entre medianas confirma alta heterogeneidade e concentração produtiva
  
## 📁 Estrutura do Repositório

      ul-biodiesel-clustering/
      │
      ├── data/            # Dados brutos utilizados no projeto
      ├── outputs/         # Gráficos gerados (histogramas, boxplots, dendrogramas, elbow, silhueta e scatter)
      ├── src/             # Scripts Python com ETL, EDA e Clustering
      │
      ├── .gitattributes
      └── README.md

