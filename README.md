![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-FF0000?logo=spyderide&logoColor=white)

# Unsupervised Machine Learning - Clustering de Dados Públicos sobre Biodiesel no Brasil

Este projeto tem como objetivo realizar Análise Exploratória de Dados(EDA) das **Matérias-Primas utilizadas na Produção de Biodiesel no Brasil**, a partir de dados abertos do [Governo Federal](https://dados.gov.br/home)

## 📊 Etapas do Projeto
1. **Coleta dos Dados**  
   - Fonte: [Painéis de Produção de Etanol e de Biodiesel](https://dados.gov.br/dados/conjuntos-dados/paineis-de-producao-de-etanol-e-de-biodiesel)
      - Arquivo: Matéria-Prima utilizadas na Produção de Biodiesel (CSV)
   
2. **Tratamento (ETL) com Python**  
   - Limpeza e padronização (remoção de acentos com *Unidecode*, ajuste de datas e nomes de colunas, entre outros) 
   - Manipulação e transformação de dados com **pandas**  
   - Uso do **Spyder** para processamento

3. **Análise Exploratória de Dados (EDA)**
   - Visualização da distribuição da variável quantitativa quantidade_m3:
     - Histograma: hist_quantidade_m3_30.html
     - Boxplot: boxplot_quantidade_m3.html
     - Identificação de outliers globais
   - Estatísticas descritivas básicas (média, desvio padrão, quartis)
   - Padronização dos dados usando Z-score para uniformizar escalas

4. **Clustering / Agrupamento de Dados**
   - Cluster Hierárquico Aglomerativo
      - Métrica: Euclidiana
      - Linkage: Single
      - Visualização: dendrograma completo e truncado (últimos 30 clusters)
      - Linha de corte para definição de clusters com outliers destacados
    - Observações:  
    Cluster 1 possui apenas 7 observações — correspondem a outliers globais  
    Cluster 0 possui mais de 4.000 observações — representa a maioria dos dados  
    Estatísticas por cluster foram analisadas para identificar padrões de escala
5. **Próximos passos**
    - Comparação com outros critérios de ligação (Average e Complete Linkage)
    - Comparação com método não hierárquico (K-Means) para avaliar robustez dos agrupamentos
  
## 📁 Estrutura do Repositório

      ul-biodiesel-clustering/
      │
      ├── data/            # Dados brutos utilizados no projeto
      ├── outputs/         # Gráficos gerados (histogramas, boxplots, dendrogramas)
      ├── src/             # Scripts Python com ETL, EDA e Clustering
      │
      ├── .gitattributes
      └── README.md

