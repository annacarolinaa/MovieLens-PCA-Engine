# CinemaMatch PCA

Sistema de recomendacao de filmes desenvolvido com **Streamlit**, **PCA** e a base **MovieLens 100k**. O projeto foi construido a partir da metodologia vista em aula, usando a matriz usuario-item, centragem por usuario e **matrix completion com PCA** para estimar ratings e gerar recomendacoes personalizadas.

## Visao geral

Este projeto transforma o fluxo da aula em um dashboard interativo e responsivo. A aplicacao permite:

- visualizar a matriz usuario-item com dados faltantes;
- aplicar PCA para reconstruir ratings ausentes;
- avaliar o modelo escondendo 10% dos ratings observados;
- explorar similaridade entre filmes populares;
- gerar recomendacoes para usuarios reais da base;
- simular um usuario convidado avaliando alguns filmes.

## Metodologia

O sistema segue a mesma logica apresentada no material da disciplina:

1. carregar os arquivos `u.data` e `u.item`;
2. construir a matriz usuario-item;
3. esconder aleatoriamente 10% dos ratings para avaliacao;
4. centralizar os ratings em torno da media de cada usuario;
5. preencher faltantes com zero na matriz centrada;
6. aplicar PCA para reduzir dimensionalidade e reconstruir a matriz;
7. calcular o erro de reconstrucao com RMSE;
8. recomendar filmes ainda nao vistos com base nos maiores scores previstos.

## Tecnologias utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset

Foi utilizada a base **MovieLens 100k**, composta por:

- `u.data`: avaliacoes dos usuarios;
- `u.item`: informacoes dos filmes, incluindo titulo e generos.

## Funcionalidades do app

- dashboard com layout responsivo;
- metricas principais do modelo;
- heatmap de dados faltantes;
- heatmap da matriz reconstruida;
- grafico de variancia explicada do PCA;
- analise de popularidade vs. nota media;
- mapa de similaridade entre filmes populares;
- recomendador por usuario do dataset;
- recomendador para usuario convidado.

## Como executar

1. Instale as dependencias:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

2. Execute a aplicacao:

```bash
streamlit run app.py
```

3. Abra o navegador no endereco exibido pelo Streamlit.

## Estrutura do projeto

```text
.
|-- app.py
|-- u.data
|-- u.item
|-- README.md
```

## Resultado esperado

Ao executar o app, o usuario pode explorar o comportamento da base, entender como o PCA reconstrui a matriz de ratings e visualizar recomendacoes de filmes de forma clara e interativa.

## Autor

Projeto academico desenvolvido para a disciplina de Sistemas de Recomendacao.
