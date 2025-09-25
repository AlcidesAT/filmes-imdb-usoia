# 🎬 Recomendador de Filmes IMDB

Um sistema simples que recomenda filmes similares com base em avaliações, diretor, tags e descrição.  

## ✨ Funcionalidades

- Escolha um filme **aleatório** ou busque por **título**.
- Mostra o **filme base** e os **5 filmes mais similares**.
- Exibe informações:
  - IMDb Rating  
  - Diretor  
  - Tags  
  - Descrição resumida  
  - Link para pôster  
- Gera um **HTML interativo** com todos os resultados.

## ⚡ Como usar

1. Instale as dependências:
- pip install pandas numpy scikit-learn tabulate
  
2. Coloque o arquivo CSV imdb-top-rated-movies-user-rated.csv na mesma pasta do script

3. Execute o script: python nome_do_script.py

4. Escolha um filme aleatório ou busque por título

5. Veja os resultados no terminal e no navegador (HTML gerado automaticamente)

## 🔧 Como funciona

1. **Carregamento dos dados**  
   - Lê CSV e seleciona colunas relevantes.

2. **Tratamento de votos**  
   - Converte valores como `1.2K` ou `3M` em inteiros.

3. **Normalização e codificação**  
   - Normaliza colunas numéricas (`IMDb Rating`, `Meta Score`, `Votes`).  
   - Codifica colunas categóricas (`Tags`, `Director`) em vetores binários.

4. **Descrição em vetores TF-IDF**  
   - Transforma textos em vetores para comparação.

5. **Cálculo de similaridade**  
   - Usa `cosine_similarity` para encontrar os filmes mais parecidos.

6. **Exibição**  
   - No terminal, usando tabelas resumidas.  
   - Em HTML, com link para pôsteres e descrições completas.

## 📌 Observações

- Filtragem automática de dados faltantes.  
- Limita o tamanho das descrições para visualização resumida no terminal.  
- O HTML pode ser aberto em qualquer navegador moderno.
