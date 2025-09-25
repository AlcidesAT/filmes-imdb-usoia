import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from tabulate import tabulate
from pathlib import Path   # Para salvar HTML
import webbrowser          # Para abrir no navegador

# carrega csv do IMDB
df_full = pd.read_csv("imdb-top-rated-movies-user-rated.csv")

# seleciona colunas relevantes e remove sem valor
colunas = ['Title', 'IMDb Rating', 'Meta Score', 'Votes', 'Tags', 'Director', 'Description', 'Poster URL']
df = df_full[colunas].dropna()

# salva IMDb Rating original (antes de normalizar) para exibição
df['IMDb Rating Original'] = df['IMDb Rating']

# converte strings de votos ("1.2K", "3M", etc.) para inteiros
def convert_votes(vote_str):
    vote_str = str(vote_str).replace(',', '').strip()
    if 'K' in vote_str:
        return int(float(vote_str.replace('K', '')) * 1_000)
    elif 'M' in vote_str:
        return int(float(vote_str.replace('M', '')) * 1_000_000)
    else:
        try:
            return int(vote_str)
        except:
            return np.nan

df['Votes'] = df['Votes'].apply(convert_votes)
df = df.dropna().reset_index(drop=True)

# --- normaliza colunas numéricas ---
num_cols = ['IMDb Rating', 'Meta Score', 'Votes']
scaler = StandardScaler()
features_num = scaler.fit_transform(df[num_cols])

# --- transforma colunas categóricas
cat_cols = ['Tags', 'Director']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_cat = encoder.fit_transform(df[cat_cols])

# --- transforma descrição em vetores TF-IDF ---
tfidf = TfidfVectorizer(max_features=500)
features_text = tfidf.fit_transform(df['Description']).toarray()

# --- concatena todas as features ---
features_all = np.hstack((features_num, features_cat, features_text))

# função para calcular filmes mais similares
def get_most_similar(index, k=5):
    base_vector = features_all[index].reshape(1, -1)
    sim = cosine_similarity(base_vector, features_all)[0]
    sim[index] = -1  # remove o próprio filme da lista
    sim_indices = sim.argsort()[::-1][:k]
    
    cols_exibir = ['Title', 'IMDb Rating Original', 'Tags', 'Director', 'Description', 'Poster URL']
    similares = df.iloc[sim_indices][cols_exibir].reset_index(drop=True)
    return similares

# busca por título (retorna índice do primeiro resultado)
def buscar_por_titulo(titulo):
    resultados = df[df['Title'].str.contains(titulo, case=False, na=False)]
    if resultados.empty:
        print("Nenhum filme encontrado com esse título.")
        return None
    else:
        return resultados.index[0]

# escolhe filme base
def escolher_filme():
    print("Escolha uma opção:")
    print("1 - Filme aleatório")
    print("2 - Buscar filme por título")
    opcao = input("Digite 1 ou 2: ").strip()
    if opcao == '1':
        return random.randint(0, len(df) - 1)
    elif opcao == '2':
        titulo = input("Digite parte do título do filme: ").strip()
        idx = buscar_por_titulo(titulo)
        if idx is None:
            print("Voltando para escolha inicial.")
            return escolher_filme()
        else:
            return idx
    else:
        print("Opção inválida, tente novamente.")
        return escolher_filme()

# gera HTML direto
def gerar_html(filme_base_row, similares_df, out_path="results.html"):
    
    # filme base com ícone mãe 🔵
    base_html = f"""
    <b>🔵 {filme_base_row['Title']}</b><br>
    IMDb Rating: <b>{filme_base_row['IMDb Rating Original']}</b><br>
    Diretor: <b>{filme_base_row['Director']}</b><br>
    Tags: {filme_base_row['Tags']}<br>
    <p>{filme_base_row['Description'][:400]}...</p>
    <a href="{filme_base_row['Poster URL']}" target="_blank">Link para Pôster</a>
    """
    
    similares_html = ""
    for _, row in similares_df.iterrows():
        similares_html += f"""
        <tr>
            <td>🔷 {row['Title']}</td>
            <td>{row['IMDb Rating Original']}</td>
            <td>{row['Director']}</td>
            <td>{row['Tags']}</td>
            <td>{row['Description'][:200]}...</td>
            <td><a href="{row['Poster URL']}" target="_blank">Link</a></td>
        </tr>
        """
    
    # HTML final
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="utf-8">
        <title>Filme base e similares</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding:20px; }}
            h2 {{ color:#222; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 1000px; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background:#f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>Filme base</h2>
        <div>{base_html}</div>
        <h2>Filmes similares</h2>
        <table>
            <thead>
                <tr>
                    <th>Título</th>
                    <th>IMDb Rating</th>
                    <th>Diretor</th>
                    <th>Tags</th>
                    <th>Descrição</th>
                    <th>Pôster</th>
                </tr>
            </thead>
            <tbody>
                {similares_html}
            </tbody>
        </table>
    </body>
    </html>
    """

    # salva e abre no navegador
    p = Path(out_path).resolve()
    p.write_text(html_content, encoding="utf-8")
    webbrowser.open_new_tab(p.as_uri())
    print(f"HTML gerado em: {p}")

# função auxiliar para encurtar texto
def resumir_texto(texto, tamanho=80):
    return (texto[:tamanho] + "...") if len(str(texto)) > tamanho else texto

# execução
if __name__ == "__main__":
    idx_base = escolher_filme()
    cols_base = ['Title', 'IMDb Rating Original', 'Tags', 'Director', 'Description', 'Poster URL']
    filme_base_row = df.iloc[idx_base][cols_base]
    similares = get_most_similar(idx_base)

    # --- preparar exibição no terminal ---
    filme_base_display = filme_base_row.copy()
    filme_base_display['Title'] = "🔵 " + str(filme_base_display['Title'])
    filme_base_display['Description'] = resumir_texto(filme_base_display['Description'], 100)

    similares_display = similares.copy()
    similares_display['Title'] = similares_display['Title'].apply(lambda x: "🔷 " + str(x))
    similares_display['Description'] = similares_display['Description'].apply(lambda x: resumir_texto(x, 100))

    print("\nFilme base:")
    print(tabulate([filme_base_display.drop(labels=['Poster URL'])],
                   headers="keys", tablefmt="fancy_grid", showindex=False))

    print("\nFilmes similares:")
    print(tabulate(similares_display.drop(columns=['Poster URL'], errors='ignore'),
                   headers="keys", tablefmt="fancy_grid", showindex=False))

    gerar_html(filme_base_row, similares)
