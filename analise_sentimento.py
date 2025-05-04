import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Lista de arquivos
arquivos = [
    "sentimentos/amazon_cells_labelled.txt",
    "sentimentos/imdb_labelled.txt",
    "sentimentos/yelp_labelled.txt"
]

# Criar um DataFrame vazio para armazenar todos os dados
df_total = pd.DataFrame()

# Carregar cada arquivo e adicionar ao DataFrame total
for arquivo in arquivos:
    df = pd.read_csv(arquivo, delimiter="\t", header=None, names=["sentenca", "sentimento"])
    df_total = pd.concat([df_total, df], ignore_index=True)

# Exibir as primeiras linhas
print(df_total.head())

# Exibir o número total de comentários carregados
print(f"Total de comentários carregados: {len(df_total)}")

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")  # Remove palavras irrelevantes em inglês
X = vectorizer.fit_transform(df_total["sentenca"])  # Transforma os textos em números
y = df_total["sentimento"]  # Variável alvo (0 = negativo, 1 = positivo)

# Dividindo os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo Naive Bayes
modelo = MultinomialNB()

# Treinando o modelo
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Calculando a acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")

# Adicionando a coluna 'predito' ao DataFrame
df_total["predito"] = modelo.predict(X)

# Adicionando a coluna 'sentimento_classificado' para identificação mais fácil dos sentimentos
df_total["sentimento_classificado"] = df_total["predito"].apply(lambda x: "positivo" if x == 1 else "negativo")

# Salvando o resultado em CSV para o Power BI
df_total.to_csv("resultado_sentimentos.csv", index=False)

print("Arquivo resultado_sentimentos.csv gerado com sucesso!")
