from unidecode import unidecode
import nltk
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud

df = pd.read_excel('BASE.xlsx')

from nltk.corpus import stopwords

custom_words = [
        'nao', 'promocao', 'veja', 'limpador', 'individual', 'fabricante', 'uso', 'azulim', 'cera', 'marca', 'cif',
        'multiuso', 'limp', 'pano', 'uau', 'bombril', 'limpol', 'squeeze', 'garrafa', 'pack', 'ype', 'qboa', 'mult',
        'acao', 'refil', 'caixa', 'liquido', 'pulverizador','kit', 'lenco', 'cloro', 'bufalo', 'carrefur', 'facile',
        'minuano', 'ajax', 'clareadora'
    ]
stopwords = stopwords.words('portuguese') + custom_words
vectorizer = TfidfVectorizer(stop_words=stopwords)

def preprocess_text(text):
    text = unidecode(text)  # Remove acentos
    text = re.sub(r'\b\w*\d\w*\b', '', text)  # Remove palavras com números
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove palavras com 1 ou 2 caracteres
    return text

# Aplicar a função de pré-processamento à coluna 'nome_produto'
df['nome_produto'] = df['nome_produto'].apply(preprocess_text)


df.dropna(subset=['atributo'], inplace=True)
X = vectorizer.fit_transform(df['nome_produto'])
y = df['atributo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
accuracy = random_forest.score(X_test, y_test)
print(accuracy)



frequencies = X.sum(axis=0)
word_frequencies = {word: frequency for word, frequency in zip(
    vectorizer.get_feature_names_out(), frequencies.tolist()[0])}
# Criar a nuvem de palavras com max_words=500
wordcloud = WordCloud(width=800, height=800, background_color='white',
                      max_font_size=150,  max_words=800).generate_from_frequencies(word_frequencies)
# Exibir a nuvem de palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



df_new = pd.read_excel('novos_atributos.xlsx')
X_new = vectorizer.transform(df_new['nome_produto'])
predict = random_forest.predict(X_new)

df_new['atributo'] = predict

df_new.to_excel('NOVO2.xlsx', index=False)