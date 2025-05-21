import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer # Stemmer para português
import re
# Supondo que 'leIA' possa ser instalado ou esteja no seu diretório
# Se 'leIA' for um módulo local, certifique-se que está no PYTHONPATH
# ou na mesma pasta do app.py
try:
    from leia import SentimentIntensityAnalyzer
except ImportError:
    st.error("A biblioteca 'leIA' não foi encontrada. Certifique-se de que está instalada ou no caminho correto.")
    # Você pode tentar instalá-la se souber o nome do pacote PyPI
    # ou instruir o usuário a instalá-la.

# --- Downloads NLTK (executar apenas uma vez, se necessário) ---
@st.cache_resource
def inicializar_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    # O stemmer RSLP não precisa de download de recurso separado, mas a tokenização sim.

inicializar_nltk()
stop_words_pt = stopwords.words('portuguese')
stemmer = RSLPStemmer()

# --- Funções de Pré-processamento e Análise (adaptadas do seu notebook) ---

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto) # Remove URLs
    texto = re.sub(r'@\w+', '', texto) # Remove menções
    texto = re.sub(r'#\w+', '', texto) # Remove hashtags
    texto = re.sub(r'[^\w\s]', '', texto) # Remove pontuação
    texto = re.sub(r'\d+', '', texto) # Remove números
    tokens = word_tokenize(texto, language='portuguese')
    tokens_limpos = [stemmer.stem(palavra) for palavra in tokens if palavra not in stop_words_pt and len(palavra) > 2]
    return " ".join(tokens_limpos)

def analisar_sentimento(texto):
    if not isinstance(texto, str) or not texto.strip():
        return None, "Neutro" # Retorna None para score se não houver texto
    
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(texto)
    
    compound_score = vs['compound']
    if compound_score >= 0.05:
        return compound_score, "Positivo"
    elif compound_score <= -0.05:
        return compound_score, "Negativo"
    else:
        return compound_score, "Neutro"

def gerar_wordcloud(texto_processado, title="Word Cloud"):
    if not texto_processado.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words_pt, # Reaplicar stopwords caso o texto já não esteja 100% limpo para WC
                          collocations=True).generate(texto_processado)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.title(title)
    return fig

# --- Interface do Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Sentimento de Clipping")

st.title("📊 Análise de Sentimento e Termos em Clippings de Notícias")
st.markdown("""
Esta ferramenta permite analisar o sentimento e os termos mais frequentes em notícias.
Faça o upload de um arquivo CSV contendo as notícias ou insira o texto de uma notícia diretamente.
O CSV deve conter uma coluna com o texto das notícias (especifique o nome da coluna abaixo).
""")

# --- Sidebar para Upload e Configurações ---
st.sidebar.header("Fonte dos Dados")
tipo_entrada = st.sidebar.radio("Escolha a fonte dos dados:", ("Upload de Arquivo CSV", "Inserir Texto Único"))

df_noticias = None
textos_para_analise = [] # Lista para armazenar os textos

if tipo_entrada == "Upload de Arquivo CSV":
    arquivo_csv = st.sidebar.file_uploader("Faça upload do seu arquivo CSV", type=["csv"])
    if arquivo_csv is not None:
        try:
            df_original = pd.read_csv(arquivo_csv)
            st.sidebar.success(f"Arquivo '{arquivo_csv.name}' carregado com sucesso!")
            st.subheader("Pré-visualização dos Dados Carregados")
            st.dataframe(df_original.head())

            colunas_disponiveis = df_original.columns.tolist()
            coluna_texto = st.sidebar.selectbox("Selecione a coluna contendo o texto das notícias:", colunas_disponiveis)

            if coluna_texto:
                # Verificar se a coluna selecionada existe e não está vazia
                if coluna_texto in df_original.columns and not df_original[coluna_texto].isnull().all():
                    df_noticias = df_original.dropna(subset=[coluna_texto]).copy() # Remove linhas onde a coluna de texto é NaN
                    textos_para_analise = df_noticias[coluna_texto].astype(str).tolist()
                else:
                    st.sidebar.error(f"A coluna '{coluna_texto}' está vazia ou não contém textos válidos.")
                    textos_para_analise = []
            
        except Exception as e:
            st.sidebar.error(f"Erro ao ler o arquivo CSV: {e}")
            textos_para_analise = []

elif tipo_entrada == "Inserir Texto Único":
    texto_unico = st.text_area("Insira o texto da notícia aqui:", height=200)
    if texto_unico:
        textos_para_analise = [texto_unico]
        df_noticias = pd.DataFrame({'texto_original': [texto_unico]}) # Criar um DataFrame para consistência

# --- Botão de Análise ---
if st.button("🚀 Iniciar Análise"):
    if not textos_para_analise:
        st.warning("Por favor, carregue um arquivo CSV com uma coluna de texto válida ou insira um texto para análise.")
    else:
        with st.spinner("Realizando a análise... Por favor, aguarde."):
            
            resultados_analise = []
            textos_processados_completos = []

            for i, texto_original in enumerate(textos_para_analise):
                st.progress((i + 1) / len(textos_para_analise), text=f"Processando notícia {i+1}/{len(textos_para_analise)}")
                texto_limpo = limpar_texto(texto_original)
                score_sentimento, sentimento_label = analisar_sentimento(texto_original) # Usar texto original para LeIA
                
                resultados_analise.append({
                    'texto_original': texto_original,
                    'texto_processado': texto_limpo,
                    'score_sentimento': score_sentimento,
                    'sentimento': sentimento_label
                })
                if texto_limpo: # Apenas adiciona se houver conteúdo após limpeza
                    textos_processados_completos.append(texto_limpo)
            
            df_resultados = pd.DataFrame(resultados_analise)

            st.success("Análise concluída!")
            st.markdown("---")
            
            # --- Exibição dos Resultados ---
            st.subheader("Resultados da Análise de Sentimento")
            if not df_resultados.empty:
                st.dataframe(df_resultados[['texto_original', 'sentimento', 'score_sentimento']].head())

                # Contagem de sentimentos
                contagem_sentimentos = df_resultados['sentimento'].value_counts()
                st.write("Distribuição dos Sentimentos:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(contagem_sentimentos)
                with col2:
                    if not contagem_sentimentos.empty:
                        fig_pie, ax_pie = plt.subplots()
                        contagem_sentimentos.plot(kind='pie', ax=ax_pie, autopct='%1.1f%%', startangle=90,
                                                  colors=[sns.color_palette("pastel")[i] for i in range(len(contagem_sentimentos))])
                        ax_pie.set_ylabel('') # Remove o label do eixo y
                        ax_pie.set_title("Distribuição Percentual de Sentimentos")
                        st.pyplot(fig_pie)
                    else:
                        st.info("Não há dados de sentimento para exibir o gráfico de pizza.")

                # Download dos resultados
                @st.cache_data # Use st.cache_data para funções que retornam dados
                def converter_df_para_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_resultados = converter_df_para_csv(df_resultados)
                st.download_button(
                    label="Download dos resultados da análise como CSV",
                    data=csv_resultados,
                    file_name='analise_sentimento_clipping.csv',
                    mime='text/csv',
                )
            else:
                st.info("Nenhum resultado de sentimento para exibir.")

            st.markdown("---")
            st.subheader("Word Cloud dos Termos Mais Frequentes")
            if textos_processados_completos:
                texto_completo_para_wc = " ".join(textos_processados_completos)
                if texto_completo_para_wc.strip(): # Verifica se há algo para gerar a nuvem
                    fig_wordcloud = gerar_wordcloud(texto_completo_para_wc, "Word Cloud Geral das Notícias Processadas")
                    if fig_wordcloud:
                        st.pyplot(fig_wordcloud)
                    else:
                        st.warning("Não foi possível gerar a Word Cloud (texto processado pode estar vazio).")
                else:
                    st.info("Não há texto suficiente após o processamento para gerar uma Word Cloud.")
            else:
                st.info("Nenhum texto processado disponível para a Word Cloud.")

            # Opcional: Mostrar Word Clouds por sentimento
            st.markdown("---")
            st.subheader("Word Clouds por Sentimento (Opcional)")
            exibir_wc_sentimento = st.checkbox("Mostrar Word Clouds por categoria de sentimento?", value=False)

            if exibir_wc_sentimento and not df_resultados.empty:
                for sentimento_cat in df_resultados['sentimento'].unique():
                    if pd.notna(sentimento_cat): # Checa se não é NaN
                        textos_sentimento = " ".join(df_resultados[df_resultados['sentimento'] == sentimento_cat]['texto_processado'].dropna())
                        if textos_sentimento.strip():
                            st.write(f"**Word Cloud para notícias com sentimento: {sentimento_cat}**")
                            fig_wc_cat = gerar_wordcloud(textos_sentimento, f"Word Cloud - {sentimento_cat}")
                            if fig_wc_cat:
                                st.pyplot(fig_wc_cat)
                            else:
                                st.info(f"Não foi possível gerar Word Cloud para '{sentimento_cat}'.")
                        else:
                            st.info(f"Não há texto processado suficiente para '{sentimento_cat}'.")


else:
    st.info("Aguardando dados e o comando para iniciar a análise.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com base no projeto [clipping_analise](https://github.com/victorgdesouza/clipping_analise)")
