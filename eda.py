import os
import streamlit as st
import re
import nltk
import json
import plotly.express as px
import pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
import subprocess
import spacy
import asyncio
import networkx as nx
from bertopic import BERTopic
from umap import UMAP
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from spacy import displacy
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Imports adicionais para as novas tarefas
from wordcloud import WordCloud
from nltk import ngrams
import gensim
from gensim import corpora, models
from transformers import pipeline

# --- Correções para evitar o erro do watcher do Streamlit e torch ---
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"
# ------------------------------------------------------------------

# Desativar paralelismo dos tokenizers para evitar deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configurar a interface do Streamlit como "wide"
st.set_page_config(layout="wide")

# Título da aplicação
st.title("Análise de Atas de Reunião com NLP")

# Baixar stopwords e punkt se necessário
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
portuguese_stopwords = set(nltk.corpus.stopwords.words('portuguese'))

# Ajustar asyncio para evitar conflitos com Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Carregar o modelo spaCy (modelo em português)
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    st.warning("Modelo 'pt_core_news_lg' não encontrado. Instalando...")
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"], check=True)
    nlp = spacy.load("pt_core_news_lg")

# Sidebar - Seleção de tarefa (incluindo a nova opção "Documentação")
st.sidebar.header("Escolha a Tarefa")
task = st.sidebar.selectbox(
    "Selecione a análise desejada:",
    [
        "0️⃣ Documentação",
        "1️⃣ Pré-processamento de Texto",
        "2️⃣ Frequência de Palavras",
        "3️⃣ Clusterização com BERTopic",
        "4️⃣ Extração de Entidades Nomeadas (NER)",
        "5️⃣ Sumarização com Ollama",
        "6️⃣ Análise com GraphRAG (Busca na Base)",
        "7️⃣ Estatísticas Descritivas",
        "8️⃣ Análise de Padrões e Estrutura",
        "9️⃣ Visualizações para Exploração",
        "🔟 Extração de Informações e Insights"
    ]
)

# Upload do arquivo
st.sidebar.subheader("Upload do Arquivo")
uploaded_file = st.sidebar.file_uploader("Carregue um arquivo .txt", type=["txt"])
if uploaded_file is not None:
    file_path = "uploaded_text.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(uploaded_file.getvalue().decode("utf-8"))
    st.sidebar.success("Arquivo carregado com sucesso!")

# ------------------------ Funções do Aplicativo ------------------------ #

@st.cache_resource
def generate_topic_model(documents):
    """
    Gera o modelo BERTopic com UMAP e retorna (topic_model, topics).
    Ajustes:
      - UMAP: n_neighbors=15, n_components=5, métrica 'cosine'
      - BERTopic: min_topic_size=10 e calculate_probabilities ativado
    """
    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
    topic_model = BERTopic(umap_model=umap_model, min_topic_size=10, calculate_probabilities=True, verbose=True)
    topics, _ = topic_model.fit_transform(documents)
    return topic_model, topics

def preprocess_text(text):
    """
    Remove quebras de linha e espaços redundantes.
    Em seguida, utiliza spaCy para lematizar os tokens, removendo stopwords e pontuação.
    Retorna (formatted_text, clean_text) onde clean_text é o texto lematizado.
    """
    formatted_text = re.sub(r'\n+', ' ', text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text).strip()
    doc = nlp(formatted_text)
    lemmatized_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    clean_text = ' '.join(lemmatized_tokens)
    return formatted_text, clean_text

def chunk_text(text, chunk_size=200):
    """
    Segmenta o texto em sentenças (usando spaCy) e agrupa sentenças em chunks de tamanho aproximado chunk_size.
    Em seguida, remove duplicatas (opcional – comente as linhas de deduplicação se desejar manter trechos semelhantes).
    """
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sent in doc.sents:
        if len(current_chunk) + len(sent.text) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent.text
        else:
            current_chunk += " " + sent.text
    if current_chunk:
        chunks.append(current_chunk.strip())
    # Deduplicação
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)
    return unique_chunks

import textwrap

def generate_meeting_minutes(
    text: str,
    base_url: str = 'http://localhost:11434/v1/',
    api_key: str = 'ollama',
    model: str = "llama3.2:3b",
    max_tokens: int = 1000
) -> str:
    """
    Gera uma ata de reunião estruturada em sessões, utilizando o modelo do Ollama.
    
    A ata é dividida em:
      1. Cabeçalho:
         - Título da ata
         - Data da reunião
         - Horário de início
         - Local (presencial, virtual ou híbrido)
         - Lista de Participantes (Presenciais, Virtuais e Ausências justificadas)
      2. Corpo do Documento (seções numeradas):
         a) Abertura da Reunião – Quem abriu a reunião e contexto inicial
         b) Apresentação e Posse dos Membros – Registros sobre posse ou apresentação
         c) Apresentação Institucional – Resumo da instituição (história, missão, visão e pilares)
         d) Gestão Estratégica – Planos, desafios e metas institucionais
         e) Projetos ou Temas Específicos – Discussão de projetos e impactos
         f) Discussões e Contribuições – Sugestões e debates registrados
         g) Encerramento – Agradecimentos, conclusões e detalhes da próxima reunião
         
    Todas as seções deverão ser redigidas em português padrão, utilizando linguagem formal, objetiva e sem inclusão de informações fictícias.
    
    Args:
        text (str): Conteúdo original da ata de reunião.
        base_url (str, opcional): URL base para acesso à API do Ollama. Padrão: 'http://localhost:11434/v1/'.
        api_key (str, opcional): Chave de acesso à API. Padrão: 'ollama'.
        model (str, opcional): Nome do modelo a ser utilizado. Padrão: "gemma2:2b".
        max_tokens (int, opcional): Número máximo de tokens na resposta. Padrão: 1000.
        
    Returns:
        str: Ata de reunião gerada ou mensagem de erro em caso de exceção.
    """
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as e:
        return f"Erro ao inicializar o cliente Ollama: {str(e)}"

    def call_api(prompt: str) -> str:
        """
        Realiza a chamada à API do Ollama com o prompt fornecido.
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro ao conectar ao Ollama: {str(e)}"

    # Geração do Cabeçalho
    header_prompt = textwrap.dedent(f"""
        Você é um assistente especializado na elaboração de atas de reunião.
        Com base nas informações fornecidas, redija o CABEÇALHO da ata de reunião, seguindo rigorosamente a estrutura abaixo,
        utilizando exclusivamente as informações fornecidas e respondendo em português.

        Estrutura do Cabeçalho:
        - Título da ata
        - Data da reunião
        - Horário de início
        - Local (indicar se a reunião foi presencial, virtual ou híbrida)
        - Lista de Participantes:
            • Presenciais: Nome dos participantes presentes fisicamente
            • Virtuais: Nome dos participantes que se conectaram remotamente
            • Ausências justificadas: Nome dos ausentes, com justificativa quando disponível

        Ata de Reunião:
        {text}
    """)
    header = call_api(header_prompt)

    # Dicionário com as seções do Corpo da Ata
    body_sections = {
        "a) Abertura da Reunião": "Quem abriu a reunião e o contexto inicial",
        "b) Apresentação e Posse dos Membros": "Registros sobre posse ou apresentação dos membros",
        "c) Apresentação Institucional": "Resumo da instituição (história, missão, visão e pilares)",
        "d) Gestão Estratégica": "Planos, desafios e metas institucionais",
        "e) Projetos ou Temas Específicos": "Discussão de projetos e impactos",
        "f) Discussões e Contribuições": "Sugestões e debates registrados",
        "g) Encerramento": "Agradecimentos, conclusões e detalhes da próxima reunião"
    }

    body_outputs = {}
    for section_title, description in body_sections.items():
        section_prompt = textwrap.dedent(f"""
            Você é um assistente especializado na elaboração de atas de reunião.
            Com base nas informações fornecidas, redija a seção "{section_title}" da ata, que deve conter:
            {description}.
            Siga estritamente a estrutura e o formato do documento de referência, utilizando exclusivamente as informações fornecidas.
            Responda em português, com linguagem formal, objetiva e sem jargões ou opiniões pessoais.

            Seção: {section_title} - {description}

            Ata de Reunião:
            {text}
        """)
        body_outputs[section_title] = call_api(section_prompt)

    # Agregando o Cabeçalho e as Seções do Corpo em um único documento
    final_minutes = "ATA DE REUNIÃO\n\n"
    final_minutes += "CABEÇALHO:\n" + header + "\n\n"
    for section_title in body_sections.keys():
        final_minutes += f"{section_title}:\n" + body_outputs[section_title] + "\n\n"
    
    return final_minutes

def search_graph_rag(text):
    st.subheader("Busca na Base de Conhecimento (GraphRAG)")
    persist_directory = "./chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_chunks = chunk_text(text, chunk_size=200)
    if not text_chunks:
        st.error("Não foi possível gerar chunks a partir do texto.")
        return
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    st.write("Base de conhecimento vetorizada criada com sucesso.")
    query = st.text_input("Digite sua consulta para buscar na base de conhecimento:")
    if query:
        try:
            raw_results = vectorstore.similarity_search(query, k=10)
            results = []
            seen_texts = set()
            for res in raw_results:
                if res.page_content not in seen_texts:
                    results.append(res)
                    seen_texts.add(res.page_content)
            st.markdown("**Resultados da busca vetorizada (deduplicados):**")
            for res in results:
                st.write(res.page_content)
            G = nx.Graph()
            chunk_entities = {}
            for idx, res in enumerate(results):
                chunk_text_result = res.page_content
                doc = nlp(chunk_text_result)
                entities = set(ent.text for ent in doc.ents if len(ent.text.strip()) > 1)
                chunk_entities[idx] = entities
                G.add_node(idx, text=chunk_text_result, entities=entities)
            for i in range(len(results)):
                for j in range(i+1, len(results)):
                    common_entities = chunk_entities[i].intersection(chunk_entities[j])
                    if len(common_entities) >= 1:
                        G.add_edge(i, j, weight=len(common_entities))
            pagerank = nx.pagerank(G, weight='weight')
            vector_scores = {}
            total = 0
            for idx, res in enumerate(results):
                score = None
                if hasattr(res, 'metadata') and isinstance(res.metadata, dict):
                    score = res.metadata.get('score')
                if score is None:
                    score = len(results) - idx
                vector_scores[idx] = score
                total += score
            for idx in vector_scores:
                vector_scores[idx] /= total
            combined_scores = {}
            for idx in range(len(results)):
                pr_score = pagerank.get(idx, 0)
                vec_score = vector_scores.get(idx, 0)
                combined_scores[idx] = 0.5 * pr_score + 0.5 * vec_score
            ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            st.markdown("**Resultados da busca com GraphRAG (combinação de centralidade e similaridade):**")
            for idx, score in ranked_results:
                st.write(f"Score combinado: {score:.4f}")
                st.write(G.nodes[idx]['text'])
                st.write("---")
            st.subheader("Visualização do Grafo de Conhecimento")
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Erro na busca: {str(e)}")

def extract_named_entities(text):
    st.subheader("Extração de Entidades Nomeadas (NER)")
    doc = nlp(text)
    entities = [{"Entidade": ent.text, "Tipo": ent.label_} for ent in doc.ents]
    if entities:
        df_entities = pd.DataFrame(entities).value_counts().reset_index(name="Frequência")
        st.dataframe(df_entities)
        st.markdown(displacy.render(doc, style="ent", minify=True), unsafe_allow_html=True)
    else:
        st.write("Nenhuma entidade nomeada encontrada.")

def descriptive_statistics(text):
    tokens = text.split()
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_lengths = [len(sent.text.split()) for sent in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    st.markdown("**Estatísticas Descritivas:**")
    st.write(f"Total de tokens: {total_tokens}")
    st.write(f"Total de tipos (tokens únicos): {unique_tokens}")
    st.write(f"Relação Tipo/Token (TTR): {ttr:.4f}")
    st.write(f"Comprimento médio das sentenças: {avg_sentence_length:.2f} palavras")

def patterns_analysis(text):
    st.markdown("**Análise de Padrões e Estrutura:**")
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)
    st.write("Bigramas mais comuns:")
    bigrams_df = pd.DataFrame(bigram_freq.most_common(10), columns=["Bigram", "Frequency"])
    st.table(bigrams_df)
    st.write("Trigramas mais comuns:")
    trigrams_df = pd.DataFrame(trigram_freq.most_common(10), columns=["Trigram", "Frequency"])
    st.table(trigrams_df)
    pos_tags = [(token.text, token.pos_) for token in doc]
    pos_df = pd.DataFrame(pos_tags[:20], columns=["Token", "POS"])
    st.write("Exemplo de POS Tagging (primeiras 20 tokens):")
    st.table(pos_df)
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        sentiment = sentiment_pipeline(text, truncation=True, max_length=512)
        st.write("Análise de Sentimento:")
        sentiment_df = pd.DataFrame(sentiment)
        st.table(sentiment_df)
    except Exception as e:
        st.write("Erro na análise de sentimento:", e)

def exploration_visualizations(text):
    st.markdown("**Visualizações para Exploração:**")
    words = text.split()
    word_lengths = [len(word) for word in words]
    st.write("Histograma do comprimento das palavras:")
    fig1, ax1 = plt.subplots()
    ax1.hist(word_lengths, bins=20)
    st.pyplot(fig1)
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_lengths = [len(sent.text.split()) for sent in sentences]
    st.write("Histograma do comprimento das sentenças:")
    fig2, ax2 = plt.subplots()
    ax2.hist(sentence_lengths, bins=20)
    st.pyplot(fig2)
    st.write("Nuvem de Palavras:")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig3, ax3 = plt.subplots()
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis("off")
    st.pyplot(fig3)

def extract_insights(text):
    st.markdown("**Extração de Informações e Insights:**")
    tokens = [token.text.lower() for token in nlp(text) if not token.is_stop and not token.is_punct]
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)
    topics_list = [f"Topic {i+1}: {topic}" for i, topic in enumerate(topics)]
    st.markdown("**Tópicos extraídos (LDA):**")
    for topic in topics_list:
        st.write(topic)
    co_occurrence = {}
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)):
            pair = tuple(sorted((tokens[i], tokens[j])))
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
    co_occurrence = {pair: count for pair, count in co_occurrence.items() if count > 1}
    cooccurrence_df = pd.DataFrame(list(co_occurrence.items())[:10], columns=["Word Pair", "Count"])
    st.markdown("**Algumas co-ocorrências de palavras:**")
    st.table(cooccurrence_df)

def explain_code():
    st.markdown("## Documentação e Explicação do Código")
    st.markdown("""
Este aplicativo foi desenvolvido para realizar uma análise completa de atas de reunião utilizando diversas técnicas de Processamento de Linguagem Natural (NLP) e métodos de visualização. A seguir, descrevemos as principais funcionalidades e as técnicas empregadas:

### Funcionalidades do Aplicativo

1. **Pré-processamento de Texto**
   - **Objetivo:** Limpar o texto, removendo quebras de linha e espaços redundantes.
   - **Técnicas:**  
     - Lematização com spaCy (reduz palavras à sua forma canônica).
     - Remoção de stopwords e pontuação.

2. **Frequência de Palavras**
   - **Objetivo:** Identificar as palavras mais frequentes no corpus.
   - **Técnicas:**  
     - Uso do CountVectorizer para contar ocorrências.
     - Visualização por meio de gráficos de barras.

3. **Clusterização com BERTopic**
   - **Objetivo:** Descobrir tópicos latentes no texto.
   - **Técnicas:**  
     - Redução de dimensionalidade com UMAP.
     - Clusterização com BERTopic (com parâmetros ajustados para melhor separação dos tópicos).
     - Visualização de um dendrograma por meio do método `visualize_hierarchy()`.

4. **Extração de Entidades Nomeadas (NER)**
   - **Objetivo:** Reconhecer e extrair nomes, organizações, datas, etc.
   - **Técnicas:**  
     - Uso do modelo spaCy para NER.

5. **Sumarização com Ollama**
   - **Objetivo:** Gerar um sumário executivo da ata.
   - **Técnicas:**  
     - Uso da API do Ollama para criar um documento resumido mantendo a estrutura formal.

6. **Análise com GraphRAG (Busca na Base)**
   - **Objetivo:** Integrar busca vetorial e análise em grafo para encontrar os trechos mais relevantes.
   - **Técnicas:**  
     - Divisão do texto em chunks (fragmentos) com spaCy.
     - Criação de uma base vetorizada com Chroma e embeddings do HuggingFace.
     - Construção de um grafo de conhecimento e cálculo da centralidade com PageRank.
     - Combinação de similaridade vetorial com a centralidade para ranquear resultados.

7. **Estatísticas Descritivas**
   - **Objetivo:** Fornecer informações quantitativas sobre o texto.
   - **Técnicas:**  
     - Contagem de tokens e tokens únicos.
     - Cálculo da relação tipo/token (TTR).
     - Cálculo do comprimento médio das sentenças.

8. **Análise de Padrões e Estrutura**
   - **Objetivo:** Explorar padrões linguísticos.
   - **Técnicas:**  
     - Extração de bigramas e trigramas.
     - POS Tagging (classificação gramatical dos tokens).
     - Análise de sentimento com pipeline do Transformers (exibida em formato tabular).

9. **Visualizações para Exploração**
   - **Objetivo:** Visualizar características do texto.
   - **Técnicas:**  
     - Histogramas do comprimento das palavras e das sentenças.
     - Geração de uma nuvem de palavras.

10. **Extração de Informações e Insights**
    - **Objetivo:** Descobrir tópicos latentes e relações entre palavras.
    - **Técnicas:**  
      - Aplicação de LDA (Latent Dirichlet Allocation) para extração de tópicos.
      - Análise de co-ocorrência de palavras para identificar relações.

### Conclusão

Este aplicativo integra diversas técnicas avançadas de NLP para permitir uma análise profunda e interativa de atas de reunião, possibilitando a extração de insights, a visualização de padrões e a criação de sumários e tópicos relevantes.
    """, unsafe_allow_html=True)

# ------------------------ Execução do Aplicativo ------------------------ #

def main():
    if uploaded_file is None:
        st.error("Por favor, carregue um arquivo .txt para prosseguir.")
        st.stop()
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        _, clean_text = preprocess_text(raw_text)
    if task == "0️⃣ Documentação":
        explain_code()
    elif task == "1️⃣ Pré-processamento de Texto":
        st.subheader("Pré-processamento de Texto")
        tokens_originais = len(raw_text.split())
        tokens_clean = len(clean_text.split())
        st.write(f"Tokens no texto original: {tokens_originais}")
        st.write(f"Tokens após remoção de stopwords: {tokens_clean}")
        st.text_area("Texto Limpo:", clean_text, height=300)
    elif task == "2️⃣ Frequência de Palavras":
        st.subheader("Frequência de Palavras")
        vectorizer = CountVectorizer(stop_words=list(portuguese_stopwords), max_features=50)
        X = vectorizer.fit_transform([clean_text])
        palavras = vectorizer.get_feature_names_out()
        frequencias = X.toarray().sum(axis=0)
        df_frequencias = pd.DataFrame({"Palavra": palavras, "Frequência": frequencias}).sort_values(by="Frequência", ascending=False)
        st.dataframe(df_frequencias)
        fig = px.bar(df_frequencias, x="Palavra", y="Frequência", title="Frequência de Palavras", text="Frequência")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)
    elif task == "3️⃣ Clusterização com BERTopic":
        st.subheader("Clusterização com BERTopic")
        documents = [doc.strip() for doc in clean_text.split('.') if doc.strip()]
        topic_model, topics = generate_topic_model(documents)
        st.dataframe(topic_model.get_topic_info())
        try:
            st.plotly_chart(topic_model.visualize_barchart(top_n_topics=5))
        except Exception as e:
            st.error(f"Erro ao exibir gráfico de tópicos: {str(e)}")
        # Exibir dendrograma/hierarquia dos tópicos
        try:
            dendrogram_fig = topic_model.visualize_hierarchy()
            st.plotly_chart(dendrogram_fig)
        except Exception as e:
            st.error(f"Erro ao exibir dendrograma: {str(e)}")
    elif task == "4️⃣ Extração de Entidades Nomeadas (NER)":
        extract_named_entities(raw_text)
    elif task == "5️⃣ Sumarização com Ollama":
        st.subheader("Sumarização com Ollama")
        st.markdown("**Gerando sumarização com o modelo do Ollama...**")
        summary = generate_meeting_minutes(raw_text)
        st.markdown("**Sumário Executivo:**")
        st.markdown(summary, unsafe_allow_html=True)
    elif task == "6️⃣ Análise com GraphRAG (Busca na Base)":
        search_graph_rag(raw_text)
    elif task == "7️⃣ Estatísticas Descritivas":
        descriptive_statistics(clean_text)
    elif task == "8️⃣ Análise de Padrões e Estrutura":
        patterns_analysis(raw_text)
    elif task == "9️⃣ Visualizações para Exploração":
        exploration_visualizations(clean_text)
    elif task == "🔟 Extração de Informações e Insights":
        extract_insights(raw_text)

if __name__ == "__main__":
    main()
