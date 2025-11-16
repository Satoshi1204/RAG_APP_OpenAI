import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI # ← OpenAI に変更
from janome.tokenizer import Tokenizer # ← この1行を追加
from janome.analyzer import Analyzer  # ← この1行を追加
from janome.tokenfilter import POSStopFilter # ← この1行を追加


# .envファイルをロードして環境変数を設定
load_dotenv()

# --- OpenAI APIキーのシンプルな設定 ---
api_key = os.getenv("OPENAI_API_KEY") # ← OpenAI に変更
if not api_key:
    st.error("APIキーが見つかりません。.envファイルに OPENAI_API_KEY が正しく設定されているか確認してください。")
    st.stop()
else:
    # OpenAIクライアントを初期化（Geminiの configure は不要）
    # この client 変数は、後で get_openai_model 関数に渡します。
    client = OpenAI(api_key=api_key) 
    st.write("OpenAI APIキーを設定しました。") # 動作確認用
# --- 修正ここまで ---


# CSVファイルを読み込む関数
@st.cache_data
def load_data(csv_file_path):
    """
    指定されたパスからCSVファイルを読み込み、DataFrameを返す。
    ファイルが見つからない場合はエラーを表示して停止する。
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        st.error(f"エラー: CSVファイル ({csv_file_path}) が見つかりません。")
        st.info(f"app.py と同じフォルダに {csv_file_path} を配置してください。")
        st.stop()  # ファイルがないと続行できないためアプリを停止
    except Exception as e:
        st.error(f"データの読み込み中に予期せぬエラーが発生しました: {e}")
        st.stop()
        
# Janome（形態素解析）の準備
# Analyzer をグローバルに初期化（キャッシュで高速化）
@st.cache_resource
def get_janome_analyzer():
    # 「の」「は」「です」などの助詞・助動詞を除外するフィルタ
    token_filters = [POSStopFilter(['助詞', '助動詞', '記号'])]
    return Analyzer(token_filters=token_filters)

analyzer = get_janome_analyzer()

def tokenize_japanese(text):
    """
    Janomeを使って日本語のテキストを単語（名詞、動詞、形容詞など）に分割し、
    スペースで区切った文字列（わかち書き）を返す。
    """
    # analyzer.analyze(text) は Token オブジェクトのリストを返す
    # token.surface は元の単語（例：「田島」）
    tokens = [token.surface for token in analyzer.analyze(text)]
    return " ".join(tokens)

# TF-IDFモデルを構築する関数
@st.cache_resource
def build_tfidf_model(texts):
    """
    与えられたテキストのリストからTF-IDFベクトルライザとTF-IDF行列を構築する。
    （Janomeによる日本語形態素解析を実行）
    """
    st.write("TF-IDFモデルを構築中...") # 動作確認用
    
    # 1. Janomeで全テキストを「わかち書き」に変換
    st.write("日本語の形態素解析（わかち書き）を実行中...")
    tokenized_texts = [tokenize_japanese(text) for text in texts]
    st.write("形態素解析完了。")

    # 2. わかち書きされたテキストを TfidfVectorizer に渡す
    vectorizer = TfidfVectorizer(
        max_features=5000,
        max_df=0.95,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(tokenized_texts) # ← 日本語対応済みのテキストで処理
    
    st.write("TF-IDFモデル構築完了。") # 動作確認用
    return tfidf_matrix, vectorizer

# SentenceTransformerの埋め込みモデルを取得する関数
@st.cache_resource
def get_embedding_model():
    """
    SentenceTransformerの埋め込みモデル（多言語対応）をロードする。
    """
    st.write("埋め込みモデル（SentenceTransformer）をロード中...")
    # 高性能な多言語対応モデルを使用
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    st.write("埋め込みモデルのロード完了。")
    return model

# テキストデータをベクトル化する関数
@st.cache_resource
def build_embedding_model(texts, _model):
    """
    与えられたテキストのリストとモデルを使って、埋め込みベクトルを計算する。
    """
    st.write("テキストのベクトル化（埋め込み）を計算中...")
    embeddings = _model.encode(texts, show_progress_bar=True)
    st.write("テキストのベクトル化完了。")
    return embeddings



# ハイブリッド検索を行う関数
def hybrid_search(query, tfidf_vectorizer, tfidf_matrix, embedding_model, embeddings, top_n=5):
    """
    TF-IDF (キーワード) と SBERT (意味) の両方を使ってハイブリッド検索を行う。
    """
    
    # 1. TF-IDFスコアの計算 (キーワード検索)
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # 2. SBERTスコアの計算 (意味検索)
    query_embedding = embedding_model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, embeddings).flatten()
    
    # 3. ハイブリッドスコアの計算 (重み付け平均)
    hybrid_scores = (0.5 * tfidf_scores) + (0.5 * semantic_scores)
    
    # 4. スコアの高い順にソートし、インデックスとスコアを取得
    top_indices = hybrid_scores.argsort()[::-1][:top_n]
    
    # [(インデックス1, スコア1), (インデックス2, スコア2), ...] の形式で返す
    results = [(int(i), float(hybrid_scores[i])) for i in top_indices]
    
    return results

# OpenAIモデルを使って応答を生成する関数 (RAG)
def respond_with_openai(query, client, results, texts, top_n=3):
    """
    RAG (Retrieval-Augmented Generation) を実行する。
    検索結果（コンテキスト）を基に、OpenAIモデルが回答を生成する。
    """
    
    # 1. 検索結果から上位 top_n 件の「記事本文」を取得
    context_list = []
    for (index, score) in results[:top_n]:
        context_list.append(texts[index])
    
    # 2. コンテキストを結合して1つの文字列にする
    context = "\n---\n".join(context_list)
    
    # 3. OpenAIへのプロンプト（指示文）を作成 (ChatCompletions API形式)
    
    # システムへの指示
    system_prompt = """
あなたは、Yahoo!ニュースの記事について回答するAIアシスタントです。
以下の「参照記事」に書かれている情報**のみ**に基づいて、ユーザーの「質問」に回答してください。

# 制約条件:
- 参照記事に書かれていない事柄や、あなたの一般的な知識で回答してはいけません。
- 参照記事に該当する情報がない場合は、その旨を正直に伝えてください（例：「ご質問に関連する記事が見つかりませんでした。」）。
"""
    
    # ユーザーからの実際の質問
    user_prompt = f"""
# 参照記事:
{context}

# 質問:
{query}

# 回答:
"""

    # 4. OpenAI APIを呼び出して回答を生成
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # または "gpt-4o" など、利用可能なモデル
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAIでの回答生成中にエラーが発生しました: {e}")
        return "申し訳ありません。回答の生成中にエラーが発生しました。"

# チャット履歴を初期化する関数
def init_chat_history():
    """
    Streamlitのセッション状態を利用してチャット履歴を初期化する。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは。Yahoo!ニュースの記事に関するご質問をどうぞ。"}
        ]

# チャット履歴を表示する関数
def display_chat_history():
    """
    現在のチャット履歴をStreamlitのチャットUIで表示する。
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Streamlitアプリのメイン実行部分 ---
st.title("RAG System")

# STEP 1 & 2: データのロードとテキストの準備
csv_file_path = "yahoo_news_articles.csv"
df = load_data(csv_file_path)

if "text" in df.columns:
    texts = df["text"].fillna("").tolist()
    st.write(f"記事データのロード完了。記事数: {len(texts)} 件") # 動作確認用
else:
    st.error("エラー: 'text' 列がCSVファイルに見つかりません。")
    st.stop()

# STEP 3: TF-IDFモデルを構築
tfidf_matrix, tfidf_vectorizer = build_tfidf_model(texts)

# STEP 4: 埋め込みモデルを構築
embedding_model = get_embedding_model()
embeddings = build_embedding_model(texts, embedding_model)


# STEP 6: チャット履歴の初期化と表示
init_chat_history()
display_chat_history()
user_input = st.chat_input("質問を入力してください")

if user_input:
    # 1. ユーザーの入力をチャット履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. RAG (Retrieval-Augmented Generation) の実行
    
    # 2a. ハイブリッド検索 (Retrieval)
    search_results = hybrid_search(
        user_input, 
        tfidf_vectorizer, 
        tfidf_matrix, 
        embedding_model, 
        embeddings, 
        top_n=5
    )
    
    # 2b. 回答生成 (Generation)
    response_text = respond_with_openai( # ← 新しい関数名
        user_input, 
        client, # ← model ではなく、冒頭で定義した client を渡す
        search_results, 
        texts, 
        top_n=3
    )
    
    # 3. AIの応答をチャット履歴に追加して表示
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)