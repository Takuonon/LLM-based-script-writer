import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# L2距離で上位1つのテキストをDBから返す
def text_rag(text):
    api_keys = os.getenv("OPENAI_API_KEY")
    embed = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = FAISS.load_local("db.faiss_only_content", embed)

    def extract_text_after_keyword(text, keyword):
        # キーワードの位置を検索します
        keyword_position = text.find(keyword)

        # キーワードが見つからなかった場合、空の文字列を返します
        if keyword_position == -1:
            return ""

        # キーワードの後のテキストを抽出します
        start_position = keyword_position + len(keyword)
        extracted_text = text[start_position:]

        return extracted_text.strip()  # 余分な空白を削除します

    def ask_database(query, db=db, threshold=0.4):
        # .similarity_search_with_scoreは取得物とともにスコアを返す
        # デフォルトはL2キョリ(コサイン類似度とかにもできるらしい？)
        db = FAISS.load_local("db.faiss_only_content", embed)

        # 1つだけ似ている文章をretriveしてくる
        docs_and_scores = db.similarity_search_with_score(query, k=1)
        retrieved_doc_text = []

        for i, (doc, score) in enumerate(docs_and_scores):
            # print(doc,score)
            text = extract_text_after_keyword(doc.page_content, "内容: ")
            retrieved_doc_text.append(text)
        return retrieved_doc_text

    return ask_database(text)
