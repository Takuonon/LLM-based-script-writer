import openai
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import OpenAIModerationChain
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain


def generate_script(ocr_result, api_keys):
    # LLMラッパーの作成
    gpt = ChatOpenAI(model_name="gpt-4", openai_api_key=api_keys)

    # few-shotの型
    example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

    # Chain1のprompt
    examples1 = [
        {
            "question": """以下は大学講義の授業スライドをOCR結果である。
            スライドのテーマと内容を箇条書きでまとめて。
            ただ、末部の授業名や教員の名前は無視して。
            また、内容の詳細情報が抜けている場合もある。:
状態エンコーディングのやり方
バイナリ (二進数) ・エンコーディング
- 4状態へはOOO1.10,1 1 とつける
- どの状態にどれを 対記させるかによって論理の複雑さは変わる
ワン・ホット・エンコーディング
1つの状態毎に1ビットを用意
どれか1つの状態ビピットだけが1 となる

ディジタル回路
東大・入江

""",
            "answer": """
タイトル: 状態エンコーディングの方法
内容1: バイナリエンコーディング
- 各状態はバイナリで、0001、0010、0100、1000の4状態にエンコードされる。
- 状態に割り当てるコードによって論理回路の複雑さが変わ
内容2: ワンホットエンコーディング
- 1つのビットが1つの状態を表す。
- 1つだけのビットが1となり、残りは0となる。
""",
        }
    ]
    prompt1 = FewShotPromptTemplate(
        examples=examples1, example_prompt=example_prompt, suffix="Question: {input}", input_variables=["input"]
    )

    chain1 = LLMChain(llm=gpt, prompt=prompt1)

    # Chain2のprompt
    examples2 = [
        {
            "question": """以下内容を1つに繋げて原稿にして。
            内容がない部分については補足しなくて大丈夫。:
タイトル: 状態エンコーディングの方法
内容1: バイナリエンコーディング
- 各状態はバイナリで、0001、0010、0100、1000の4状態にエンコードされる。
- 状態に割り当てるコードによって論理回路の複雑さが変わ
内容2: ワンホットエンコーディング
- 1つのビットが1つの状態を表す。
- 1つだけのビットが1となり、残りは0となる。
""",
            "answer": """
バイナリエンコーディングとワンホットエンコーディングはデジタルシステム設計における重要な選択肢です。バイナリエンコーディングでは、各状態はバイナリでエンコードされ、例えば4状態は0001、0010、0100、1000のようによく表現される。状態に割り当てるコードによって、論理回路の複雑さが変わりますが、最適化は一般的に難しいです。
一方、ワンホットエンコーディングでは1つのビットが1つの状態を表し、ビットが1つだけ1となり、残りは0となります。このエンコーディング方法ではフリップフロップの数が多く必要となり、しかし次状態決定および出力決定のロジックが単純になる可能性があります。
""",
        }
    ]

    prompt2 = FewShotPromptTemplate(
        examples=examples2, example_prompt=example_prompt, suffix="Question: {input}", input_variables=["input"]
    )

    chain2 = LLMChain(llm=gpt, prompt=prompt2)

    # chainの連結
    overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

    # 実行
    return overall_chain(ocr_result)
