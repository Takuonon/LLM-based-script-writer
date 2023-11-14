import openai
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import OpenAIModerationChain
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": """
    「スライド1枚目:
    タイトル: アセンブリ言語
内容1: プログラムの表記
- 機械語は読みにくいため、アセンブリ言語が使用されることがある。
内容2: アセンブリ言語の特性
- 英語に近い記号で表記される。
- 機械語と1対1で対応している。
スライド2枚目:
タイトル: 命令セット
内容1: 命令の分類
- 算術論理演算命令
- データ移動命令
- 分岐命令
内容2: 算術論理演算命令」
    のような流れで授業がしたい。教科書に以下のような表現があるから、なるべく教科書の表現を使って授業の原稿を作ってほしい。ただ、過不足があれば必要に応じて調整して。
    - そこで,英語に近い記号で機械語のプログラムを表現することが考案された。これをアセンブリ言語(assembly language)による表現という。アセンプリ言語による表現の例を,図6.3に示す。
    - 他の命令もALU(など演算器)への制御命令を変えることで同様に実現できる。
    - 機械語のプログラムは,命令を適切な順序で並べたものである。命令は2進数で表現されるから、プログラムは2進数の並んだものになる。2進数の並びは,正確だが人間が理解するのが困難である。
    - 分岐命令は,コンピュータの命令実行の順序を変更する命令である。無条件分岐命令と条件分岐命令に大別される。
    - コンピュータの命令は.算術論理演算命令.メモリ操作命令,分岐命令に大別される。これらの動作の概略は,5.2節および5.3節で示した。ここでは,個々の命令について,その表現形式と動作を学んでいく。
    - 以上より、この本ではデータ移動の命令として.メモリーレジスタ間のデータ移動だけを考えれば良いことになった。
    - 他の命令もALU(など演算器)への制御命令を変えることで同様に実現できる。

    ただ図番号や表番号、教科書の節の番号には絶対に言及しないこと。
    """,
        }
    ],
)
print(response["choices"][0]["message"]["content"])
