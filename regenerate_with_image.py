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
    「はじめに、アセンブリ言語について解説します。通常、機械語のプログラムは命令を適切な順序で並べたものであり、それらの命令は2進数で表現されます。しかし、2進数の並びは正確ではありますが、人間が理解することは困難です。そこで考案されたのが、アセンブリ言語です。このアセンブリ言語を用いると、機械語のプログラムを英語に近い記号で表現することが可能になります。一つのアセンブリ言語の命令は機械語と1対1で対応しており、人間にとって理解しやすい表記法となっています。

次に、命令セットについて学びましょう。コンピュータの命令は主に算術論理演算命令、メモリ操作命令、分岐命令のようなものに大別できます。初めに、算術論理演算命令について説明します。これらの命令は、ALUといった演算器への制御命令を変えることで実現されます。これにより、計算を制御することが可能になります。

次に、データ移動命令についてです。データ移動命令とは、例えばメモリとレジスタ間のデータ移動を制御するための命令です。データ移動の命令としては、メモリーレジスタ間のデータ移動だけ考えれば良く、これにより、効率的なプログラムの動作を設計することができます。

最後に、分岐命令についてです。分岐命令は、コンピュータの命令実行の順序を変更する命令です。分岐命令は、無条件分岐命令と条件分岐命令の2つに大別できます。これらの命令により、プログラムの進行方向や進行条件を制御することが可能になります。

授業の内容は以上となります。アセンブリ言語と各種命令セットの理解を深めて、プログラムの基礎を学んでいきましょう。」
    上記が授業の原稿であるが、これに図表の説明を加えてほしい。以下が教科書から抜き出した図表の説明だから、これに忠実に図表の説明を書き足して。


    - そこで,英語に近い記号で機械語のプログラムを表現することが考案された。これをアセンブリ言語(assembly language)による表現という。アセンプリ言語による表現の例を,図6.3に示す。
    - 表6.1および表6.2の各項目で,addとかsubとか並んでいるものは,アセンプリ言語の操作コード(場合によってauxフィールドを一部含む場合がある)である。
    - 典型的な算術演算命令の一覧を表6.1に,論理演算命令の一覧を表6.2に示す。算術論理演算命令は,レジスタ間の演算か,レジスタと即値との間の演算のどちらかに限る。したがって,命令形式は,R型かI型となる。

    今回の授業中には「図6.3,表6.1,表6,2」が登場しているからこれらについて追記して。スライドに図表の番号はついていないので、図表に言及するときは、「表では」あるいは「図では」のような表現に改めること。ベースの原稿に変化は加えなくて良い。
    """,
        }
    ],
)
print(response["choices"][0]["message"]["content"])
