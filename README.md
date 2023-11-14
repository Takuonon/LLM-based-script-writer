# LLM ベースの授業スライド → 授業原稿作成 AI

人工知能演習(2023 年度・第 1 ターム)の自由課題で作成したものです。

## 概要

![全体像](image/%E5%85%A8%E4%BD%93%E5%83%8F.png)
![役割](image/%E5%BD%B9%E5%89%B2.png)

- 授業スライドを入力すると、授業の原稿を出力します。
- 教科書のテキスト、図表 DB を作成・エンべディングし、RAG(Retrieval Augmented Generation)を実装しました。
- これにより原稿には、教科書由来の分かりやすい表現、正確な表現が積極的に使用されています。

詳細はこちらのスライドをご覧下さい。→[SlideShare のリンク](https://www.slideshare.net/TakuoTachibana1/llm-7cb3)

## 使用データ

YOLO の学習データ及び、RAG の取得先データとして以下の教材を使用しました。

- 坂井修一著・電子情報通信学会編「コンピュータアーキテクチャ」(2004 年、コロナ社)
- 坂井修一著「実践コンピュータアーキテクチャ」(2020 年、コロナ社)
- 坂井修一著「2023 年度夏学期開講『コンピュータアーキテクチャ』授業スライド」

## コードの簡単な解説

API キーや学習データ/RAG データとして使っている坂井先生の教科書・授業スライドの流出を避けるためデータ類や一部のコードは載せていません。実際に動かすことは基本的にできないと思いますが、なるべくコードは載せているつもりです。

一部 drive のリンクを添付していますが、東大学内メアドのみで閲覧可能です。

### API キーについて

以下の有料 API を使用しています。(コードでは消しているのでこのままでは動きません。)

- google cloud vision api
- OpenAI api

### 大事な部分の解説

ファイル構成が複雑になっており、かつ整理し切れていない(まとまっていない)ので、重要なプロセスに対応するするファイルを示します。

#### 画像分離(YOLO)

- YOLO の学習データ

  自分で annotate した坂井先生の授業スライドを用いています。 →[drive](https://drive.google.com/drive/folders/1C5eWUI9phhVtZt1MaRNVQxD1j5bEFK3Y?usp=sharing)

- YOLO の訓練

  yolo.ipynb で行いました。訓練済みモデルも上の drive に置いてあります。

- YOLO の実行

  yolo.py で行いました。

#### メインプロセス(スライド要約)

- スライドから OCR で、テキスト抽出

  ocr.py で行いました。

- ocr 結果を整理

  generate.py 中の 1 つ目の chain で実行

- 整理した結果から教科書テキスト RAG 用の仮原稿を作成

  generate.py 中の 2 つ目の chain で実行

#### 教科書テキスト RAG

- ベクトル DB から retrieve

  text_rag.py で実行(L ２距離が最も近いものを取ってくる)

- 教科書テキストデータからベクトル DB を作成

  make_text_vector_DB.ipynb で実行

- 教科書テキストデータ

  テキストデータはこちらです →[ドライブ](https://docs.google.com/spreadsheets/d/1sHJdseAsHr3MiTPXirv_79-2E-GPxKkDgxV67LTplJY/edit?usp=drive_link)

#### 教科書図表 RAG

- 教科書テキストデータからベクトル DB を作成 / ベクトル DB から retrieve

  image_retrieval.ipynb で実行

- 教科書図表データ

  図表とテキストの対応データはこちらです →[ドライブ](https://docs.google.com/spreadsheets/d/19l9d0WhZmB3WDB3Ij5WrJCMjSv0mdRY2cGK4HK1r8m0/edit?usp=drive_link)

#### 最終的な原稿生成

- regenerate_with_text.py
- regenerate_with_image.py

  の 2 つのステップに分けて、最終的な原稿を生成しました。
