## Logical Thinking Meeting (20190531)

### Indexing:
- [テーマに関しての考え](#テーマに関しての考え)
- [既存VisualDialogの関連データセット](#既存VisualDialogの関連データセット)
- [Dialog 評価指標](#Dialog-評価指標)
- [Twitterからデータを集めて洗うプロセス](#Twitterからデータを集めて洗うプロセス)
- [論述構造と同時予測による論述的な意見生成](#論述構造と同時予測による論述的な意見生成)
- [Todo](#Todo)
- [References](#References)

---
### テーマに関しての考え
#### 論理性に関しての理解
*会話の論理性*
- 大規模データセットにより**Natural Dialogue**の特徴をとらえる
- 画像との関連付けが相対的しやすい

*論題に関しての意見（賛成，反対）や根拠となる理由の説明*
- 大規模データセットはまず構築しにくい
- 分野が絞られる
- 画像との関連付けが相対的にしにくい

*まとめ*
- 以上のことから，会話に含まれる論理性を学習することを考える

#### 視覚情報と会話
*視覚情報の種類*
- 画像
‐ 画像序列
- ビデオ
- などなど

*画像序列やビデオと会話*
- 幅広いタイプの会話を取り扱う場合，画像序列やビデオなどと会話内容は時間軸で関連性が付けにくい
‐ 画像序列やビデオと会話の内容の一致性が高い場面を想像すると，Instruction類しか考えられない(cooking, navigationなど)
‐ Instruction類を取り扱うとVideoBERTとぶつかる

*１枚の画像と会話１*
- 画像：二人が会話しているシーン
- 会話：その二人の会話内容
- 画像と会話の関連付け予想１：画像のシーン（Office, Classroom, Campus, Factory, Airplane）と会話内容
- 画像と会話の関連付け予想２：画像中の感情成分(Sadness, Happiness)と会話の内容や感情
- 画像と会話の関連付け予想３：画像中の詳細内容成分(Season, Object, Time)と会話の内容や詳細の一致性
- まとめ：画像と会話のより高次元的な特徴を関連させる；
- Tip 1：BERTを使う場合，画像の低次元から高次元までの特徴を階層的に抽出し，それぞれ固定長のベクトルにエンディングし，BERTに入力することはあり？
- Tip 2：画像とキーワードを入れて文章の分布をある程度絞る方が良いかも

*１枚の画像と会話２*
- 画像：幅広い任意画像
- 会話：画像に関する会話；Question-Answerの形式ではなく，Goal-DrivenとFree-Talkの間に位置付ける会話が良い
- 評価指標：１.話に関しての従来のNLP系評価指標；2.Human評価；３.　**どうやって定量的に自動的に会話と画像の一致性を評価する？**　生成した会

---
### 既存VisualDialogの関連データセット

#### [GuessWhat?](https://arxiv.org/pdf/1611.08481.pdf)
- 画像源：MS COCO 
- 会話源：CrowdSourced
- 画像枚数：66,537
- 会話数：155,280
- １会話あたりのターン数：5
- 内容：画像中に特定物体を特定するための会話；Question-Answer形式;AnswerがYes/NOだけ
- 提案年：2016

#### [VisDial1.0](https://arxiv.org/pdf/1611.08669.pdf)
- 画像源：MS COCO 
- 会話源：CrowdSourced
- 画像枚数：120K
- 会話数：1.2M
- １会話あたりのターン数：10
- 内容：画像内容を理解するためのQuestion-Answer10ターン
- 提案年：2016

#### [IGC(Image-Grounded Conversation)](https://arxiv.org/pdf/1701.08251.pdf)
*Traning 集*
- 画像源：Twitter Firehose
- 会話源：Twitter Firehose
- 画像枚数：250K
- 会話数：250K
- １会話あたりのターン数：3
- 内容：goal-drivenとfree-talkの間に位置付けるImage-Grounded free talk
- 提案年：2017
- **最もやりたいことと近いが；トレーニング集が公開されていない**

*Val, Test 集*
- 画像源：MS COCO
- 会話源：Crowdsourced
- 画像枚数：4,222
- 会話数：25,332
- １会話あたりのターン数：-
- **最低限データセットとしてそのまま使えるかもしれない**

#### [AVSD(Audio Visual Scene-Aware Dialog)](https://arxiv.org/pdf/1901.09107.pdf)
- 画像源：Charades dataset (AMT, daily activities) ビデオ
- 会話源：Crowdsourced
- 画像枚数：11,816
- 会話数：11,816
- １会話あたりのターン数：10
- 内容：ビデオ内容理解のためのQuestion-Answer10ターン
- 提案年：2018

#### まとめ
- Twitterから画像と会話ペアを大量に集めて洗うことによりデータを集める
- Twitterにあがっている画像はCaptionも同時に含むことが多くて，ある程度会話の分布を絞れる
- 画像から会話全体を予測することが従来研究が少ない
- TwitterのDialog研究があるが，Visual Dialogがいまだにほぼないので，研究する価値がある

---
### Dialog 評価指標
#### 評価指標に関する論文
- [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/abs/1603.08023)
- [∆BLEU: A Discriminative Metric for Generation Tasks with Intrinsically Diverse Targets](https://arxiv.org/pdf/1506.06863.pdf)


#### Goal-driven dialog and Goal-free dialog
*Goal-driven dialog*
- Examples: booking a flight
- Evaluation 1: task-completion rate
- Evaluation 2: time to task completion
- The shorter the better
- 現状のVisual Dialogがこっちに近い

*Goal-free dialog*
- Examples: chit-chat
- Evaluation: The longer the better
- 我々の目標はこっち？それとも両方の間？

--- 
### Twitterからデータを集めて洗うプロセス
#### Twitterからデータを集める
- 基本的にはお金がかかる
- データノイズが大きい

#### IGCの洗うプロセス
- API : Twitter Firehose

*Constraints*
- 初期ターンが画像相関内容；２ターン目の会話は質問から始まるデータだけ残す
- Twitter発信者の制限：3か月間に最小30会話を交換した
- 最大文字数を80に制限
- 画像にURLがないように制限

*難しい点*
- 統計によりTwitterの会話の46%は会話する人達の前の会話歴史などと関連
- Screenshotやnon-photograph画像が多い


---
### 論述構造と同時予測による論述的な意見生成
*まとめ*

- 本研究では，議論の的となる論題に対して自動で意見を生成する意見生成において、論述構造との同時予測を行うことで、BLEUの自動評価から、テスト用データの意見側との一致度の向上がみられた一方、人手評価から、論述構造の内容面での向上は見られなかったが、レトリックの面での向上がみられた．

*感想*

- 大規模データセットの作成が難しい
- 詳細的な論理性と比べ最も抽象的で緩やかなな論理性の定義が良いかもしれない

---
### Todo
- 評価指標に関する論文を読む
- Twitterからデータを集めて洗うプロセスを調べる

---
### References
- [Visual Dialog](https://github.com/qiuyue1993/Notes/blob/master/VisualDialog/Paper_Summarize/Paper-Summarize_Visual-Dialog.md)
- [Visual Storytelling](https://github.com/qiuyue1993/Notes/blob/master/Language%20and%20Vision/Paper%20Summarize/Visual%20Storytelling.md)
- [BERT](https://github.com/qiuyue1993/Notes/blob/master/NLP/Paper_Summarize/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding.md)
- [VideoBERT](https://github.com/qiuyue1993/Notes/blob/master/NLP/Paper_Summarize/VideoBERT-A-Joint-Model-for-video-and-language-representation-learning.md)

---
