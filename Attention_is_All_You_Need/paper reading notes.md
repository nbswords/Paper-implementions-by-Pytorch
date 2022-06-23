# Attention Is All You Need

[Paper Link](https://arxiv.org/abs/1706.03762)

---
## Intro

* 近年來序列模型的Sota方法都是RNN,LSTM,GRU等模型
* 即使在引入Attention機制做Seq2Seq之後, 仍然是以RNN,CNN作為Encoder/Decoder來建造模型
  * [Visualize Seq2Seq with Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* 但這種模型一直因為RNN的序列結構而使訓練速度受限, 即使之後有許多論文嘗試解決, 問題仍然存在
* Attention機制在此之前都是和RNN之類的模型一起使用 , 從未有一個模型只有Attention
* 本篇論文提出的就是一個只有encoder-decoder和attention機制且可以平行化訓練的模型 - **Transformer**

## Model Architecture

### Encoder and Decoder
* Encoder/Decoder 是甚麼?  能幹嘛?
  * 一種解決Seq2Seq問題的模型
    * Seq2Seq是輸入一個序列x, 生成另一個序列y  
  * Encoder將一串輸入序列轉成一個固定長度的向量  
    $x_1,...,x_n$ &rarr; $z_1,...,z_n$ 
  
  * Decoder將encoder輸出的向量再轉回序列
      $z_1,...,z_n$ &rarr; $y_1,...,y_n$ 
* Auto-regrssive
  * 將前一個產生的symbol作為產生下一個symbol的輸入
  * 類似RNN中殘差的概念
* 在Encode和decoder中由Self-Attention和Pointwise還有fc layer所組成

![](https://i.imgur.com/JBqsUsH.png)


(1) 左邊的Encoder : 每一層有2個sublayers 總共有六層
 * 由Multi-head self-attention 和 FC Feed Forward所組成
 * 每兩個sublayers之間用Residual connection連接

(2) 右邊的decoder : 基於原本的Encoder 2層上再加一層sublayers, 每一層都有3個sublayers 總共有六層
 * Multi-head self-attention (Encoder的Output) 和 FC Feed Forward
 * **Masked multi-head self-attention** : 將上一個decoder的output作為當前decoder的input , 並確保位置i的輸出只會依照小於i的input輸出  (只讓decoder attend到已經產生出來的seqence)
 * 一樣有用Residual連接

(3)
* **add&norm**: add是將Multi-head attention的input和output相加, norm是[Layer normalization](https://arxiv.org/abs/1607.06450)
   * Layer norm = 給一筆data, 使其各個不同dimension的data的mean=0, sigma=1
   * Batch norm = 使整個batch中同一dimension的mean=0, sigma=1



### Attention

* 注意力function做的事情可被描述為mapping一個query和一群key-value的pairs到輸出 (Q, K, V, Output都是vector)
* Output is computed as a weighted sum of the values
* 而values的weight則是由query和其相對應的key所算出
* **Q,K,V簡單來說就是 Q是在找哪個字的key vector可能會貢獻我的語意最多 , K是這個字可以貢獻給哪個字最多語意, V是最後的輸出 也就是這個字的語意是什麼**

---
**補充解釋 by 李宏毅:** [Video Link](https://www.youtube.com/watch?v=ugWDIIOHtPA)
Q : to match other
K : to be matched
V : information to be extracted
Attention就是吃兩個向量 輸出一個分數來代表這兩個向量有多匹配、多相關
Self Attention: 拿每個q去對每個k做Attention (Scaled Dot-product attention)

![](https://i.imgur.com/JW804EO.jpg)

* Matrix O 就是Self Attention的輸出

---
### Scaled Dot-Product Attention: Attention的計算方式

* 它們自己的attention function 
* function的input有queries和keys的dimension$d_k$ 還有values的 dimension $d_k$
* 先將$Q$和所有的$K$做內積之後再除以$\sqrt{d_k}$ 最後加上softmax映射成機率之後就當成$V$的Weight (如圖)
  * 需要注意的是 Q的計算和這個function的計算是同時進行的
  ![](https://i.imgur.com/hanF6VK.png)
![](https://i.imgur.com/mAyAYhh.jpg)[video link](https://www.youtube.com/watch?v=aButdUV0dxI&list=PLvOO0btloRntpSWSxFbwPIjIum3Ub4GSC)
* 兩個常見的attention function有**additive attention**和**dot-product attention** 他們選擇dot-product的原因是因為它在實作上更快且更有效率 
  * 更快更有效率的原因:since it can be implemented using highly optimized matrix multiplication code. 
* 值得一提的是當$d_k$很小的時候兩種attention方法沒甚麼差別,但當$d_k$很大的時候,additive attention的表現會比dot-product還來得更好
  * 研究員對此的懷疑是在$d_k$很大的時候dot product的規模變太大,並導致softmax後的數值太小 (extremely small gradients), 因此才會加上$\sqrt{}$來緩解

![](https://i.imgur.com/4xMlLna.png)



### Multi-Head Attention : Transformer最重要的機制

* 其實就是很多個self-attention concat後所得出的結果
* 用來計算每一個字對當前這個字能給予多少的資訊量 (讓每個head都能學到不同feature的特徵)
 
![](https://i.imgur.com/uB3Rnno.jpg)


* 本篇共用個 $h=8$個平行的attention head (layers)
* 每一個 $d_k = d_v = d_model/h = 64$
* 由於每個head有降維 其實計算上並不會和single-head差太多

### Application of Attention in Transformer

**3 Diffent ways using multi-head attention**
* Encoder-Decoder attention layers
  * 在這一層中 , query來自上一層decoder layer, 而key和value則是來自Encoder的輸出
  * 這可以讓每個位置的Decoder都能考慮到目前所有輸入的序列
* Self-attention layer in **Encoder**
  * 在這一層中 所有的key, values, queries都是從同一個地方來的, 他們都是從前一層的encoder的output來的
* Self-attnention layer in **Decoder**
  * 這一層中 q,k,v都能attend目前位置之前的序列, 且為了避免attend到目前位置後面的序列, 會用masking(設為負無限)濾掉所有未知的輸出


### Position-wise Feed-Forward Networks

* 除了在sub-layers以外, encoders和decoders中的每一層都有Feed-Forward Network(FFN)
* FFN由兩個線性變換和ReLU組成 如圖
![](https://i.imgur.com/9RwC3F5.jpg)
* 雖然Apply到每個不同位置的線性變換都是一樣的, 但層和層之間是用不同的參數丟入FFN之中
* 也可將FFN看成kernel size=1的2個Conv
* 實際上與 MobileNetV2 的 linear bottleneck 相同

### Embeddings and Softmax

* Transformer跟以前的Seq2Seq model一樣有用embedding去將input token轉成$d$維的vector (廢話)
* 一樣有用softmax將decoder的輸出轉為機率去預測下一個token會是甚麼的機率 (一樣廢話)
* 兩個embedding之間和softmax之前的線性變換如同[此篇論文用法](https://arxiv.org/abs/1608.05859)一樣, 他們將weight乘上$\sqrt{d_{model}}$
* 使用 softmax 的原因是比較符合我們對於語言的假設，也就是說一句話應該只有一部分是相較於其他部分都更能表達語意的關鍵句，最終就是會有一個勝者
    * 可思考如果換成 sigmoid 的話，就會變成一句話有多個重要的地方要看


### Postional Encoding

![](https://i.imgur.com/iXUcYIv.jpg)

* 因為Transformer中沒有Conv跟Recurrent, **沒有東西可以表示token在序列中的"相對位置或是絕對位置"** 
  * 也就是說Attention機制沒有考慮"順序" 
* 因此他在encoder和decoder最底層的input embedding加上**positional encodings**來表示位置 
 ![](https://i.imgur.com/BDvNjGb.jpg)
* $a^i = x^i * W^I$
* $e^i + a^i = x^i * W^I + P^i * W^p$
  * $P^i$是一個表示位置的One-hot vector 
- Attention is all you need 模型是使用**Sinusoidal position encoding**，雖然可以描述相對位置關係，但這種編碼方式本質上是絕對位置編碼
  - 在 [Encoding word order in complex embeddings, 2020](https://arxiv.org/pdf/1912.12333.pdf) 中提到 Learned Positional Embedding 和 Sinusoidal Position Encoding 兩種在位置編碼方法在 Transformer 中表現沒什麼差別，而使用 Complex embedding 獲得的提升則大於前兩者
- 所謂的 10000 指的是最大不重複字句的數量

![](https://i.imgur.com/RM9lIx0.jpg)

## Why Self-attention

* RNN/CNN時間複雜度太大
* RNN/CNN計算無法平行化運算的困境 
* RNN/CNN無法執行長程記憶的困境 
* 總之在比時間複雜度


## Conclusion

* Transformer是第一個完全僅基於Attention的Seq2Seq model , 之前的研究都是只把attention用在decoder上
* 在機器翻譯的task上比RNN Seq2Seq model更快且更好
* 計畫未來讓Transformer也能用在video, images , audio上


## Additional Reference

- [Theaisummer](https://theaisummer.com/transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformers: Attention in Disguise](https://www.mihaileric.com/posts/transformers-attention-in-disguise/)