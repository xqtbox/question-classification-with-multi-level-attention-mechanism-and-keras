# question-classification-with-multi-level-attention-mechanism-and-keras
question classification with multi-level attention mechanism 使用多层级注意力机制和keras实现问题分类

[toc]

本文的部分工作、代码、数据共享到gethub网站《使用多层级注意力机制和keras实现问题分类》：https://github.com/xqtbox/question-classification-with-multi-level-attention-mechanism-and-keras

# 1 准备工作
## 1.1 什么是词向量?
”词向量”（词嵌入）是将一类将词的语义映射到向量空间中去的自然语言处理技术。即将一个词用特定的向量来表示，向量之间的距离（例如，任意两个向量之间的L2范式距离或更常用的余弦距离）一定程度上表征了的词之间的语义关系。由这些向量形成的几何空间被称为一个嵌入空间。

传统的独热表示（ one-hot representation）仅仅将词符号化，不包含任何语义信息。必须考虑将语义融入到词表示中。

解决办法将原来稀疏的巨大维度压缩嵌入到一个更小维度的空间进行分布式表示。

![image](https://img-blog.csdn.net/20170404205445720?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvU2NvdGZpZWxkX21zbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center) 

这也是词向量又名词嵌入的缘由了。


例如，“椰子”和“北极熊”是语义上完全不同的词，所以它们的词向量在一个合理的嵌入空间的距离将会非常遥远。但“厨房”和“晚餐”是相关的话，所以它们的词向量之间的距离会相对小。

理想的情况下，在一个良好的嵌入空间里，从“厨房”向量到“晚餐”向量的“路径”向量会精确地捕捉这两个概念之间的语义关系。在这种情况下，“路径”向量表示的是“发生的地点”，所以你会期望“厨房”向量 - “晚餐"向量（两个词向量的差异）捕捉到“发生的地点”这样的语义关系。基本上，我们应该有向量等式：晚餐 + 发生的地点 = 厨房（至少接近）。如果真的是这样的话，那么我们可以使用这样的关系向量来回答某些问题。例如，应用这种语义关系到一个新的向量，比如“工作”，我们应该得到一个有意义的等式，工作+ 发生的地点 = 办公室，来回答“工作发生在哪里？”。

词向量通过降维技术表征文本数据集中的词的共现信息。方法包括神经网络(“Word2vec”技术)，或矩阵分解。

## 1.2 获取词向量

词向量 对与中文自然语言处理任务 是基石，一般情况下 有两种获取方式：

- 别人训练好的百科数据。优势：包含词语多，符合日常用语的语义；劣势：专有名词不足，占用空间大；
- 自己训练。优势：专有名词，针对具体任务语义更准确； 劣势：泛化性差。

步骤：

```
graph LR
文本-->分词
分词-->训练词向量
训练词向量-->保存词向量
```


具体代码：
```python
import gensim

## 训练自己的词向量，并保存。
def trainWord2Vec(filePath):
    sentences =  gensim.models.word2vec.LineSentence(filePath) # 读取分词后的 文本
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4) # 训练模型

    model.save('./CarComment_vord2vec_100')


def testMyWord2Vec():
    # 读取自己的词向量，并简单测试一下 效果。
    inp = './CarComment_vord2vec_100'  # 读取词向量
    model = gensim.models.Word2Vec.load(inp)

    print('空间的词向量（100维）:',model['空间'])
    print('打印与空间最相近的5个词语：',model.most_similar('空间', topn=5))


if __name__ == '__main__':
    #trainWord2Vec('./CarCommentAll_cut.csv')
    testMyWord2Vec()
    pass
```
这样我们就 拥有了 预训练的 词向量文件`CarComment_vord2vec_100` 。

下一单元 继续讲解如何在keras中使用它。

# 2 转化词向量为keras所需格式
上一步拿到了所有词语的词向量，但还需转化词向量为keras所需格式。众所周知，keras中使用预训练的词向量的层是`Embedding层`，而`Embedding层`中所需要的格式为 一个巨大的“矩阵”：第i列表示词索引为i的词的词向量

所以，本单元的总体思路就是给 Embedding 层提供一个 `[ word : word_vector]` 的词典来初始化`Embedding层`中所需要的大矩阵 ，并且标记为不可训练。



## 2.1 获取所有词语word和词向量

首先要导入 预训练的词向量。
```python
## 1 导入 预训练的词向量
myPath = './CarComment_vord2vec_100' # 本地词向量的地址
Word2VecModel = gensim.models.Word2Vec.load(myPath) # 读取词向量

vector = Word2VecModel.wv['空间']  # 词语的向量，是numpy格式
```
`gensim`的`word2vec`模型 把所有的单词和 词向量 都存储在了`Word2VecModel.wv`里面，讲道理直接使用这个`.wv`即可。 但是我们打印这个东西的 类型

```python
print(type(Word2VecModel.wv)) # 结果为：Word2VecKeyedVectors

for i,j in Word2VecModel.wv.vocab.items():
    print(i) # 此时 i 代表每个单词
    print(j) # j 代表封装了 词频 等信息的 gensim“Vocab”对象，例子：Vocab(count:1481, index:38, sample_int:3701260191)
    break

```
发现它是 gensim自己封装的一种数据类型：`Word2VecKeyedVectors`，
```
<class 'gensim.models.keyedvectors.Word2VecKeyedVectors'>
```
不能使用`for循环`迭代取单词。


## 2.2 构造“词语-词向量”字典

第二步 构造数据：
- 构造 一个list存储所有单词：vocab_list 存储所有词语。
- 构造一个字典word_index :`{word : index}` ，key是每个词语，value是单词在字典中的序号。  在后期 tokenize（序号化） 训练集的时候就是用该词典。
构造包含
- 构造一个 大向量矩阵embeddings_matrix （按照embedding层的要求）：行数 为 所有单词数，比如 10000；列数为 词向量维度，比如100。

代码：

```python
## 2 构造包含所有词语的 list，以及初始化 “词语-序号”字典 和 “词向量”矩阵
vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]# 存储 所有的 词语

word_index = {" ": 0}# 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
word_vector = {} # 初始化`[word : vector]`字典

# 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
```
## 2.3 填充字典和矩阵

第三步：填充 上述步骤中 的字典 和 大矩阵

```python
## 3 填充 上述 的字典 和 大矩阵
for i in range(len(vocab_list)):
    # print(i)
    word = vocab_list[i]  # 每个词语
    word_index[word] = i + 1 # 词语：序号
    word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
    embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵
```

## 2.4 在 keras的Embedding层中使用 预训练词向量


```python
from keras.layers import Embedding

EMBEDDING_DIM = 100 #词向量维度

embedding_layer = Embedding(input_dim = len(embeddings_matrix), # 字典长度
                            EMBEDDING_DIM, # 词向量 长度（100）
                            weights=[embeddings_matrix], # 重点：预训练的词向量系数
                            input_length=MAX_SEQUENCE_LENGTH, # 每句话的 最大长度（必须padding） 
                            trainable=False # 是否在 训练的过程中 更新词向量
                           
                            )
```

- Embedding层的输入shape

此时 输入Embedding层的数据的维度是
形如`（samples，sequence_length）`的2D张量，注意，此时句子中的词语word已经被转化为 index（依靠`word_index`，所以在 `embedding层`之前 往往结合 `input层`， 用于将 文本 分词 转化为数字形式）


- Embedding层的输出shape

Embedding层把 所有输入的序列中的整数，替换为对应的词向量矩阵中对应的向量（也就是它的词向量）,比如一句话[1,2,8]将被序列[词向量第[1]行,词向量第[2]行,词向量第[8]行]代替。

这样，输入一个2D张量后，我们可以得到一个3D张量：`(samples, sequence_length, embeddings_matrix)`

## *2.5 不使用“预训练”而直接生成词向量

我们也可以直接使用Keras自带的Embedding层训练词向量，而不用预训练的word2vec词向量。代码如下所示：


```python
embedding_layer = Embedding(len(word_index) + 1, # 由于 没有预训练，设置+1 
                            EMBEDDING_DIM, # 设置词向量的维度
                            input_length=MAX_SEQUENCE_LENGTH) #设置句子的最大长度
```
可以看出在使用 Keras的中Embedding层时候，不指定参数`weights=[embeddings_matrix]` 即可自动生成词向量。

先是随机初始化，然后，在训练数据的过程中训练。

在参考文献1中 做的对比实验，对于新闻文本分类任务：直接使用Keras自带的Embedding层训练词向量而不用预训练的word2vec词向量，得到0.9的准确率。

使用预训练的word2vec词向量，同样的模型最后可以达到0.95的分类准确率。

所以使用预训练的词向量作为特征是非常有效的。一般来说，在自然语言处理任务中，当样本数量非常少时，使用预训练的词向量是可行的（实际上，预训练的词向量引入了外部语义信息，往往对模型很有帮助）。


# 3 整体代码：在Keras模型中使用预训练的词向量

文本数据预处理，将每个文本样本转换为一个数字矩阵，矩阵的每一行表示一个词向量。下图梳理了处理文本数据的一般步骤。
![image](https://ask.qcloudimg.com/http-save/yehe-1654149/lklnss2gke.jpeg?imageView2/2/w/1620)



## 3.1 读取数据

```python
def load_file():
    dataFrame_2016 = pd.read_csv('data\\nlpcc2016_kbqa_traindata_zong_right.csv',encoding='utf-8')
    print(dataFrame_2016.columns) # 打印列的名称

    texts = []   # 存储读取的 x
    labels = []  # 存储读取的y
    # 遍历 获取数据
    for i in range(len(dataFrame_2016)):
        texts.append(dataFrame_2016.iloc[i].q_text) # 每个元素为一句话“《机械设计基础》这本书的作者是谁？”
        labels.append(dataFrame_2016.iloc[i].q_type) # 每个元素为一个int 代表类别 # [2, 6, ... 3] 的形式

    ## 把类别从int 3 转换为(0,0,0,1,0,0)的形式
    labels = to_categorical(np.asarray(labels)) # keras的处理方法，一定要学会# 此时为[[0. 0. 1. 0. 0. 0. 0.]....] 的形式
    return texts, labels # 总文本，总标签
```
## 3.2 句子分词

```python
## 2. cut_sentence2word 句子分词
def cut_sentence2word(texts):
    texts = [jieba.lcut(Sentence.replace('\n', '')) for Sentence in texts] # 句子分词
    return texts
```

## 3.3 *构造词向量字典

```python
## 3.获取word2vec模型， 并构造，词语index字典，词向量字典
def get_word2vec_dictionaries(texts):
    def get_word2vec_model(texts=None): # 获取 预训练的词向量 模型，如果没有就重新训练一个。
        if os.path.exists('data_word2vec/Word2vec_model_embedding_25'): # 如果训练好了 就加载一下不用反复训练
            model = Word2Vec.load('data_word2vec/Word2vec_model_embedding_25')
            # print(model['作者'])
            return model
        else:
            model = Word2Vec(texts, size = EMBEDDING_LEN, window=7, min_count=10, workers=4)
            model.save('data_word2vec/Word2vec_model_embedding_25') # 保存模型
            return model

    Word2VecModel = get_word2vec_model(texts) #  获取 预训练的词向量 模型，如果没有就重新训练一个。

    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]  # 存储 所有的 词语


    word_index = {" ": 0}# 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
    word_vector = {} # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # 词向量矩阵

    return word_index, word_vector, embeddings_matrix
```

## 3.4 文本序号化Tokenizer

在上文中已经得到了每条文本的文字了，但是text-CNN等深度学习模型的输入应该是数字矩阵。可以使用Keras的Tokenizer模块实现转换。


简单讲解Tokenizer如何实现转换。当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。可能这时会有疑问：Tokenizer是如何判断文本的一个词呢？其实它是以空格去识别每个词。因为英文的词与词之间是以空格分隔，所以我们可以直接将文本作为函数的参数，但是当我们处理中文文本时，我们需要使用分词工具将词与词分开，并且词间使用空格分开。具体实现如下：

![image](https://ask.qcloudimg.com/http-save/yehe-1654149/ou9hi0yyev.jpeg?imageView2/2/w/1620)

当然 ，也可以使用之前构建的`word_index`字典，手动 构建文本tokenizer句子：（推荐这种方法，这样序号下标与预训练词向量一致。）

```python
# 序号化 文本，tokenizer句子，并返回每个句子所对应的词语索引
def tokenizer(texts, word_index):

    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)
            
        data.append(new_txt)

    texts = sequence.pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return texts
```
## 3.5 切分数据

```python
## 5.切分数据
def split_data(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test
```
## 3.6 使用Embedding层将每个词编码转换为词向量
通过以上操作，已经将每个句子变成一个向量，但上文已经提及text-CNN的输入是一个数字矩阵，即每个影评样本应该是以一个矩阵，每一行代表一个词，因此，需要将词编码转换成词向量。使用Keras的Embedding层可以实现转换。

![image](https://ask.qcloudimg.com/http-save/yehe-1654149/z1jhticqn5.jpeg?imageView2/2/w/1620)

需要声明一点的是Embedding层是作为模型的第一层，在训练模型的同时，得到该语料库的词向量。当然，也可以使用已经预训练好的词向量表示现有语料库中的词。




```python
embedding_layer = Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = EMBEDDING_LEN,  # 词向量 长度（25）
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=MAX_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'embedding_layer'
                                )
```

然后利用 keras的建模能力，把Embedding层嵌入到 模型中去即可。后面可以接CNN或者LSTM


# 参考文献

> 参考《Keras的中Embedding层 官方文档》：  https://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/

> 参考1 官方文档《在Keras模型中使用预训练的词向量》：https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/

> 参考2 ：《Keras 模型中使用预训练的 gensim 词向量（word2vec） 和可视化》 https://eliyar.biz/using-pre-trained-gensim-word2vector-in-a-keras-model-and-visualizing/

> 参考3《Embedding原理和Tensorflow-tf.nn.embedding_lookup()》：https://blog.csdn.net/laolu1573/article/details/77170407


