import numpy as np
import pandas as pd
import jieba
import os
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.layers import Input, Dense, Multiply,Flatten,RepeatVector,Permute,Dropout
from keras.layers import Conv1D,Bidirectional,LSTM
from keras.models import Model


# 超参数
EMBEDDING_LEN = 100   # 词向量长度
MAX_SEQUENCE_LENGTH = 10  # 句子的长度
FILTER_NUM = 5  #   卷积核的数量
FILTER_WINDOW_LEN = 3   #   卷积核的宽度

# 所有标签的字典
q_dict = {1: '描述', 2: '人物',
          3: '地点', 4: '数字',
          5: '时间', 6: '实体'}

### 子函数部分，5个

## 1.加载数据
def load_file():
    dataFrame_2016 = pd.read_csv('data\\nlpcc2016_kbqa_traindata_zong_right.csv',encoding='utf-8')
    print(dataFrame_2016.columns) # 打印列的名称

    texts = []   # 存储读取的 x
    labels = []  # 存储读取的y
    # 遍历 获取数据
    for i in range(len(dataFrame_2016)):
        texts.append(dataFrame_2016.iloc[i].q_text) # 每个元素为一句话“《机械设计基础》这本书的作者是谁？”
        labels.append(dataFrame_2016.iloc[i].q_type - 1) # 每个元素为一个int 代表类别 # [2, 6, ... 3] 的形式。减一为了 从0开始

    ## 把类别从int 3 转换为(0,0,0,1,0,0)的形式
    labels = to_categorical(np.asarray(labels)) # keras的处理方法，一定要学会# 此时为[[0. 0. 1. 0. 0. 0. 0.]....] 的形式
    return texts, labels # 总文本，总标签

## 2. cut_sentence2word 句子分词
def cut_sentence2word(texts):
    texts = [jieba.lcut(Sentence.replace('\n', '')) for Sentence in texts] # 句子分词
    return texts

## 3.获取word2vec模型， 并构造，词语index字典，词向量字典
def get_word2vec_dictionaries(texts):
    def get_word2vec_model(texts=None): # 获取 预训练的词向量 模型，如果没有就重新训练一个。
        if os.path.exists('data_word2vec/Word2vec_model_embedding_100'): # 如果训练好了 就加载一下不用反复训练
            model = Word2Vec.load('data_word2vec/Word2vec_model_embedding_100')
            # print(model['作者'])
            return model
        else:
            model = Word2Vec(texts, size = EMBEDDING_LEN, window=7, min_count=10, workers=4)
            model.save('data_word2vec/Word2vec_model_embedding_100') # 保存模型
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

## 4. 序号化 文本，tokenizer句子，并返回每个句子所对应的词语索引
def tokenizer(texts, word_index):
    yiwenci_lst = ['谁', '哪', '哪里', '哪个', '哪年', '哪天', '哪月', '何时', '如何', '何事', '何人', '何地', '多高', '多远', '多大', '多少', '怎样',
                   '怎么']  # 疑问词集合

    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)
            if word in yiwenci_lst:  # attentiom
                try:
                    new_txt.append(word_index[word])  #把句子中的 疑问词再加一遍
                except:
                    new_txt.append(0)
        data.append(new_txt)

    texts = sequence.pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return texts

## 5.切分数据
def split_data(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.05)
    return x_train, x_test, y_train, y_test

## 6. 定义网络结构CNN模型，训练，并保存
def train_mac_lstm(embeddings_matrix,x_train,y_train,x_test,y_test):
    print('定义Keras Model...')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32',name='sequence_input_later')

    embedding_layer = Embedding(input_dim=len(embeddings_matrix),  # 字典长度
                                output_dim = EMBEDDING_LEN,  # 词向量 长度（25）
                                weights=[embeddings_matrix],  # 重点：预训练的词向量系数
                                input_length=MAX_SEQUENCE_LENGTH,  # 每句话的 最大长度（必须padding） 10
                                trainable=False,  # 是否在 训练的过程中 更新词向量
                                name= 'embedding_layer'
                                )(sequence_input) # 128 x 10 x25

    print('疑问词注意力')
    InterrogativeAttention = Flatten(name='attention_Flatten')(embedding_layer) # 128x250
    InterrogativeAttention = Dense(MAX_SEQUENCE_LENGTH, activation='softmax',name='attention_Dense')(InterrogativeAttention) # 128 x 10
    InterrogativeAttention = RepeatVector(EMBEDDING_LEN,name = 'attention_repeat')(InterrogativeAttention)#128x25x10
    InterrogativeAttention = Permute((2,1))(InterrogativeAttention)# 重新排列张量的轴。keras.backend.permute_dimensions(x, pattern)其中(0, 2, 1)表示0轴不变，更换1轴和2轴# (?, 10, 25)
    x = Multiply()([embedding_layer, InterrogativeAttention]) # keras的merge层的新用法：https://keras-cn.readthedocs.io/en/latest/layers/merge/

    print('卷积层特征抽取')
    cnn1d = Conv1D(128, 5,
               border_mode='same',
               activation='relu')(x) #(None, 10, 128)
    print('注意力连接')
    connectAttention = Flatten(name='connectAttention_Flatten')(cnn1d) # (None, 1280)
    connectAttention = Dense(MAX_SEQUENCE_LENGTH, activation='softmax',name='connectAttention_Dense')(connectAttention) # (None, 10)
    connectAttention = RepeatVector(128 ,name = 'connectAttention_repeat')(connectAttention)# (None, 128, 10)
    connectAttention = Permute((2,1))(connectAttention)# 重新排列张量的轴。# (None, 10, 128)
    x = Multiply()([cnn1d, connectAttention]) # (None, 10, 128)


    print('双向LSTM抽取时序特征')
    x = Bidirectional(LSTM(256, activation='tanh',
                       recurrent_activation='hard_sigmoid',
                       use_bias=True,
                       ))(x)

    x = Dropout(0.4)(x)
    preds = Dense(len(q_dict), activation='softmax')(x)
    model = Model(sequence_input, preds)
    print(model.summary()) # 打印模型结构
    print('激活 the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    print("训练...")
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))

    print("评价...")
    score = model.evaluate(x_test, y_test, batch_size=32)
    print('Test score:', score)

    print('保存模型')
    model.save('data_model/my_model.h5')

if __name__ == '__main__':
    texts, labels = load_file() # 此时labels为numpy了，texts还是list需要进一步处理
    # 2. cutSentence2Word 句子分词
    texts = cut_sentence2word(texts)
    print(texts)
    # 3. 构造 词向量 字典 矩阵
    word_index, word_vector, embeddings_matrix = get_word2vec_dictionaries(texts)
    # print(word_index, word_vector, embeddings_matrix)
    # 4. 将文本转化为 index，
    texts = tokenizer(texts, word_index) # 此时texts也是numpy了，可以直接输入 模型计算了。
    print(texts[0])

    # 5.切分数据
    x_train, x_test, y_train, y_test = split_data(texts, labels)
    print(x_test[:3], y_test[:3])

    # 5.定义网络结构LSTM模型，训练，并保存
    print("5.定义网络结构LSTM模型，训练，保存，并输出精确度...")
    train_mac_lstm(embeddings_matrix, x_train, y_train, x_test, y_test)  # 精度为：0.92353
