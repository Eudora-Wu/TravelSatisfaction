#!/usr/bin/env python
# coding: utf-8

# 导入模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = "SimHei"
plt.rcParams['axes.unicode_minus'] = "False"
plt.rcParams['font.size'] = 15


# 导入、查看数据
data = pd.read_csv('data.csv')
#data.info()
data.isnull().sum()# 查看数据的NAN情况
data = data.drop(labels='Unnamed: 0',axis=1)# 去除多余索引列
data
data.drop_duplicates(inplace=True)#查找重复值并删除
print(data.duplicated().sum())



#文本清洗
import re
import string
def remove_punctuation(s):
#     print(s)
    delEStr = string.punctuation + ''+ string.digits + string.ascii_letters# 标点符号、数字、字母
    identify = str.maketrans(","," ",delEStr)# 这里感觉并没有用到映射的功能，而是着重使用到了 删除指定字符串 的功能
    s = s.translate(identify)
#     print(s,type(s))
    s = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\\\]¯～_`{|}~·—！，。？、￥…（）：；【】《》‘’“”\s]+",'',s) # 正则匹配并替换
    return s

data['content'] = data['content'].apply(remove_punctuation)
data.sample(5)

for i in data.iloc[:,0]:
    clean_data = remove_punctuation(i)
    print(clean_data)


#分词
import jieba 
def cut_word(text):
    return jieba.cut(text)
data['content']=data['content'].apply(cut_word)
data.sample(5)



#去停用词
def get_stopword():
    s = set()
    with open('stop_words.txt','r',encoding='utf-8') as f:
        for line in f:
            s.add(line.strip())
    return s 

def remove_stopword(words):
    return [word for word in words if word not in stopword]

stopword = get_stopword()
data['content'] = data['content'].apply(remove_stopword)
data.sample(5)


#描述性统计
t = data['sentiment'].value_counts()#情感分类柱状图
print(t)
t.plot(kind="bar")


from itertools import chain #词汇统计
from collections import Counter
li_2d = data['content'].tolist()
li_1d = list(chain.from_iterable(li_2d))
print(f'总词汇量:{len(li_1d)}')
c = Counter(li_1d)
print(f'不重复词汇量:{len(c)}')
common = c.most_common(15)
print(common)

      

d = dict(common)#词语频数
plt.figure(figsize=(15,5))
plt.bar(d.keys(),d.values())



total = len(li_1d)#词语频率
percentage = [v*100/total for v in d.values()]
plt.figure(figsize=(15,5))
plt.bar(d.keys(),percentage)




from wordcloud import WordCloud#词云
from PIL import Image
#wc = WordCloud(font_path=r'C:\Users\dell\Desktop\旅游\font.ttf',width=600,height=400)
#bg = WordCloud(mask = plt.imread(r'C:\Users\dell\Desktop\旅游\bgf.jpg'))
img = Image.open(r'bgf.jpg') # 打开背景图片
img_array = np.array(img) # 将图片转换为数组
wc = WordCloud(
    background_color="black", # 将背景颜色设置为黑色，也可根据个人喜好更改
    mask=img_array,#背景图
    max_words=100,
    font_path = r"font.ttf", #字体设置
    width=1000).generate_from_frequencies(c)
plt.imshow(wc)
plt.axis('off')
wc.to_file("pic.png")


#将列表转化为字符串
def join(text_list):
    return " ".join(text_list)

data["content"] = data["content"].apply(join)
data.sample(5)


#标签列转换为离散值
data["sentiment"] = data["sentiment"].map({"正":0,"中":1,"负":2})
data["sentiment"].value_counts()


#对数据打乱顺序并保存好训练和测试数据
data  = data.sample(frac=1).reset_index(drop=True)
data
data.shape[0]*0.8
train_data = data[:3051]
test_data = data[3051:]
train_data
test_data
train_data.info()
test_data.info()
x_train = train_data["content"]
y_train = train_data["sentiment"]

x_test = test_data["content"]
y_test = test_data["sentiment"]


#向量化和特征选择
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
x_train_tran = vec.fit_transform(x_train)
x_test_tran = vec.transform(x_test)
display(x_train_tran,x_test_tran)

from sklearn.feature_selection import f_classif
f_classif(x_train_tran,y_train)

from sklearn.feature_selection import SelectKBest

x_train_tran = x_train_tran.astype(np.float64)
x_test_tran = x_test_tran.astype(np.float64)
selector = SelectKBest(f_classif,k=min(1000,x_train_tran.shape[1]))
selector.fit(x_train_tran,y_train)
x_train_tran = selector.transform(x_train_tran)
x_test_tran = selector.transform(x_test_tran)
print(x_train_tran.shape,x_test_tran.shape)


#机器学习分类模型
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=False)
clf.fit(x_train_tran,y_train)  
pred = clf.predict(x_test_tran)
print(classification_report(y_test,pred))


from sklearn.tree import DecisionTreeClassifier
clf6 = DecisionTreeClassifier()
clf6.fit(x_train_tran,y_train)  
pred = clf6.predict(x_test_tran)
print(classification_report(y_test,pred))


from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier()
clf2.fit(x_train_tran,y_train)  
pred = clf2.predict(x_test_tran)
print(classification_report(y_test,pred))


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(x_train_tran, y_train)
pred = classifier.predict(x_test_tran)
print(classification_report(y_test,pred))


from sklearn.svm import SVC
clf3 = SVC(kernel='rbf')
clf3.fit(x_train_tran,y_train)  
pred = clf3.predict(x_test_tran)
print(classification_report(y_test,pred))



from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression()
clf4.fit(x_train_tran,y_train)  
pred = clf4.predict(x_test_tran)
print(classification_report(y_test,pred))


#CNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight as cw

from keras import Sequential

from keras.models import Model

from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,BatchNormalization,Add,concatenate,Flatten
from keras.layers import Conv1D,Conv2D,Convolution1D,MaxPool1D,SeparableConv1D,SpatialDropout1D,GlobalAvgPool1D,GlobalMaxPool1D,GlobalMaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import MaxPooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import load_model

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

x_train = train_data["content"]
y_train = train_data["sentiment"]

x_test = test_data["content"]
y_test = test_data["sentiment"]

x_train


y_train



# 分词器Tokenizer   Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
       # 类方法
           # fit_on_texts(texts) :texts用于训练的文本列表
          # texts_to_sequences(texts):texts待转为序列的文本列表 返回值:序列的列表，列表中的每个序列对应于一段输入文本

# 填充序列pad_sequences  将长为nb_smaples的序列转换为(nb_samples,nb_timesteps)2Dnumpy attay.如果提供maxlen,nb_timesteps=maxlen,否则其值为最长序列的长度。
# 其它短于该长度的序列都会在后部填充0以达到该长度。长与nb_timesteps的序列会被阶段，以使其匹配该目标长度。

#max_words = 1000
#max_len = 150
max_words = len(set(" ".join(x_train).split()))
max_len = x_train.apply(lambda x:len(x)).max()


tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(x_train)

sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


# one-hot编码
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_train

# 计算各个类别的weights
def get_weight(y):
    class_weight_current = cw.compute_class_weight("balanced",np.unique(y),y)
    return class_weight_current

class_weight = get_weight(y_train.flatten())


# - sklearn.utils.class_weight 样本均衡
#     - 当我们的数据，有多个类别，每个类别的数据量有很大差距时，这时需要对每个类别的样本做一次均衡，这样会让每个类别的特征都在一定程度上被模型学习

# - ModelCheckpoint：
#     - 作用：该回调函数将在每个epoch后保存模型到filepath
#     - 参数：
#         - filename:字符串，保存模型的路径，filepath可以是格式化的字符串，里面的
#         - monitor:需要监视的值，通常为:val_acc或val_loss或acc或loss
#         - verbose:信息展示模型，0或1。默认为0表示不输出该信息，为1表示输出epoch模型保存信息。
#         - save_best_only:当设置为Trur时，将只保存在验证集上性能最好的模型
#         - mode:"auto","min","max"之一，在save_best_only=True时决定性能最佳模型的评判准则。
#         - save_weights_only:若设置为True时，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
#         - period:CheckPoint之间的间隔的epoch数
#         
# - EarlyStopping：
#     - 作用：当监测值不再改善时，该回调函数将中止训练
#     - 参数：
#         - monitor:需要监视的量，通常为val_acc或val_loss或acc或loss
#         - patience:当early stop被激活（如发现loss相比上patience个epoch训练没有下降），则经过patience个epoch后停止训练。
#         - verbose:信息展示模型
#         - mode:"auto","min","max"之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
#         
# - ReduceLROnPlateau:
#     - 作用：当评价指标不再提升时，减少学习率。当学习停滞时，减少2倍或10倍的学习率通常能够获得较好的效果。该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率。
#     - 参数：
#         - monitor:被监测的量
#         - factor:每次减少学习率的因子，学习率将以lr=lr*factor的形式被技术那好
#         - patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
#         - mode:"auto","min","max"之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
#         - epsilon:阈值，用来确定是否进入检测值的“平原区”
#         - cooldown:学习率减少后，会经过cooldown个epoch才重新进行正常操作
#         - min_lr:学习率的下限。

# In[ ]:


print("Setting Callbacks")

checkpoint = ModelCheckpoint("model.h5",
                                                     monitor="val_acc",
                                                     save_best_only=True,
                                                     mode="max")

early_stopping = EarlyStopping(monitor="val_loss",
                                                     patience=6,
                                                     verbose=1,
                                                     restore_best_weights=True,
                                                     mode="min")

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                                      factor=0.6,
                                                      patience=2,
                                                      verbose=1,
                                                      mode="min")

callbacks=[checkpoint,early_stopping,reduce_lr]



# 定义CNN模型
def CNN():
    model=Sequential()
    
    model.add(Embedding(max_words,100,input_length=max_len))
    
    model.add(Conv1D(64,3,padding="valid",activation="relu",strides=1))
    model.add(Conv1D(128,3,padding="valid",activation="relu",strides=1))
    model.add(Conv1D(256,3,padding="valid",activation="relu",strides=1))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.25))    
    model.add(Dense(64,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dropout(0.25))    
    model.add(Dense(3,activation="softmax"))
    model.summary()
    return model




cnn_model = CNN()



# CNN模型训练
print("Starting...\n")


print("\n\nCompliling Model...\n")
learning_rate=0.001
optimizer=Adam(learning_rate)
cnn_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics)

verbose = 1
epochs=100
batch_size=8
validation_split=0.1
print("Trainning Model...\n")
cnn_history=cnn_model.fit(sequences_matrix,
                                            y_train,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            validation_split=validation_split,
                                            class_weight=class_weight)



# 将带预测数据转为序列
predict_sequences = tok.texts_to_sequences(x_test)
predict_sequences_matrix = sequence.pad_sequences(predict_sequences,maxlen=max_len)

# LDA模型（在分训练集和测试集前完成）
from keras.models import load_model
pred = cnn_model.predict_classes(predict_sequences_matrix)
print(classification_report(y_test,pred))

from gensim import corpora, models, similarities
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

data=data.loc[data['content'].apply(len)>=4]
data.info()

data['content'].apply(lambda x:x.split())
data['content'].to_csv(r'C:\Users\dell\Desktop\旅游.txt',index=False)

travel = open(r'C:\Users\dell\Desktop\旅游.txt', 'r',encoding='utf-8')
train = []
for line in travel.readlines():
  line = [word.strip() for word in line.split()]
  train.append(line)
travel.close()

dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4,passes=100)
topic_list = lda.print_topics(4,10)
print("4个主题的单词分布为：\n")
for topic in topic_list:
 print(topic)







