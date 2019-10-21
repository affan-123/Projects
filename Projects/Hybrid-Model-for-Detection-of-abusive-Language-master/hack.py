from distutils.errors import LibError

import pandas as pd
import re
mainstr="Bastard shut the fuck up"

mainarray=[]

df=pd.read_csv ("C:\\Users\\AFFAN SHAIKH\\Downloads\\crowdflower-hate-speech-identification\\original\\data.csv",encoding='iso-8859-1')


def clean_tweet(tweet):
     return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w+:\ / \ / \S+)| (\S*\d\S*)", " ", tweet).split())
df['tweet'] = df['tweet'].apply(lambda x:clean_tweet(x))

X=df['tweet']
Y=df['label']
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
cachedStopWords = stopwords.words("english")
def remove_stopwords(sentence):
    words=word_tokenize(sentence)
    words=[w for w in words if w not in cachedStopWords]
    sentence=" ".join(words)
    return sentence
df['tweet']=df['tweet'].apply(lambda x: remove_stopwords(x))
mainstr=remove_stopwords(mainstr)
Ycon=df['confidence']

xtrain,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4)
print("Using CountVectorizer :")
from sklearn.feature_extraction.text import CountVectorizer
vect_cv = CountVectorizer(min_df=5, ngram_range=(1,3)).fit(xtrain)
X_train_vectorized=vect_cv.transform(xtrain)



from sklearn.linear_model import LogisticRegression
LRCV=LogisticRegression()
LRCV.fit(X_train_vectorized,y_train)

predictionsLR=LRCV.predict(vect_cv.transform(X_test))
from sklearn.metrics import roc_auc_score
print ("Logistics Regression:",roc_auc_score(y_test,predictionsLR))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
RF=RandomForestClassifier(n_estimators=150)
RF.fit(X_train_vectorized,y_train)
predictionsRF=RF.predict(vect_cv.transform(X_test))
print ("Random Forest:",roc_auc_score(y_test,predictionsRF))

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
KNN.fit(X_train_vectorized,y_train)
predictionsKNN=KNN.predict(vect_cv.transform(X_test))
print ("KNN:",roc_auc_score(y_test,predictionsKNN))


if(LRCV.predict(vect_cv.transform([mainstr]))):
    print("Output: Its an abusive sentence")
    mainarray.append(1)
else:
    print("Output: It is not an abusive sentence")
    mainarray.append(0)

print("Using Tfidf :")
from sklearn.feature_extraction.text import TfidfVectorizer

vect_tf = TfidfVectorizer(min_df=5).fit(xtrain)
X_train_vectorized = vect_tf.transform(xtrain)
LRTF= LogisticRegression()
LRTF.fit(X_train_vectorized, y_train)

predictions = LRTF.predict(vect_tf.transform(X_test))

print('Logistic Regression: ', roc_auc_score(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
RF=RandomForestClassifier(n_estimators=150)
RF.fit(X_train_vectorized,y_train)
predictionsRF=RF.predict(vect_tf.transform(X_test))
print ("RandomForest:",roc_auc_score(y_test,predictionsRF))

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
KNN.fit(X_train_vectorized,y_train)
predictionsKNN=KNN.predict(vect_tf.transform(X_test))
print ("KNN:",roc_auc_score(y_test,predictionsKNN))

if(LRTF.predict(vect_tf.transform([mainstr]))):
    print("Output: Its an abusive sentence")
    mainarray.append(1)
else:
    print("Output: It is not an abusive sentence")
    mainarray.append(0)


X=df['tweet']
from  sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,random_state=22,test_size=0.2)

from gensim.models import Word2Vec as wc
sentences=[]
for w in df['tweet']:
      sentences.append(w.split())
model1=wc(sentences,min_count=1)
#print(model1['Warning'])
X_TRAIN=[]
import numpy as np
#Sent=df['tweet']
for i in xtrain:
    sentence=[]
    for words in i.split():
        temp=model1[words]

        sentence.append(temp)
        from keras.preprocessing import sequence

    X_TRAIN.append(sentence)
X_TRAIN = sequence.pad_sequences(X_TRAIN, maxlen=30, value=0.0, padding='post',dtype='float32')


X_TRAIN=np.asarray(X_TRAIN)
from keras.layers import Dense,Dropout
from keras import Sequential
model=Sequential()
from keras.layers.embeddings import  Embedding
from keras.layers import LSTM



model.add(Dense(30,input_shape=(30,100)))
model.add(LSTM(30))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size = 128

model.fit(X_TRAIN,ytrain, batch_size=batch_size, epochs=100)
model.save('abusive_v3.h5')
X_TEST=[]
for i in xtest:
    sentence=[]
    for words in i.split():
        temp=model1[words]

        sentence.append(temp)
        from keras.preprocessing import sequence

    X_TEST.append(sentence)
X_TEST = sequence.pad_sequences(X_TEST, maxlen=30, value=0.0, padding='post',dtype='float32')


X_TEST=np.asarray(X_TEST)



score=model.evaluate(X_TEST,ytest,verbose=0)
from sklearn.metrics import classification_report
print (model.predict(X_TEST))
#print(classification_report(ytest,np.asarray(model.predict(X_TEST)).ravel()))
#print(score[1]*100)


# from keras.models import load_model
# model=load_model('abusive.h5')
# score=model.evaluate(xtest,ytest,verbose=0)
# print(score[1]*100)

import numpy as np
#str="Thats a bad product"
strvec=[]
for w in mainstr.split():
    strvec.append(model1[w])
strvec=np.asarray(strvec)
strvec1=[]
strvec1.append(strvec)
strvec1=np.asarray(strvec1)
from keras.preprocessing import sequence
strvec1 = sequence.pad_sequences(strvec1, maxlen=30, value=0.0, padding='post',dtype='float32')

x=np.where(model.predict(strvec1)>0.5,1,0)
if(x):
    mainarray.append(1)
else:
    mainarray.append(0)
from statistics import mode
print (mainarray)
print (mode(mainarray))



pred_LR_CV=np.asarray(LRCV.predict(vect_cv.transform(xtest)))

pred_LR_TF=np.asarray(LRTF.predict(vect_tf.transform(xtest)))

pred_LSTM=np.asarray(np.where(model.predict(X_TEST)>0.5,1,0)).ravel()
print(pred_LR_CV.shape,pred_LR_TF.shape,pred_LSTM.shape)
print('TF')
print(pred_LR_TF)
print('LSTM')
print(pred_LSTM)

# hybrid=[]
# hybrid.append(pred_LR_CV)
# hybrid.append(pred_LR_TF)
# hybrid.append(pred_LSTM)
# hybrid =np.asarray(hybrid)
# print("hybrid")
# print (hybrid)
#
hybrid=[]
for i in range (2902):
    temp=[]
    temp.append(pred_LR_CV[i])
    temp.append(pred_LR_TF[i])
    temp.append(pred_LSTM[i])
    temp=np.asarray(temp).ravel()
    hybrid.append(mode(temp))
hybrid=np.asarray(hybrid)
print("hybrid")
print(hybrid.shape)
print("final accuracy")
print(roc_auc_score(ytest,hybrid))


from sklearn.metrics import classification_report
print(classification_report(ytest,hybrid))
print(classification_report(ytest,pred_LSTM))

