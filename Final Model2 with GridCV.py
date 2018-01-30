
# coding: utf-8

# In[101]:


import csv
import pandas as pd
import parser
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


# In[2]:


df1=pd.read_csv("twitter_dec.csv", encoding="Latin-1")
df2=pd.read_csv("twitter_dec2.csv", encoding="Latin-1")
df3=pd.read_csv("twitter_dec3.csv", encoding="Latin-1")
df4=pd.read_csv("twitter_dec4.csv", encoding="Latin-1")
df5=pd.read_csv("twitter2.csv", encoding="Latin-1")


# In[3]:


df=pd.concat([df1, df2, df3, df4, df5], ignore_index=True)


# In[4]:


len(df)


# In[5]:


df=df.dropna(axis=0, how="all")
df=df.reset_index()


# In[6]:


r=df.groupby(["place country"]).size()[df.groupby(["place country"]).size()>10]


# In[7]:


country= [r.index[i] for i in range(len(r.index))]


# In[8]:


len(country)


# In[9]:


df=df[df["place country"].isin(country)]


# In[10]:


len(df)


# In[11]:


def extract_hashtag(text):
    return ', '.join(set(tag.strip("#") for tag in text.split() if tag.startswith("#")))

df["Hashtags"]=df["text"].apply(extract_hashtag)


# In[12]:


import re

def extract_username(text):
    return ", ".join(re.findall("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)",text, re.I ))
    

df["Username_in_text"]= df["text"].apply(extract_username)


# In[13]:


def clean_tweet(text):
    return  ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

df["Clean_text"]=df["text"].apply(clean_tweet)


# In[14]:


df=df.reset_index(drop=True)


# In[36]:


import random
index=random.sample(range(len(df)), int(0.2*len(df)))
df_training=df[~df.index.isin(index)]
df_test=df[df.index.isin(index)]


# In[37]:


len(df_training.groupby(["place country"]).size()[df_training.groupby(["place country"]).size()>10])


# In[38]:


df_training=df_training.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)


# In[39]:


df_training.columns


# In[40]:


#a=df_training[df_training["place country"]=="United States"]
print(a["coordinate-coordinates-latitude"].mean(),a["coordinate coordinates longitude"].mean())


# In[41]:


print(len(df_training), len(df_test))


# In[42]:


def seconds(text):
    time_str=text.split(" ")[1]
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
df_training["seconds"]= df_training["created_at"].apply(seconds)


# In[43]:


df_training_new=df_training[['source', 'user description','user lang','user followers_count', 'user friends_count', 'user utc_offset','Hashtags', 'Clean_text','seconds','place country', 'coordinate coordinates longitude', 'coordinate-coordinates-latitude']]  
df_training_new[['user followers_count', 'user friends_count']]=df_training_new[['user followers_count', 'user friends_count']].apply(pd.to_numeric, errors='ignore')
df_training_new[["user description"]]=df_training_new[["user description"]].astype(str)


# In[47]:


from collections import Counter
import numpy as np

# aggregation functions
agg_funcs = {'Clean_text' : lambda x: ' '.join(x),
             'user description' : lambda x: ' '.join(x),
             'user lang': lambda x: Counter(x).most_common(1)[0][0],
             'Hashtags' : lambda x: ' '.join(x),
             'user followers_count' : np.mean,
             'user friends_count' : np.mean,
             'source' : lambda x: Counter(x).most_common(1)[0][0],
             'user utc_offset': lambda x: Counter(x).most_common(1)[0][0],
             'seconds' : np.median,
             'coordinate coordinates longitude': np.mean,
             'coordinate-coordinates-latitude' : np.mean}

# Groupby 'screen_name' and then apply the aggregation functions in agg_funcs
df_1 = df_training_new.groupby(['place country']).agg(agg_funcs).reset_index()


# In[48]:


df1.columns


# In[49]:


df_training_new=df_1


# In[50]:


df_training_new.loc[df_training_new['user lang'].isin(['en-GB','en-gb','sr',"fil"]), 'user lang'] = "en"
df_training_new.loc[df_training_new['user lang']=="es-MX", 'user lang'] = "es"
df_training_new.loc[df_training_new['user lang'].isin(['zh-CN','zh-TW','zh-cn','zh-tw']), 'user lang'] = "zh"
df_training_new.loc[df_training_new['user lang'].isin(['uk']), 'user lang'] = "nl"


# In[51]:


set(df_training_new["user lang"].tolist())


# In[52]:


import json

with open('stopwords-all.json', encoding="UTF-8") as json_data:
    stopwords_all = json.load(json_data)


# In[53]:


f = lambda x: x["user lang"] in ["fil","sr","uk", "hr","pl","sk","sv", "ja"] and x["Clean_text"] or 'Yet to be Filled'

df_training_new["removed_stopwords"]=df_training_new.apply(f, axis=1)


# In[54]:


def remove_stopwords(row):
    return ' '.join([word.lower() for word in row["Clean_text"].split() if word.lower() not in stopwords_all[row["user lang"]]])

df_training_new["removed_stopwords"]=df_training_new.apply(remove_stopwords, axis=1)


# In[55]:


import snowballstemmer

def stemming(row):
    if(row["user lang"]=="ca" or row["user lang"]=="eu"):
        stemmer = snowballstemmer.SpanishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])    
    if(row["user lang"]=="da"):
        stemmer = snowballstemmer.DanishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="nl"):
        stemmer = snowballstemmer.DutchStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="en" or row["user lang"]=="fi"):
        stemmer = snowballstemmer.EnglishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="fu"):
        stemmer = snowballstemmer.FinnishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="fr"):
        stemmer = snowballstemmer.FrenchStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="de"):
        stemmer = snowballstemmer.GermanStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="hu"):
        stemmer = snowballstemmer.HungarianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="it"):
        stemmer = snowballstemmer.ItalianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="no"):
        stemmer = snowballstemmer.NorwegianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="pt"):
        stemmer = snowballstemmer.PortugueseStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="ro"):
        stemmer = snowballstemmer.RomanianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="ru"):
        stemmer = snowballstemmer.RussianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="es"):
        stemmer = snowballstemmer.SpanishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])
#    if(row["user lang"]=="sv"):
#        stemmer = snowballstemmer.SwedishStemmer(word)
#        return ' '.join([stemmer.stemWord() for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="tr"):
        stemmer = snowballstemmer.TurkishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])

df_training_new["removed_stopwords_stemmed"]=df_training_new.apply(stemming, axis=1)


# In[56]:


def stemming_user_description(row):
    if(row["user lang"]=="ca" or row["user lang"]=="eu"):
        stemmer = snowballstemmer.SpanishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["removed_stopwords"].split()])    
    if(row["user lang"]=="da"):
        stemmer = snowballstemmer.DanishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="nl"):
        stemmer = snowballstemmer.DutchStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="en" or row["user lang"]=="fi"):
        stemmer = snowballstemmer.EnglishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="fu"):
        stemmer = snowballstemmer.FinnishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="fr"):
        stemmer = snowballstemmer.FrenchStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="de"):
        stemmer = snowballstemmer.GermanStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="hu"):
        stemmer = snowballstemmer.HungarianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="it"):
        stemmer = snowballstemmer.ItalianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="no"):
        stemmer = snowballstemmer.NorwegianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="pt"):
        stemmer = snowballstemmer.PortugueseStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="ro"):
        stemmer = snowballstemmer.RomanianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="ru"):
        stemmer = snowballstemmer.RussianStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    if(row["user lang"]=="es"):
        stemmer = snowballstemmer.SpanishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])
    #if(row["user lang"]=="sv"):
    #    stemmer = snowballstemmer.SwedishStemmer(word)
    #    return ' '.join([stemmer.stemWord() for word in row["removed_stopwords"].split()])
    if(row["user lang"]=="tr"):
        stemmer = snowballstemmer.TurkishStemmer()
        return ' '.join([stemmer.stemWord(word) for word in row["user description2"].split()])


# In[58]:


df_training_new["user description2"]=df_training_new["user description"].apply(clean_tweet)
df_training_new["Hashtags"]=df_training_new["Hashtags"].apply(clean_tweet)

def remove_stopwords_user_description(row):
    return ' '.join([word.lower() for word in row["user description2"].split() if word.lower() not in stopwords_all[row["user lang"]]])

df_training_new["user_description2"]=df_training_new.apply(remove_stopwords_user_description, axis=1)
df_training_new["user_description2"]=df_training_new.apply(stemming_user_description, axis=1)


# In[208]:


df_training_new.groupby("user lang").size()
df_training_new["user lang"].str.get_dummies().columns


# In[59]:


df_training_new=df_training_new.join(df_training_new["user lang"].str.get_dummies())
df_training_new = df_training_new.drop('user lang', axis=1)


# In[72]:


df_training_new=df_training_new[~df_training_new["removed_stopwords_stemmed"].isnull()]


# In[ ]:


##New Model


# In[95]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

             

doc=df_training_new[["removed_stopwords_stemmed","Hashtags","place country", 'coordinate-coordinates-latitude', 'coordinate coordinates longitude']]

doc["text"]= doc["removed_stopwords_stemmed"] + " " + doc["Hashtags"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(doc["text"].tolist())


# In[ ]:


#K-Means K=1


# In[96]:


true_k = len(doc)
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
clusters = km.labels_.tolist()


# In[97]:


clus = pd.Series(clusters)
doc['Cluster']=clus.values


# In[121]:


cluster_mapping=doc[["place country","Cluster"]]


# In[107]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                alpha=1e-3, n_iter=5, random_state=42))])
text_clf_svm.fit(doc["text"], doc["Cluster"].values)
#predicted_svm = text_clf_svm.predict(twenty_test.data)
#np.mean(predicted_svm == twenty_test.target)


# In[ ]:


##K-means =65 (5 groups)


# In[258]:


k = 163
new_km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10)
new_km.fit(X)
new_clusters = new_km.labels_.tolist()


# In[259]:


new_clus = pd.Series(new_clusters)
doc['new_Cluster']=new_clus.values


# In[260]:


a=doc.groupby(["new_Cluster"])["place country"].apply(lambda x: list(x))


# In[261]:


new_cluster_mapping=pd.DataFrame({'new_Cluster':a.index, 'place country':a.values})


# In[262]:


new_cluster_mapping["place country"].apply(lambda x: len(x)).tolist()


# In[263]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

new_text_clf_svm = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                alpha=1e-3, n_iter=5, random_state=42))])
new_text_clf_svm.fit(doc["text"], doc["new_Cluster"].values)
#predicted_svm = text_clf_svm.predict(twenty_test.data)
#np.mean(predicted_svm == twenty_test.target)


# In[148]:


#Non-linear SVM
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)

nonlinear_clf_svm = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('rbf', RBFSampler(gamma=1, random_state=1)),
                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                alpha=1e-3, n_iter=None, random_state=42))])
nonlinear_clf_svm.fit(doc["text"], doc["Cluster"].values)


# In[109]:


#Linear SVC

linear_SVC = LinearSVC(class_weight='balanced',penalty='l2')
linear_SVC.fit(X, doc["Cluster"].values)


# In[110]:


#Non-linear SVC

Nonlinear_SVC= svm.NuSVC(kernel='rbf',nu=0.01)
Nonlinear_SVC.fit(X, doc["Cluster"].values)


# In[ ]:


#Testing set


# In[112]:


df_test["seconds"]= df_test["created_at"].apply(seconds)


# In[113]:


df_test_new=df_test[['source', 'user description','user lang','user followers_count', 'user friends_count', 'user utc_offset','Hashtags', 'Clean_text','seconds','place country', 'coordinate coordinates longitude', 'coordinate-coordinates-latitude']]  
df_test_new[['user followers_count', 'user friends_count']]=df_test_new[['user followers_count', 'user friends_count']].apply(pd.to_numeric, errors='ignore')
df_test_new[["user description"]]=df_test_new[["user description"]].astype(str)


# In[ ]:


# Groupby 'screen_name' and then apply the aggregation functions in agg_funcs
df_2 = df_test_new.groupby(['place country']).agg(agg_funcs).reset_index()


# In[ ]:


df_test_new=df_2


# In[114]:


df_test_new.loc[df_test_new['user lang'].isin(['en-GB','en-gb','sr',"fil"]), 'user lang'] = "en"
df_test_new.loc[df_test_new['user lang']=="es-MX", 'user lang'] = "es"
df_test_new.loc[df_test_new['user lang'].isin(['zh-CN','zh-TW','zh-cn','zh-tw']), 'user lang'] = "zh"
df_test_new.loc[df_test_new['user lang'].isin(['uk']), 'user lang'] = "nl"
df_test_new.loc[df_test_new['user lang'].isin(['pt-PT']), 'user lang'] = "pt"


# In[115]:


def select_lang(x):
    if(x in df_training_new.columns):
        return True
    else:
        return False

df_test_new=df_test_new[df_test_new["user lang"].apply(select_lang)]


# In[116]:


df_test_new["removed_stopwords"]=df_test_new.apply(f, axis=1)


# In[117]:


df_test_new["removed_stopwords"]=df_test_new.apply(remove_stopwords, axis=1)


# In[118]:


df_test_new["removed_stopwords_stemmed"]=df_test_new.apply(stemming, axis=1)


# In[119]:


df_test_new["user description2"]=df_test_new["user description"].apply(clean_tweet)
df_test_new["Hashtags"]=df_test_new["Hashtags"].apply(clean_tweet)

df_test_new["user_description2"]=df_test_new.apply(remove_stopwords_user_description, axis=1)
df_test_new["user_description2"]=df_test_new.apply(stemming_user_description, axis=1)


# In[130]:


df_test_new=df_test_new[~df_test_new["removed_stopwords_stemmed"].isnull()]


# In[ ]:


#Prediction


# In[132]:



test_doc=df_test_new[["removed_stopwords_stemmed","Hashtags","place country", 'coordinate-coordinates-latitude', 'coordinate coordinates longitude']]

test_doc["text"]= test_doc["removed_stopwords_stemmed"] + " " + test_doc["Hashtags"]


X_test = vectorizer.fit_transform(test_doc["text"].tolist())


# In[ ]:


#Linear SVM prediction


# In[134]:


predicted_svm = text_clf_svm.predict(test_doc["text"])
#np.mean(predicted_svm == test_doc["place country].values)


# In[138]:


test_label=[cluster_mapping[doc["Cluster"]==i]["place country"].values[0] for i in predicted_svm]


# In[142]:


sum(test_label==test_doc["place country"].values)


# In[143]:


len(test_label)


# In[154]:


joblib.dump(km, 'Clustering_final.pkl')
joblib.dump(text_clf_svm, 'text_clf_svm.pkl')


# In[155]:


df_training_new.to_csv("df_training_new_final.csv", index=False)
df_test_new.to_csv("df_test_new_final.csv", index=False)


# In[ ]:


##K-means =65


# In[264]:


new_predicted_svm = new_text_clf_svm.predict(test_doc["text"])
#np.mean(predicted_svm == test_doc["place country].values)


# In[265]:


new_test_label=[new_cluster_mapping[new_cluster_mapping["new_Cluster"]==i]["place country"].values[0] for i in new_predicted_svm]


# In[205]:


def accuracy(test_label,actual_label):
    correct=[]
    for i in range(len(new_test_label)):
        if(actual_label.values[i] in test_label[i]):
            correct.append(1)
    return(sum(correct))


# In[ ]:


#K means k=65 (5 groups)


# In[206]:


accuracy(new_test_label,test_doc["place country"])


# In[203]:


len(new_test_label)


# In[ ]:


#K means k=163 (2 group)


# In[215]:


accuracy(new_test_label,test_doc["place country"])


# In[ ]:


#K means k=108 (3 groups)


# In[223]:


accuracy(new_test_label,test_doc["place country"])


# In[266]:


accuracy(new_test_label,test_doc["place country"])


# In[363]:


len(new_test_label[1])


# In[ ]:


c=new_test_label[3]


# In[347]:


b=df_training_new[df_training_new["place country"].isin(c)]
country=b["place country"].values
text=b["removed_stopwords_stemmed"].values
latlong=list(zip(b["coordinate-coordinates-latitude"],b["coordinate coordinates longitude"]))


# In[348]:


from sklearn.metrics.pairwise import cosine_similarity


# In[349]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(text)  #finds the tfidf score with normalization
d=cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train) #here the first element of tfidf_matrix_train is matched with other three elements


# In[354]:


e=d.tolist()[0]
e.pop(0)
e


# In[357]:


index=e.index(max(e))


# In[299]:


index=d.tolist()[0].index(max(d.tolist()[0]))


# In[359]:


[country[index], latlong[index]]


# In[365]:


def cosine_prediction(new_test_label,test_sample, df_training_new):
    b=df_training_new[df_training_new["place country"].isin(new_test_label)]
    country=b["place country"].values
    text=b["removed_stopwords_stemmed"].values
    latlong=list(zip(b["coordinate-coordinates-latitude"],b["coordinate coordinates longitude"]))
    if(len(new_test_label)>1):
        new_text=[test_sample]+text
        tfidf_vectorizer=TfidfVectorizer()
        compare_tfidf= tfidf_vectorizer.fit_transform(new_text)
        #test_tfidf=tfidf_vectorizer.transform(test_sample)
        d=cosine_similarity(compare_tfidf[0:1], compare_tfidf)
        lst=d.tolist()[0]
        lst.pop(0)
        index=lst.index(max(lst))
        return([country[index], latlong[index]])
    else:
        return([country[0], latlong[0]])


# In[ ]:


test_samples=test_doc["text"].values
check=[]
for i,j in zip(new_test_label,range(len(test_samples))):
    check.append(cosine_prediction(i, test_samples[j],df_training_new))


# In[ ]:


#Non linear SVM


# In[149]:


nonlinear_predicted_svm = nonlinear_clf_svm.predict(test_doc["text"])
#np.mean(predicted_svm == test_doc["place country].values)


# In[152]:


non_linear_test_label=test_label=[cluster_mapping[doc["Cluster"]==i]["place country"].values[0] for i in nonlinear_predicted_svm]


# In[153]:


sum(non_linear_test_label==test_doc["place country"].values)


# In[156]:


len(test_label)


# In[210]:


#Ensemble Model


# In[211]:


def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr.tokenize(tweet)


# In[212]:


df_training_new=df_training_new[~df_training_new["removed_stopwords_stemmed"].isnull()]
randomized_df = df_training_new.sample(frac=1, random_state=111)

# Split randomized_df into two disjoint sets
half_randomized_df = int(randomized_df.shape[0] / 2)
base_df = randomized_df.iloc[:half_randomized_df, :]      # used to train the base classifiers
meta_df = randomized_df.iloc[half_randomized_df:, :]      # used to train the meta classifier

# Create variables for the known the geotagged locations from each set
base_y = base_df['place country'].values
meta_y = meta_df['place country'].values


# In[213]:


print(len(base_df),len(meta_df))


# In[214]:


# Raw text of user tweets
base_location_doc = base_df['removed_stopwords_stemmed'].values
meta_location_doc = meta_df['removed_stopwords_stemmed'].values

# fit_transform a tf-idf vectorizer using base_location_doc and use it to transform meta_location_doc
location_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2))
base_location_X = location_vectorizer.fit_transform(base_location_doc.ravel())
meta_location_X = location_vectorizer.transform(base_location_doc)#meta_location_doc

# Fit a Linear SVC Model with 'base_location_X' and 'base_y'. Note: it is important to use 
# balanced class weights otherwise the model will overwhelmingly favor the majority class.
location_SVC = LinearSVC(class_weight='balanced')
location_SVC.fit(base_location_X, base_y)

# We can now pass meta_location_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
location_SVC_decsfunc = location_SVC.decision_function(meta_location_X)

# Pickle the location vectorizer and the linear SVC model for future use
joblib.dump(location_vectorizer, 'USER_TWEETS.pkl')
joblib.dump(location_SVC, 'USER_LOCATION_SVC.pkl')


# In[215]:


print(len(location_SVC_decsfunc), print(location_vectorizer))


# In[216]:


df_training_new.isnull().sum()


# In[217]:


# Raw text of user description
base_description_doc = base_df['user description2'].values
meta_description_doc = meta_df['user description2'].values

# fit_transform a tf-idf vectorizer using base_location_doc and use it to transform meta_location_doc
description_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2))
base_description_X = description_vectorizer.fit_transform(base_description_doc.ravel())
meta_description_X = description_vectorizer.transform(base_description_doc)

# Fit a Linear SVC Model with 'base_location_X' and 'base_y'. Note: it is important to use 
# balanced class weights otherwise the model will overwhelmingly favor the majority class.
description_SVC = LinearSVC(class_weight='balanced')
description_SVC.fit(base_description_X, base_y)

# We can now pass meta_location_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
description_SVC_decsfunc = description_SVC.decision_function(meta_description_X)

# Pickle the location vectorizer and the linear SVC model for future use
joblib.dump(description_vectorizer, 'USER_description.pkl')
joblib.dump(description_SVC, 'USER_description_SVC.pkl')


# In[221]:


# Raw text of user description
base_hashtag_doc = base_df['Hashtags'].values
meta_hashtag_doc = meta_df['Hashtags'].values

# fit_transform a tf-idf vectorizer using base_location_doc and use it to transform meta_location_doc
hashtag_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2))
base_hashtag_X = hashtag_vectorizer.fit_transform(base_hashtag_doc.ravel())
meta_hashtag_X = hashtag_vectorizer.transform(base_hashtag_doc)

# Fit a Linear SVC Model with 'base_location_X' and 'base_y'. Note: it is important to use 
# balanced class weights otherwise the model will overwhelmingly favor the majority class.
hashtag_SVC = LinearSVC(class_weight='balanced')
hashtag_SVC.fit(base_hashtag_X, base_y)

# We can now pass meta_location_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
hashtag_SVC_decsfunc = hashtag_SVC.decision_function(meta_hashtag_X)

# Pickle the location vectorizer and the linear SVC model for future use
joblib.dump(hashtag_vectorizer, 'USER_hashtag.pkl')
joblib.dump(hashtag_SVC, 'USER_hashtag_SVC.pkl')


# In[222]:


df_training_new.isnull().sum()


# In[223]:


df_training_new.columns


# In[230]:


len(meta_df['tr'].values.reshape(meta_df.shape[0], 1))


# In[231]:


# additional features from meta_df to pull into the final model
friends_count = meta_df['user friends_count'].values.reshape(meta_df.shape[0], 1)
#utc_offset = meta_df['user utc_offset'].values.reshape(meta_df.shape[0], 1)
tweet_time_secs = meta_df['seconds'].values.reshape(meta_df.shape[0], 1)
followers_count = meta_df['user followers_count'].values.reshape(meta_df.shape[0], 1)
ca= meta_df['ca'].values.reshape(meta_df.shape[0], 1)
da= meta_df['da'].values.reshape(meta_df.shape[0], 1)
de=meta_df['de'].values.reshape(meta_df.shape[0], 1)
en= meta_df['en'].values.reshape(meta_df.shape[0], 1)
es=meta_df['es'].values.reshape(meta_df.shape[0], 1)
eu=meta_df['eu'].values.reshape(meta_df.shape[0], 1)
fi=meta_df['fi'].values.reshape(meta_df.shape[0], 1)
fr=meta_df['fr'].values.reshape(meta_df.shape[0], 1)
hr=meta_df['hr'].values.reshape(meta_df.shape[0], 1)
hu=meta_df['hu'].values.reshape(meta_df.shape[0], 1)
it=meta_df['it'].values.reshape(meta_df.shape[0], 1)
nl=meta_df['nl'].values.reshape(meta_df.shape[0], 1)
no=meta_df['no'].values.reshape(meta_df.shape[0], 1)
pl= meta_df['pl'].values.reshape(meta_df.shape[0], 1)
pt= meta_df['pt'].values.reshape(meta_df.shape[0], 1)
ja= meta_df['ja'].values.reshape(meta_df.shape[0], 1)
sk= meta_df['sk'].values.reshape(meta_df.shape[0], 1)
sv= meta_df['sv'].values.reshape(meta_df.shape[0], 1)
tr=meta_df['tr'].values.reshape(meta_df.shape[0], 1)

# np.hstack these additional features together
add_features = np.hstack((friends_count, 
                        #  utc_offset, 
                          tweet_time_secs,
                          followers_count,
                          ca,
                          da,
                          de,
                          en,
                          es,
                          eu,
                          fi,
                          fr,
                          hr,
                          hu,
                          it,
                          nl,
                          no,
                          pl,
                          pt,
                          ja,
                          sk,
                          sv,
                          tr))

# np.hstack the two decision function variables from steps 2 & 3 with add_features
meta_X = np.hstack((location_SVC_decsfunc,
                    description_SVC_decsfunc,        # from Step 2 above
                    hashtag_SVC_decsfunc,           # from Step 3 above
                    add_features))

# Fit Random Forest with 'meta_X' and 'meta_y'
meta_RF = RandomForestClassifier(n_estimators=80, n_jobs=-1)
meta_RF.fit(meta_X, meta_y)

# Pickle the meta Random Forest for future use
joblib.dump(meta_RF, 'META_RF.pkl')


# In[232]:


meta_RF


# In[ ]:


##Testing set


# In[262]:


df_test["seconds"]= df_test["created_at"].apply(seconds)


# In[349]:


df_test.columns


# In[350]:


df_test_new=df_test[['source', 'user description','user lang','user followers_count', 'user friends_count', 'user utc_offset','Hashtags', 'Clean_text','seconds','place full name','place country']]  
df_test_new[['user followers_count', 'user friends_count']]=df_test_new[['user followers_count', 'user friends_count']].apply(pd.to_numeric, errors='ignore')
df_test_new[["user description"]]=df_test_new[["user description"]].astype(str)


# In[235]:


# Groupby 'screen_name' and then apply the aggregation functions in agg_funcs
df_2 = df_test_new.groupby(['place country']).agg(agg_funcs).reset_index()


# In[283]:


df_test_new=df_2


# In[351]:


df_test_new.loc[df_test_new['user lang'].isin(['en-GB','en-gb','sr',"fil"]), 'user lang'] = "en"
df_test_new.loc[df_test_new['user lang']=="es-MX", 'user lang'] = "es"
df_test_new.loc[df_test_new['user lang'].isin(['zh-CN','zh-TW','zh-cn','zh-tw']), 'user lang'] = "zh"
df_test_new.loc[df_test_new['user lang'].isin(['uk']), 'user lang'] = "nl"
df_test_new.loc[df_test_new['user lang'].isin(['pt-PT']), 'user lang'] = "pt"


# In[353]:


def select_lang(x):
    if(x in df_training_new.columns):
        return True
    else:
        return False

df_test_new=df_test_new[df_test_new["user lang"].apply(select_lang)]


# In[354]:


df_test_new["removed_stopwords"]=df_test_new.apply(f, axis=1)


# In[355]:


df_test_new["removed_stopwords"]=df_test_new.apply(remove_stopwords, axis=1)


# In[356]:


df_test_new["removed_stopwords_stemmed"]=df_test_new.apply(stemming, axis=1)


# In[357]:


df_test_new["user description2"]=df_test_new["user description"].apply(clean_tweet)
df_test_new["Hashtags"]=df_test_new["Hashtags"].apply(clean_tweet)

df_test_new["user_description2"]=df_test_new.apply(remove_stopwords_user_description, axis=1)
df_test_new["user_description2"]=df_test_new.apply(stemming_user_description, axis=1)


# In[333]:


#df_test_new.groupby("user lang").size()
df_test_new["user lang"].str.get_dummies().columns
#len(df_test_new)


# In[334]:


df_training_new.columns


# In[358]:


df_test_new.columns


# In[359]:


df_test_new=df_test_new.join(df_test_new["user lang"].str.get_dummies())
df_test_new = df_test_new.drop('user lang', axis=1)


# In[360]:


len(df_test_new)


# In[362]:


df_test_new=df_test_new[~df_test_new["removed_stopwords_stemmed"].isnull()]


# In[341]:


class UserLocationClassifier: 
    
    def __init__(self):
        '''
        Load the stacked classifier's pickled vectorizers, base classifiers, and meta classifier
        '''
        
        self.tweets_vectorizer = joblib.load('USER_TWEETS.pkl')
        self.tweets_SVC = joblib.load('USER_LOCATION_SVC.pkl')
        
        self.description_vectorizer = joblib.load('USER_description.pkl')
        self.description_SVC = joblib.load('USER_description_SVC.pkl')        
        
        self.hashtag_vectorizer = joblib.load('USER_hashtag.pkl')
        self.hashtag_SVC = joblib.load('USER_hashtag_SVC.pkl') 
        
        self.meta_RF = joblib.load('META_RF.pkl')

    def predict(self, df):
        '''
        INPUT: Cleaned and properly formatted dataframe to make predictions for
        OUTPUT: Array of predictions
        '''
        # Get text from 'user_described_location' column of DataFrame
        tweets_doc = df['removed_stopwords'].values
        description_doc= df['user_description2'].values
        hashtag_doc=df["Hashtags"].values
        
        # Vectorize 'location_doc' and 'tweet_doc'
        tweets_X = self.tweets_vectorizer.transform(tweets_doc.ravel())
        description_X = self.description_vectorizer.transform(description_doc.ravel())
        hashtag_X=self.hashtag_vectorizer.transform(hashtag_doc.ravel())
        
        # Store decision functions for 'location_X' and 'tweet_X'
        tweets_decision_function = self.tweets_SVC.decision_function(tweets_X)
        description_decision_function = self.description_SVC.decision_function(description_X)
        hashtag_decision_function = self.hashtag_SVC.decision_function(hashtag_X)
        
        
        #df["hr"]=0
        #df["pl"]=0
        #df["ru"]=0
        # additional features from meta_df to pull into the final model
        friends_count = df['user friends_count'].values.reshape(df.shape[0], 1)
      #  utc_offset = df['user utc_offset'].values.reshape(df.shape[0], 1)
        tweet_time_secs = df['seconds'].values.reshape(df.shape[0], 1)
        followers_count = df['user followers_count'].values.reshape(df.shape[0], 1)
        ca= df['ca'].values.reshape(df.shape[0], 1)
        da= df['da'].values.reshape(df.shape[0], 1)
        de= df['de'].values.reshape(df.shape[0], 1)
        en= df['en'].values.reshape(df.shape[0], 1)
        es= df['es'].values.reshape(df.shape[0], 1)
        eu= df['eu'].values.reshape(df.shape[0], 1)
        fi= df['fi'].values.reshape(df.shape[0], 1)
        fr= df['fr'].values.reshape(df.shape[0], 1)
        hu= df['hu'].values.reshape(df.shape[0], 1)
        hr= df['hr'].values.reshape(df.shape[0], 1)
        it= df['it'].values.reshape(df.shape[0], 1)
        ja= df['ja'].values.reshape(df.shape[0], 1)
        nl= df['nl'].values.reshape(df.shape[0], 1)
        no= df['no'].values.reshape(df.shape[0], 1)
        pl= df['pl'].values.reshape(df.shape[0], 1)
        pt= df['pt'].values.reshape(df.shape[0], 1)
        sk= df['sk'].values.reshape(df.shape[0], 1)
        sv= df['sv'].values.reshape(df.shape[0], 1)
        tr= df['tr'].values.reshape(df.shape[0], 1)

        # np.hstack these additional features together
        add_features = np.hstack((friends_count, 
                                 # utc_offset, 
                                  tweet_time_secs,
                                  followers_count,
                                  ca,
                                  da,
                                  de,
                                  en,
                                  es,
                                  eu,
                                  fi,
                                  fr,
                                  hu,
                                  hr,
                                  it,
                                  nl,
                                  hr,
                                  no,
                                  pl,
                                  pt,
                                  sk,
                                  sv,
                                  tr))

        # np.hstack the two decision function variables from steps 2 & 3 with add_features
        meta_X = np.hstack((tweets_decision_function,
                            description_decision_function,        # from Step 2 above
                            hashtag_decision_function,           # from Step 3 above
                            add_features))


        # Feed meta_X into Random Forest and make predictions
        return self.meta_RF.predict(meta_X)


# In[342]:


if __name__ == "__main__":
    
    # Load evaluation_df into pandas DataFrame
    evaluation_df = df_test_new
    
    # Load UserLocationClassifier
    clf = UserLocationClassifier()
    
    # Get predicted locations
    predictions = clf.predict(evaluation_df)
    
    # Create a new column called 'predicted_location'
    evaluation_df.loc[:, 'predicted_location'] = predictions
    
    # Pickle the resulting DataFrame with the location predictions
    evaluation_df.to_pickle('evaluation_df_with_predictions.pkl')


# In[343]:


r=evaluation_df[["place country","predicted_location"]]


# In[345]:


(r["place country"]==r["predicted_location"]).sum()


# In[348]:


df_test.columns


# In[ ]:


#stanford


# In[315]:


from nltk.tag import StanfordNERTagger
import nltk


# In[316]:


st = StanfordNERTagger('C:\\Anaconda3\\stanford-ner-2017-06-09\\classifiers\\english.all.3class.distsim.crf.ser.gz',
					   'C:\\Anaconda3\\stanford-ner-2017-06-09\\stanford-ner.jar',
					   encoding='utf-8')


# In[375]:


text = "My location is Eugene, Oregon for most of the Year or in Sèoul, south korea depending on School holidays. My primary time zone is the Pacific time zone."


tokenized_text = nltk.word_tokenize(text)


# In[376]:


classified_text = st.tag(tokenized_text)


# In[377]:


values=[]
for word in classified_text:
    if(word[1]=="LOCATION"):
        values.append(word[0])#.encode('utf-8'))

#values[1].decode("utf-8")


# In[385]:


df_test_new["place country list"]=df_test_new["place full name"].apply(lambda x: x.split(", "))


# In[389]:


def location_model(text):
    tokenized_text = nltk.word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    values=[]
    for word in classified_text:
        if(word[1]=="LOCATION"):
            values.append(word[0])#.encode('utf-8'))
    return(values)


# In[ ]:


df_test_new["predicted_NER"]=df_test_new["removed_stopwords_stemmed"].apply(location_model)


# In[388]:


df_training_new.to_csv("df_training_new.csv", index=False)
df_test_new.to_csv("df_test_new.csv", index=False)


# In[ ]:


#LSI model


# In[ ]:


import logging, sys, pprint


# In[ ]:


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# In[ ]:


from gensim.corpora import TextCorpus, MmCorpus, Dictionary


# In[ ]:


#Lsi model


# In[ ]:


df_training_new=pd.read_csv("df_training_new.csv", encoding='Latin-1')
df_test_new=pd.read_csv("df_test_new.csv", encoding='Latin-1')
df_training_new=df_training_new.drop("Unnamed: 0", axis=1)
df_test_new=df_test_new.drop("Unnamed: 0", axis=1)


# In[ ]:


from nltk.tokenize import word_tokenize
gen_docs={}
lst_doc=[]
for country,text in zip(df_training_new["place country"].tolist(),df_training_new["removed_stopwords_stemmed"].tolist()):
    lst=[w.lower() for w in word_tokenize(text)]
    lst_doc.append(lst)
    gen_docs.update({country : lst})


# In[ ]:


len(lst_doc)


# In[ ]:


dictionary=corpora.Dictionary(lst_doc)


# In[ ]:


corpus=[dictionary.doc2bow(text) for text in lst_doc]


# In[ ]:


lsi_model=models.LsiModel(corpus, id2word=dictionary, num_topics=2)


# In[ ]:


df_test_new=df_test_new[~df_test_new["removed_stopwords_stemmed"].isnull()]


# In[ ]:


def predict(test_case):
    vec_bow = dictionary.doc2bow(test_case.lower().split())

    vec_lsi = lsi_model[vec_bow] 
    index = similarities.MatrixSimilarity(lsi_model[corpus]) 

    sims = index[vec_lsi] # perform a similarity query against the corpus
    test_dict=[]
    for i,j in zip(df_training_new["place country"].values,sims):
        test_dict.append([i,j])
    sorted_by_second = sorted(test_dict, key=lambda tup: tup[1], reverse=True)
    new_lst=[]
    for i in range(5):
        new_lst.append(sorted_by_second[i][0])
    return(new_lst)


# In[ ]:


predicted_list=[]
for test_case in df_test_new["removed_stopwords_stemmed"].values:
    lst=predict(test_case)
    predicted_list.append(lst)


# In[ ]:


correct=0
for i in range(len(df_test_new)):
    if(df_test_new["place country"].values[i] in predicted_list[i]):
        correct+=1

print(correct/len(df_test_new))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


K=range(362)
GridSearchCV(cv=None, error_score=...,
       estimator=SGDClassifier(C=1.0, cache_size=..., class_weight=..., coef0=...,
                     decision_function_shape='ovr', degree=..., gamma=...,
                     kernel='rbf', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=...,
                     verbose=False),
       fit_params=None, iid=..., n_jobs=1,
       param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
       scoring=..., verbose=...)

