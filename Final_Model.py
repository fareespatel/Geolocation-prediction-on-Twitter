
# coding: utf-8

# In[82]:


import csv
import pandas as pd
import parser
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


# In[11]:


df1=pd.read_csv("twitter_dec.csv", encoding="Latin-1")
df2=pd.read_csv("twitter_dec2.csv", encoding="Latin-1")
df3=pd.read_csv("twitter_dec3.csv", encoding="Latin-1")
df4=pd.read_csv("twitter_dec4.csv", encoding="Latin-1")
df5=pd.read_csv("twitter2.csv", encoding="Latin-1")


# In[168]:


df=pd.concat([df1, df2, df3, df4, df5], ignore_index=True)


# In[169]:


len(df)


# In[170]:


df=df.dropna(axis=0, how="all")
df=df.reset_index()


# In[171]:


r=df.groupby(["place country"]).size()[df.groupby(["place country"]).size()>10]


# In[172]:


country= [r.index[i] for i in range(len(r.index))]


# In[173]:


len(country)


# In[174]:


df=df[df["place country"].isin(country)]


# In[175]:


len(df)


# In[176]:


def extract_hashtag(text):
    return ', '.join(set(tag.strip("#") for tag in text.split() if tag.startswith("#")))

df["Hashtags"]=df["text"].apply(extract_hashtag)


# In[177]:


import re

def extract_username(text):
    return ", ".join(re.findall("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)",text, re.I ))
    

df["Username_in_text"]= df["text"].apply(extract_username)


# In[178]:


def clean_tweet(text):
    return  ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

df["Clean_text"]=df["text"].apply(clean_tweet)


# In[183]:


df=df.reset_index(drop=True)


# In[188]:


import random
index=random.sample(range(len(df)), int(0.2*len(df)))
df_training=df[~df.index.isin(index)]
df_test=df[df.index.isin(index)]


# In[193]:


len(df_training.groupby(["place country"]).size()[df_training.groupby(["place country"]).size()>10])


# In[190]:


df_training=df_training.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)


# In[192]:


print(len(df_training), len(df_test))


# In[194]:


def seconds(text):
    time_str=text.split(" ")[1]
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
df_training["seconds"]= df_training["created_at"].apply(seconds)


# In[195]:


df_training_new=df_training[['source', 'user description','user lang','user followers_count', 'user friends_count', 'user utc_offset','Hashtags', 'Clean_text','seconds','place country']]  
df_training_new[['user followers_count', 'user friends_count']]=df_training_new[['user followers_count', 'user friends_count']].apply(pd.to_numeric, errors='ignore')
df_training_new[["user description"]]=df_training_new[["user description"]].astype(str)


# In[196]:


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
             'seconds' : np.median}

# Groupby 'screen_name' and then apply the aggregation functions in agg_funcs
df_1 = df_training_new.groupby(['place country']).agg(agg_funcs).reset_index()


# In[197]:


df_training_new=df_1


# In[198]:


df_training_new.loc[df_training_new['user lang'].isin(['en-GB','en-gb','sr',"fil"]), 'user lang'] = "en"
df_training_new.loc[df_training_new['user lang']=="es-MX", 'user lang'] = "es"
df_training_new.loc[df_training_new['user lang'].isin(['zh-CN','zh-TW','zh-cn','zh-tw']), 'user lang'] = "zh"
df_training_new.loc[df_training_new['user lang'].isin(['uk']), 'user lang'] = "nl"


# In[200]:


set(df_training_new["user lang"].tolist())


# In[202]:


import json

with open('stopwords-all.json', encoding="UTF-8") as json_data:
    stopwords_all = json.load(json_data)


# In[203]:


f = lambda x: x["user lang"] in ["fil","sr","uk", "hr","pl","sk","sv", "ja"] and x["Clean_text"] or 'Yet to be Filled'

df_training_new["removed_stopwords"]=df_training_new.apply(f, axis=1)


# In[204]:


def remove_stopwords(row):
    return ' '.join([word.lower() for word in row["Clean_text"].split() if word.lower() not in stopwords_all[row["user lang"]]])

df_training_new["removed_stopwords"]=df_training_new.apply(remove_stopwords, axis=1)


# In[205]:


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


# In[206]:


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


# In[207]:


df_training_new["user description2"]=df_training_new["user description"].apply(clean_tweet)
df_training_new["Hashtags"]=df_training_new["Hashtags"].apply(clean_tweet)

def remove_stopwords_user_description(row):
    return ' '.join([word.lower() for word in row["user description2"].split() if word.lower() not in stopwords_all[row["user lang"]]])

df_training_new["user_description2"]=df_training_new.apply(remove_stopwords_user_description, axis=1)
df_training_new["user_description2"]=df_training_new.apply(stemming_user_description, axis=1)


# In[208]:


df_training_new.groupby("user lang").size()
df_training_new["user lang"].str.get_dummies().columns


# In[209]:


df_training_new=df_training_new.join(df_training_new["user lang"].str.get_dummies())
df_training_new = df_training_new.drop('user lang', axis=1)


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


#Check below line


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

