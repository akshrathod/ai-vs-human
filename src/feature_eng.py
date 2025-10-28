#!/usr/bin/env python
# coding: utf-8

# ## 5. Feature Engineering

# #### Import statements

# In[1]:


import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_excel('scraped_and_ai_data.xlsx')

# Load the preprocessed file
df = pd.read_excel('preprocessed_data.xlsx')
# Decode JSON strings back into Python lists
for col in ["Question", "Human Answer", "ChatGPT Answer"]:
    df[col] = df[col].apply(json.loads)


# ### A) Grammatical Mistakes

# In[3]:


import nltk
nltk.download('averaged_perceptron_tagger')

def gram_mistakes(tokens):

    # List to store number of grammatical mistakes
    mist = []

    n = len(tokens)

    for i in range(n):
        # POS tagging
        tagged = nltk.pos_tag(tokens[i])
        # Bigrams
        bi = list(nltk.bigrams(tagged))

        ### COMMON GRAMMATICAL MISTAKES ###

        # Preposition - Adverb pairs
        p_adv = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='IN' and y[1]=='RB'])

        # Verb - Adverb pairs
        vb_adv = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='VB' and y[1]=='RB'])     

        # Adverb - Adjective pairs
        adv_adj = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='RB' and y[1]=='JJ'])

        # Adjective - Adjective pairs
        adj_adj = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='JJ' and y[1]=='JJ']) 

        # Preposition - Preposition pairs
        p_p = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='IN' and y[1]=='IN'])

        # Pronoun - Pronoun pairs
        pro_pro = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='PRP' and y[1]=='PRP'])

        # Adverb - Adverb pairs
        adv_adv = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='RB' and y[1]=='RB'])

        # Adjective - Pronoun pairs
        adj_p = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='JJ' and y[1]=='PRP'])

        # Interjection - Conjunction pairs
        in_con = len([(x[0],y[0]) for (x,y) in bi \
         if x[1]=='UH' and y[1]=='IN'])

        # Ratio of total grammatical mistakes
        if len(bi)!=0:
            gram_mist = (p_adv + vb_adv + adv_adj + adj_adj + \
                        p_p + pro_pro + adv_adv + adj_p + in_con)/len(bi)*100
        else:
            gram_mist = 0

        # Storing ratio of grammatical mistakes
        mist.append(gram_mist)

    return mist


# In[4]:


gram_mist_human = gram_mistakes(df['Human Answer'])
gram_mist_chatgpt = gram_mistakes(df['ChatGPT Answer'])


# In[5]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Grammar Mistakes Human
axes[0].hist(gram_mist_human, bins= 20, facecolor='cadetblue', edgecolor='black');
axes[0].set_xlabel('Percent of Grammatical Mistakes');
axes[0].set_ylabel('Frequency');
axes[0].set_title('Percentage of Common Grammar Mistakes in Human Answers');

# Grammar Mistakes ChatGPT
axes[1].hist(gram_mist_chatgpt, bins= 20, facecolor='cadetblue', edgecolor='black');
axes[1].set_xlabel('Percent of Grammatical Mistakes');
axes[1].set_ylabel('Frequency');
axes[1].set_title('Percentage of Common Grammar Mistakes in ChatGPT Answers');


# ### B) Ratio of Different Parts-of-Speech

# In[6]:


import nltk
nltk.download('averaged_perceptron_tagger')

def pos_ratio(tokens):

    # List to store ratio of various pos
    noun = []
    verb = []
    adj = []
    adv = []
    pro = []
    prep = []
    det = []
    conj = []
    inter = []

    n = len(tokens)

    for i in range(n):
        # POS tagging
        tagged = nltk.pos_tag(tokens[i])
        l = len(tagged)
        # Nouns
        n = len([x[0] for x in tagged if x[1].startswith('NN')])
        noun.append(n/l)

        # Verbs
        v = len([x[0] for x in tagged if x[1].startswith('VB')])
        verb.append(v/l)

        # Adjectives
        adject = len([x[0] for x in tagged if x[1].startswith('JJ')])
        adj.append(adject/l)

        # Adverbs
        ad = len([x[0] for x in tagged if x[1].startswith('RB') or x[1]=='WRB'])
        adv.append(ad/l)

        # Pronouns
        p = len([x[0] for x in tagged if x[1].startswith('PRP') or x[1].startswith('WP')])
        pro.append(p/l)

        # Prepositions
        pre = len([x[0] for x in tagged if x[1]=='IN'])
        prep.append(pre/l)

        # Determiners
        d = len([x[0] for x in tagged if x[1].endswith('DT')])
        det.append(d/l)

        # Conjunctions
        c = len([x[0] for x in tagged if x[1]=='CC'])
        conj.append(c/l)

        # Interjections
        i = len([x[0] for x in tagged if x[1]=="UH"])
        inter.append(i/l)

    return noun, verb, adj, adv, prep, det, conj, inter


# In[7]:


noun_h, verb_h, adj_h, adv_h, prep_h, det_h, conj_h, inter_h = pos_ratio(df['Human Answer'])
noun_c, verb_c, adj_c, adv_c, prep_c, det_c, conj_c, inter_c = pos_ratio(df['ChatGPT Answer'])


# In[8]:


pos_df_h = pd.DataFrame({"Noun":noun_h, "Verb":verb_h, "Adjective":adj_h, "Adverb":adv_h,\
                         "Preposition":prep_h, "Determiner":det_h, "Conjunction":conj_h, "Interjection":inter_h})
pos_df_c = pd.DataFrame({"Noun":noun_c, "Verb":verb_c, "Adjective":adj_c, "Adverb":adv_c,\
                         "Preposition":prep_c, "Determiner":det_c, "Conjunction":conj_c, "Interjection":inter_c})


# In[9]:


pos_df_h.head()


# In[10]:


pos_df_c.head()


# In[11]:


column_means_h = pos_df_h.mean()
column_means_c = pos_df_c.mean()
column_means_h['Others'] = 1- sum(column_means_h)
column_means_c['Others'] = 1- sum(column_means_c)
columns = column_means_h.index
for col in columns:
    if column_means_h[col]>column_means_c[col]:
        print(f"{col} more likely to occur in human answers.")
    else:
        print(f"{col} more likely to occur in ChatGPT answers.")


# In[12]:


X_axis = np.arange(len(column_means_h))

plt.barh(X_axis - 0.2, column_means_h.values, 0.4, label = 'Human', color = 'lightcoral')
plt.barh(X_axis + 0.2, column_means_c.values, 0.4, label = 'ChatGPT', color = 'cadetblue')

plt.gca().invert_yaxis()
plt.yticks(X_axis, column_means_h.index)
plt.ylabel("Parts of Speech")
plt.xlabel("Ratio in the Answer")
plt.title("Ratio of Different Parts-of-Speech in Answers")
plt.legend()
plt.show()


# ## Preparing Data for Classification Model

# ### A) Concatenate Data (Human and ChatGPT Answers)

# In[13]:


# Creating a temporary dataframe by dropping the 'Human Answer' column from datafile
temporary_df1 = df.drop('Human Answer', axis=1)
# Renaming the columns of the temporary dataframe
temporary_df1.columns = ['Question','Answer','Category']
# Creating another temporary dataframe by dropping the 'ChatGPT Answer' column from datafile
temporary_df2 = df.drop('ChatGPT Answer', axis=1)
# Renaming the columns of the temporary dataframe
temporary_df2.columns = ['Question','Answer','Category']
# Combining the two temporary dataframes
classification_df = pd.concat([temporary_df1,temporary_df2],ignore_index=True)


# In[14]:


# Creating target column where 0 represents ChatGPT answer and 1 represents human answer
target = [0] * len(temporary_df1) + [1] * len(temporary_df2)
# Adding a new column 'Target' to the data
classification_df['Target'] = target
classification_df


# ### B) Concatenate Data for Neural Networks

# In[15]:


# Creating a temporary dataframe by dropping the 'Human Answer' column from datafile
temporary_df1_nn = data.drop('Human Answer', axis=1)
# Renaming the columns of the temporary dataframe
temporary_df1_nn.columns = ['Question','Answer','Category']
# Creating another temporary dataframe by dropping the 'ChatGPT Answer' column from datafile
temporary_df2_nn = data.drop('ChatGPT Answer', axis=1)
# Renaming the columns of the temporary dataframe
temporary_df2_nn.columns = ['Question','Answer','Category']
# Combining the two temporary dataframes
classification_df_nn = pd.concat([temporary_df1_nn,temporary_df2_nn],ignore_index=True)


# In[16]:


# Creating target column where 0 represents ChatGPT answer and 1 represents human answer
target_nn = [0] * len(temporary_df1_nn) + [1] * len(temporary_df2_nn)
# Adding a new column 'Target' to the data
classification_df_nn['Target'] = target_nn
classification_df_nn


# ### C) Test Train Split

# In[17]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_data, test_data = train_test_split(classification_df, test_size = 0.2, random_state = 42, shuffle = True)

# reset indices of train and test dataframes
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


# In[18]:


train_data.head()


# In[19]:


test_data.head()


# ### D) Feature Matrix

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

import scipy
from sklearn.metrics import precision_recall_fscore_support, \
classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


# #### Extra Features

# In[21]:


def get_features(tokens):
    # Get additional features
    gram_mist = gram_mistakes(tokens)
    noun, verb, adj, adv, prep, det, conj, inter = pos_ratio(tokens)
    ans_len = []
    for i in range(len(tokens)):
        ans_len.append(len(tokens[i]))

    # Features matrix
    features = np.column_stack((gram_mist, noun, verb, adj, adv, prep, det, conj, inter, ans_len))

    # Normalized features matrix
    mm = MinMaxScaler()
    features = mm.fit_transform(features)

    return features


# #### TF-IDF

# In[22]:


def compute_tfidf(train_df, test_df = None, stop_words = None, min_df = 1):

    # Initializing the TfidfVectorizer
    vect = TfidfVectorizer(stop_words = stop_words, min_df = min_df)   

    # Generating tf-idf matrix
    tfidf_vect = vect.fit_transform([' '.join(doc) for doc in train_df])

    # Get feature names
    feature_names = vect.get_feature_names_out()
    if test_df is not None:
        # Transforming test data by the fitted TfidfVectorizer
        test_tfidf = vect.transform([' '.join(doc) for doc in test_df])
        return tfidf_vect, test_tfidf, feature_names
    else:
        return tfidf_vect, feature_names


# #### Combined Features
# 

# In[23]:


def combined_features(train_df, test_df = None, stop_words = None, min_df = 1):

    # Obtaining other features
    train_features = get_features(train_df)    

    if test_df is not None:
        # Tf-idf matrix for train and test data
        tfidf_vect, test_tfidf, feature_names = compute_tfidf(train_df, test_df, stop_words = None, min_df = 1)
    else: 
        tfidf_vect, feature_names = compute_tfidf(train_df, stop_words = None, min_df = 1)

    # Combining features of train data
    train_feature_vect = scipy.sparse.hstack([tfidf_vect,train_features])
    feature_names = list(feature_names)
    feature_names = ['Word: ' + word for word in feature_names]
    feature_names.extend(['Grammar Mistakes', 'Noun Ratio', 'Verb Ratio', \
                          'Adjective Ratio', 'Adverb Ratio', 'Preposition Ratio', \
                          'Determiner Ratio', 'Conjunction Ratio', 'Interjection Ratio', 'Answer Length'])

    if test_df is not None:
        # Obtaining other features
        test_features = get_features(test_df)
        # Combining features of test data
        test_feature_vect = scipy.sparse.hstack([test_tfidf,test_features])
        return train_feature_vect, test_feature_vect, feature_names
    else:
        return train_feature_vect, feature_names


# ## Mutual Information Test to find Importance of Features

# In[24]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

def mutual_info_test(X_train, y_train, k_value, X_test = None, transform_test = False):
 # configure to select all features
    fs = SelectKBest(score_func = mutual_info_classif, k = k_value)
 # learn relationship from training data
    fs.fit(X_train, y_train)
 # transform train input data
    X_train_fs = fs.transform(X_train)
 # transform test input data
    X_test_fs = None
    if transform_test and X_test:
        X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs


# In[25]:


from sklearn.metrics import mutual_info_score
import pandas as pd

feature_vect, feature_names = combined_features(classification_df['Answer'], test_df=None, stop_words=None, min_df=1)



# In[26]:


 # Mutual information score for all the features
_,_,mi_scores = mutual_info_test(feature_vect, classification_df['Target'],False,'all')

# Create a pandas Series to sort the scores
sorted_scores = pd.Series(mi_scores.scores_, name='Scores', index=feature_names).sort_values(ascending=False)


# In[27]:


pyplot.barh(sorted_scores[:20].index, sorted_scores[:20].values, color = 'cadetblue')
pyplot.title("Top 20 Features - Mutual Information Score")
plt.gca().invert_yaxis()
pyplot.show()


# In[28]:


# Columns that contain lists
list_cols = ["Question", "Answer", "Category"]

# Convert list columns to JSON strings before saving
for col in list_cols:
    classification_df[col] = classification_df[col].apply(json.dumps)

# Storing the train test data in excel file
classification_df.to_excel("train_test_data.xlsx", index=False)


# In[29]:


# Storing the train test data for neural networks in excel file
classification_df_nn.to_excel("train_test_data_nn.xlsx", index=False)


# In[ ]:




