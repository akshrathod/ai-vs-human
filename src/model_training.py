#!/usr/bin/env python
# coding: utf-8

# ## 6. Model Training and Evaluation

# #### Import statements

# In[1]:


import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

import nltk
nltk.download('averaged_perceptron_tagger')

import scipy
from sklearn.metrics import precision_recall_fscore_support, \
classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


# In[2]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

from sentence_transformers import SentenceTransformer

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import regularizers


# In[3]:


data = pd.read_excel('scraped_and_ai_data.xlsx')

# Load the preprocessed file
df = pd.read_excel('preprocessed_data.xlsx')
# Decode JSON strings back into Python lists
for col in ["Question", "Human Answer", "ChatGPT Answer"]:
    df[col] = df[col].apply(json.loads)

# Load the train_test file
classification_df = pd.read_excel('train_test_data.xlsx')
# Decode JSON strings back into Python lists
for col in ["Question", "Answer", "Category"]:
    classification_df[col] = classification_df[col].apply(json.loads)

# Load the train_test file for neural networks
classification_df_nn = pd.read_excel('train_test_data_nn.xlsx')


# ### Test Train Split

# In[4]:


# Split the data into training and testing sets
train_data, test_data = train_test_split(classification_df, test_size = 0.2, random_state = 42, shuffle = True)

# reset indices of train and test dataframes
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


# ### Feature Matrix

# #### Grammar Mistakes

# In[5]:


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


# #### Ratio of Different Parts-of-Speech

# In[6]:


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


# #### Extra Features

# In[7]:


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

# In[8]:


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

# In[9]:


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


# ### A) Best Parameters

# In[10]:


def search_para(docs, y, model_type = 'svm'):

    if model_type == 'svm':
        model = svm.LinearSVC()
    elif model_type == 'nb':
        model = MultinomialNB()
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('model', model)])

    parameters = {'tfidf__stop_words': [None, 'english'], 'tfidf__min_df': [1,2,3,5]}
    metric =  "f1_macro"
    gridsearch_clf = GridSearchCV(pipeline, param_grid=parameters,scoring=metric, cv=5)
    gridsearch_clf.fit([' '.join(doc) for doc in docs], y)
    final_params = {}
    for param_name in gridsearch_clf.best_params_:
        print("{0}:\t{1}".format(param_name, gridsearch_clf.best_params_[param_name]))
        final_params[param_name] = gridsearch_clf.best_params_[param_name]
    print("best f1 score: {:.3f}".format(gridsearch_clf.best_score_))

    return final_params


# In[11]:


# Best parameters for SVM
svm_params = search_para(train_data['Answer'], train_data['Target'], model_type = 'svm')


# In[12]:


# Best parameters for Naive Bayes
nb_params = search_para(train_data['Answer'], train_data['Target'], model_type = 'nb')


# ### B) Model Implementation - SVM and Naive Bayes

# In[13]:


def create_model(train_df, train_y, test_df, test_y, model_type='svm', \
                 stop_words=None, min_df = 1, print_result = True):

    model, tfidf_vect, auc_score, prc_score = None, None, None, None

    # Feature matrix for train and test data
    train_feature_vect, test_feature_vect, feature_names = combined_features(train_df, test_df, stop_words = stop_words, min_df = min_df)

    # Train model
    if model_type=='svm':
        # Training an SVM model using the training data
        model = svm.LinearSVC().fit(train_feature_vect, train_y)
    elif model_type=='nb':
        # Training a Multinomial Naive Bayes model using the training data
        model = MultinomialNB().fit(train_feature_vect, train_y)

    # Predicting labels for the test data
    pred_labels = model.predict(test_feature_vect)

    # Indices of the data that were predicted incorrectly
    incorrect_indices = [i for i in range(len(test_y)) if test_y[i] != pred_labels[i]]

    # Classification report
    report = classification_report(test_y, pred_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_y, pred_labels)

    # Probabilities as predictions
    if model_type=='svm':
        pred_y = model.decision_function(test_feature_vect)
    elif model_type=='nb':
        pred_prob = model.predict_proba(test_feature_vect)
        pred_y = pred_prob[:,1]

    # Computing fpr/tpr by different thresholds (positive class 1)
    fpr, tpr, thresholds = roc_curve(test_y, pred_y, pos_label=1)
    # AUC Score
    auc_score = auc(fpr, tpr)

    # Computing precision/recall by different thresholds (positive class 1)
    precision, recall, thresholds = precision_recall_curve(test_y, pred_y, pos_label=1)
    # PRC Score
    prc_score = auc(recall, precision)

    if print_result:

        # Confusion matrix
        conf_mat_plot(conf_matrix)

        # Classification report
        print(f"\nClassification Report:\n{report}")

    return model, tfidf_vect, auc_score, prc_score, incorrect_indices, fpr, tpr, recall, precision


# In[14]:


def conf_mat_plot(cf_matrix):
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    cmap = sns.cubehelix_palette(as_cmap=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap)
    plt.show()


# In[15]:


# Train SVM with the best parameters
print('SVM Model\n\n')
svm_model, svm_tfidf_vect, svm_auc_score, svm_prc_score, svm_incorrect_indices, svm_fpr, svm_tpr, svm_re, svm_pre = \
                                                create_model(train_data['Answer'], train_data['Target'],\
                                                test_data['Answer'], test_data['Target'],\
                                                model_type = 'svm', stop_words = svm_params['tfidf__stop_words'],\
                                                min_df = svm_params['tfidf__min_df'], print_result = True)

# Train Naive Bayes with the best parameters
print('\n\nNaive Bayes Model\n\n')
nb_model, nb_tfidf_vect, nb_auc_score, nb_prc_score, nb_incorrect_indices, nb_fpr, nb_tpr, nb_re, nb_pre = \
                                                  create_model(train_data['Answer'], train_data['Target'],\
                                                  test_data['Answer'], test_data['Target'],\
                                                  model_type = 'nb', stop_words = nb_params['tfidf__stop_words'], \
                                                  min_df = nb_params['tfidf__min_df'], print_result = True)


# In[16]:


# Plot ROC curves of both models
plt.plot(svm_fpr, svm_tpr, color = 'cadetblue', label = 'SVM');
plt.plot(nb_fpr, nb_tpr, color = 'lightcoral', label = 'Naive Bayes');
plt.plot([0, 1], [0, 1], color = 'olive', lw = 2, linestyle = '--');
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('ROC Curve');
plt.legend();
plt.show();
print("\nAUC of SVM: {:.2%}".format(svm_auc_score), "\nAUC of Naive Bayes: {:.2%}".format(nb_auc_score))
print('\n\n')

# Plot PRC curves of both models
plt.figure();
plt.plot(svm_re, svm_pre, color = 'cadetblue', label = 'SVM');
plt.plot(nb_re, nb_pre, color = 'lightcoral', label = 'Naive Bayes');
plt.xlabel('Recall');
plt.ylabel('Precision');
plt.title('PRC Curve');
plt.legend();
plt.show();
print("\nPRC of SVM: {:.2%}".format(svm_prc_score), "\nPRC of Naive Bayes: {:.2%}".format(nb_prc_score))


# ### C) Model Implementation - Neural Networks

# In[17]:


multi_lang_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

eng_lang_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


# In[18]:


def text_embedding_func(model, data):
    embedded_text = []

    for text in data['Answer']:
        embeddings = model.encode(text)
        embedded_text.append(embeddings)

    return np.array(embedded_text)


# In[20]:


# Split the data into training and testing sets
train_data_nn, test_data_nn = train_test_split(classification_df_nn, test_size = 0.2, random_state = 42, shuffle = True)

# reset indices of train and test dataframes
train_data_nn = train_data_nn.reset_index(drop=True)
test_data_nn = test_data_nn.reset_index(drop=True)

# Embedding
train_eng_lang_embedding = text_embedding_func(eng_lang_model, train_data_nn) 
test_eng_lang_embedding = text_embedding_func(eng_lang_model, test_data_nn) 


# In[21]:


pd.DataFrame(train_eng_lang_embedding).to_csv('train_eng_lang_embedding', index=False)
pd.DataFrame(test_eng_lang_embedding).to_csv('test_eng_lang_embedding', index=False)


# In[22]:


def plot_graphs(history):

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

  # plot data on the first subplot
  ax1.plot(history.history['loss'])
  ax1.plot(history.history['val_loss'])
  ax1.set_title('Model Loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.legend(['Train', 'Validation'], loc='upper right')

  # plot data on the second subplot
  ax2.plot(history.history['accuracy'])
  ax2.plot(history.history['val_accuracy'])
  ax2.set_title('Model accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('accuracy')
  ax2.legend(['Train', 'Validation'], loc='upper right')

  # display the plot
  plt.show()


# In[23]:


model2 = Sequential([
    BatchNormalization(),
    Dense(512, activation='relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(8, activation='relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.0005)
model2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history2 = model2.fit(train_eng_lang_embedding, np.array(train_data_nn['Target']), batch_size=16, epochs=60, validation_split=0.2)


# In[24]:


plot_graphs(history2)


# In[25]:


model2.summary()


# In[26]:


# Prediction for test data
y_pred = model2.predict(test_eng_lang_embedding)


# In[27]:


y_pred = (y_pred > 0.5).astype(int)
conf_mat_plot(confusion_matrix(test_data_nn['Target'], y_pred))

report = classification_report(test_data_nn['Target'], y_pred)
print(report)


# In[ ]:





# In[ ]:




