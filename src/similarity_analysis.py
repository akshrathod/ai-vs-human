#!/usr/bin/env python
# coding: utf-8

# ## 7. Similarity Analysis

# #### Import statements

# In[1]:


import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# In[2]:


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


# #### TF-IDF

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
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


# #### Grammar Mistakes

# In[4]:


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


# ### A) Cosine Similarity

# In[5]:


from sklearn.metrics.pairwise import cosine_similarity

def assess_similarity(human_vect, chatgpt_vect):
    result = None

    sim = []

    for i in range(human_vect.shape[0]):
        doc1 = human_vect[i,:]
        doc2 = chatgpt_vect[i,:]
        sim.append(cosine_similarity(doc1.reshape(1, -1), doc2.reshape(1, -1))[0][0])

    return sim


# In[6]:


tf_idf_cos_sim, feature_names_cos_sim = compute_tfidf(classification_df['Answer'])
k_cos_sim = int(len(classification_df['Answer'])/2)
tf_idf_human_cos_sim = tf_idf_cos_sim[:k_cos_sim,:]
tf_idf_chatgpt_cos_sim = tf_idf_cos_sim[k_cos_sim:,:]
cos_sim = assess_similarity(tf_idf_human_cos_sim, tf_idf_chatgpt_cos_sim)


# ### B) BLEU Score

# In[7]:


from nltk.translate.bleu_score import sentence_bleu

def blue_score_func(answers):

    k = int(len(answers)/2)
    human_ans = answers[:k]
    chatgpt_ans = answers[k:]

    bleuScore = []

    for i in range(k):
        ans1 = human_ans.iloc[i].split()
        ans2 = chatgpt_ans.iloc[i].split()
        bleuScore.append(sentence_bleu(ans1, ans2, weights=(1, 0, 0, 0)))

    return bleuScore


# In[8]:


bleu_scores = blue_score_func(classification_df_nn['Answer'])


# In[9]:


similarity_df = data.copy()
similarity_df['Cosine Similarity'] = cos_sim 
similarity_df['BLEU Score'] = bleu_scores 
similarity_df


# ### C) Similarity Analysis

# In[10]:


# Average Similarity
similarity_df['Cosine Similarity'].mean()


# In[11]:


# Average similarity for each category
average_cat_sim = similarity_df.groupby('Category')['Cosine Similarity'].mean()
average_cat_sim


# In[12]:


# Maximum similarity for each category
max_cat_sim = similarity_df.groupby('Category')['Cosine Similarity'].max()
max_cat_sim


# In[13]:


# Minimum similarity for each category
min_cat_sim = similarity_df.groupby('Category')['Cosine Similarity'].min()
min_cat_sim


# In[14]:


threshold = similarity_df['Cosine Similarity'].mean()
# Similarities less than the threshold
less_than_threshold = similarity_df.groupby('Category')['Cosine Similarity'].apply(lambda x: (x < threshold).sum())
print(less_than_threshold)


# In[15]:


# Number of answers with 0 similarity
less_than_threshold = similarity_df.groupby('Category')['Cosine Similarity'].apply(lambda x: (x == 0).sum())
print(less_than_threshold)


# In[16]:


# Sorting DataFrame based on Cosine Similarity in descending order
similarity_df_sorted = similarity_df.sort_values(by = 'Cosine Similarity', ascending = False)
top_n = similarity_df_sorted[['Question', 'Human Answer', 'ChatGPT Answer', 'Category']]


# In[17]:


# Top 10 Questions and their category, with most similar answers 
for index, row in top_n.head(10).iterrows():
    print(f"Question: {row['Question']}\nCategory: {row['Category']}\nHuman Answer: {row['Human Answer']}\nChatGPT Answer: {row['ChatGPT Answer']}\n\n\n")


# In[18]:


# Top 10 Questions and their category, with least similar answers 
for index, row in top_n.tail(10).iterrows():
    print(f"Question: {row['Question']}\nCategory: {row['Category']}\nHuman Answer: {row['Human Answer']}\nChatGPT Answer: {row['ChatGPT Answer']}\n\n\n")


# ### D) Capabilities and Limitations of ChatGPT

# #### Category-wise Capability/Limitation of ChatGPT to Answer Similar to Human

# In[19]:


# Category with highest similarity questions
highest_sim = top_n.head(100)['Category'].value_counts()

# Create a pie chart
plt.pie(highest_sim.values, labels = highest_sim.index, autopct = '%1.1f%%', \
        colors = ['lightcoral', 'thistle', 'darkgray', 'lightblue'], explode = [0.01, 0.01, 0.01, 0.01])

plt.title('Distribution of Most Similar Answers')
plt.show()


# In[20]:


# Category with least similarity questions
highest_sim = top_n.tail(100)['Category'].value_counts()

# Create a pie chart
plt.pie(highest_sim.values, labels = highest_sim.index, autopct = '%1.1f%%', \
        colors = ['lightblue', 'darkgray', 'thistle', 'lightcoral'], explode = [0.01, 0.01, 0.01, 0.01])

plt.title('Distribution of Least Similar Answers')
plt.show()


# In[21]:


threshold = similarity_df['Cosine Similarity'].mean()
# Similarities less than the threshold
less_than_threshold = similarity_df.groupby('Category')['Cosine Similarity'].apply(lambda x: (x < threshold).sum())

# Create a pie chart
plt.pie(less_than_threshold.values, labels = less_than_threshold.index, autopct = '%1.1f%%', \
        colors = ['lightcoral', 'lightblue', 'thistle', 'darkgray'], explode = [0.01, 0.01, 0.01, 0.01])

plt.title('Distribution of Answers with Similarity less than Mean Similarity')
plt.show()


# In[22]:


# Similarities less than the threshold
zero_sim = similarity_df.groupby('Category')['Cosine Similarity'].apply(lambda x: (x == 0).sum())

# Create a pie chart
plt.pie(zero_sim.values, labels = zero_sim.index, autopct = '%1.1f%%', \
        colors = ['lightcoral', 'lightblue', 'thistle', 'darkgray'], explode = [0.01, 0.01, 0.01, 0.01])

plt.title('Distribution of Answers with Zero Similarity')
plt.show()


# #### Comparison of Lengths for the Most Similar and Least Similar Answers

# In[23]:


top100_indices = top_n.head(100).index
top100_df = data.iloc[top100_indices]
top100_df = top100_df.reset_index(drop = True)
top100_df.head()


# In[24]:


low100_indices = top_n.tail(100).index
low100_df = data.iloc[low100_indices]
low100_df = low100_df.reset_index(drop = True)
low100_df.head()


# In[25]:


top100_h_len = []
top100_c_len = []
for i in range(len(top100_df)):
    top100_h_len.append(len(set(top100_df['Human Answer'][i].split())))
    top100_c_len.append(len(set(top100_df['ChatGPT Answer'][i].split())))

X_axis = np.arange(len(top100_df))

plt.figure(figsize=(15,5))
plt.bar(X_axis - 0.2, top100_h_len, 0.4, label = 'Human', color = 'lightcoral')
plt.bar(X_axis + 0.2, top100_c_len, 0.4, label = 'ChatGPT', color = 'cadetblue')

plt.xticks(X_axis, fontsize = 6)
plt.xlabel("Answers")
plt.ylabel("Answer Length")
plt.title("Answer Length Comparison for Top 100 Similar Answers")
plt.axhline(y = 200, linestyle = '--', color = 'black', label = 'Answer Length > 200')
plt.legend()
plt.show()

print('\n\n')

low100_h_len = []
low100_c_len = []
for i in range(len(low100_df)):
    low100_h_len.append(len(set(low100_df['Human Answer'][i].split())))
    low100_c_len.append(len(set(low100_df['ChatGPT Answer'][i].split())))

X_axis = np.arange(len(low100_df))

plt.figure(figsize=(15,5))
plt.bar(X_axis - 0.2, low100_h_len, 0.4, label = 'Human', color = 'lightcoral')
plt.bar(X_axis + 0.2, low100_c_len, 0.4, label = 'ChatGPT', color = 'cadetblue')

plt.xticks(X_axis, fontsize = 6)
plt.xlabel("Answers")
plt.ylabel("Answer Length")
plt.title("Answer Length Comparison for Least 100 Similar Answers")
plt.axhline(y = 200, linestyle = '--', color = 'black', label = 'Answer Length > 200')
plt.legend()
plt.show()


# In[26]:


top100_ratio_list = []
for i in range(len(top100_h_len)):
    top100_ratio_list.append(max(top100_h_len[i], top100_c_len[i]) / min(top100_h_len[i], top100_c_len[i]))
labels = range(len(top100_df))

low100_ratio_list = []
for i in range(len(low100_h_len)):
    low100_ratio_list.append(max(low100_h_len[i], low100_c_len[i]) / min(low100_h_len[i], low100_c_len[i]))
labels = range(len(low100_df))

print(f'Average Ratio of Human Answer to ChatGPT answer for Most Similar Answers: {np.mean(top100_ratio_list)}\n')
print(f'Average Ratio of Human Answer to ChatGPT answer for Least Similar Answers: {np.mean(low100_ratio_list)}\n')


# #### Comparison of Sentiment for Most Similar and Least Similar Answers

# In[27]:


sia = SentimentIntensityAnalyzer()

top100_h_scores = []
top100_h_scores = top100_df['Human Answer'].apply(lambda x: sia.polarity_scores(x))

top100_c_scores = []
top100_c_scores = top100_df['ChatGPT Answer'].apply(lambda x: sia.polarity_scores(x))


low100_h_scores = []
low100_h_scores = low100_df['Human Answer'].apply(lambda x: sia.polarity_scores(x))

low100_c_scores = []
low100_c_scores = low100_df['ChatGPT Answer'].apply(lambda x: sia.polarity_scores(x))


# In[28]:


top100_comp_score_h = [score['compound'] for score in top100_h_scores]
top100_comp_score_c = [score['compound'] for score in top100_c_scores]

low100_comp_score_h = [score['compound'] for score in low100_h_scores]
low100_comp_score_c = [score['compound'] for score in low100_c_scores]

X_axis = np.arange(len(top100_comp_score_h))

plt.figure(figsize=(15,5))
plt.bar(X_axis - 0.2, top100_comp_score_h, 0.4, label = 'Human', color = 'lightcoral')
plt.bar(X_axis + 0.2, top100_comp_score_c, 0.4, label = 'ChatGPT', color = 'cadetblue')

plt.xticks(X_axis, fontsize = 6)
plt.xlabel("Answers")
plt.ylabel("Sentiment")
plt.title("Sentiment Comparison for Top 100 Similar Answers")
plt.legend()
plt.show()

print('\n\n')

X_axis = np.arange(len(low100_comp_score_h))

plt.figure(figsize=(15,5))
plt.bar(X_axis - 0.2, low100_comp_score_h, 0.4, label = 'Human', color = 'lightcoral')
plt.bar(X_axis + 0.2, low100_comp_score_c, 0.4, label = 'ChatGPT', color = 'cadetblue')

plt.xticks(X_axis, fontsize = 6)
plt.xlabel("Answers")
plt.ylabel("Sentiment")
plt.title("Sentiment Comparison for Least 100 Similar Answers")
plt.legend()
plt.show()


# In[29]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Compound Score Human for Top 100 Similar Answers
axes[0,0].hist(top100_comp_score_h, bins= 20, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[0,0].set_xlabel('Sentiment')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Distribution of Compound Score Human for Top 100 Similar Answers')

# Compound Score ChatGPT for Top 100 Similar Answers
axes[0,1].hist(top100_comp_score_c, bins= 20, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[0,1].set_xlabel('Sentiment')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Distribution of Compound Score ChatGPT for Top 100 Similar Answers')

# Compound Score Human for Least 100 Similar Answers
axes[1,0].hist(low100_comp_score_h, bins= 20, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[1,0].set_xlabel('Sentiment')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Distribution of Compound Score Human for Least 100 Similar Answers')

# Compound Score ChatGPT for Least 100 Similar Answers
axes[1,1].hist(low100_comp_score_c, bins= 20, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[1,1].set_xlabel('Sentiment')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Compound Score ChatGPT for Least 100 Similar Answers')


# #### Comparison of Grammar Mistakes of Most Similar and Least Similar Answers

# In[30]:


top100_gram_mist_h = gram_mistakes([ans.split() for ans in top100_df['Human Answer']])
top100_gram_mist_c = gram_mistakes([ans.split() for ans in top100_df['ChatGPT Answer']])

low100_gram_mist_h = gram_mistakes([ans.split() for ans in low100_df['Human Answer']])
low100_gram_mist_c = gram_mistakes([ans.split() for ans in low100_df['ChatGPT Answer']])

top100_diff_list = []
for i in range(len(top100_gram_mist_h)):
    top100_diff_list.append(abs(top100_gram_mist_h[i] - top100_gram_mist_c[i]))
labels = range(len(top100_df))

low100_diff_list = []
for i in range(len(low100_gram_mist_h)):
    low100_diff_list.append(abs(low100_gram_mist_h[i] - low100_gram_mist_c[i]))
labels = range(len(low100_df))


plt.figure(figsize=(15,5))
plt.bar(labels, top100_diff_list, color = 'cadetblue');
plt.title("Grammar Mistakes Difference for Top 100 Similar Answers");
plt.xlabel("Answers");
plt.ylabel("Grammar Mistakes Difference");
plt.xticks(labels, fontsize = 6);
plt.axhline(y = 2, linestyle = '--', color = 'lightcoral', label = 'Ratio of answers > 2');
plt.legend();
plt.show();
print('\n\n')

plt.figure(figsize=(15,5))
plt.bar(labels, low100_diff_list, color = 'cadetblue');
plt.title("Grammar Mistakes Difference for Least 100 Similar Answers");
plt.xlabel("Answers");
plt.ylabel("Grammar Mistakes Difference");
plt.xticks(labels, fontsize = 6);
plt.axhline(y = 2, linestyle = '--', color = 'lightcoral', label = 'Ratio of answers > 2');
plt.legend();
plt.show();


# In[31]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Grammar Mistakes Human
axes[0,0].hist(top100_gram_mist_h, bins= 20, facecolor='cadetblue', edgecolor='black');
axes[0,0].set_xlabel('Percent of Grammatical Mistakes');
axes[0,0].set_ylabel('Frequency');
axes[0,0].set_title('Common Grammar Mistakes in Human Answers (Top 100)');

# Grammar Mistakes ChatGPT
axes[0,1].hist(top100_gram_mist_c, bins= 20, facecolor='cadetblue', edgecolor='black');
axes[0,1].set_xlabel('Percent of Grammatical Mistakes');
axes[0,1].set_ylabel('Frequency');
axes[0,1].set_title('Common Grammar Mistakes in ChatGPT Answers (Top 100)');

# Grammar Mistakes Human
axes[1,0].hist(low100_gram_mist_h, bins= 20, facecolor='cadetblue', edgecolor='black');
axes[1,0].set_xlabel('Percent of Grammatical Mistakes');
axes[1,0].set_ylabel('Frequency');
axes[1,0].set_title('Common Grammar Mistakes in Human Answers (Least 100)');

# Grammar Mistakes ChatGPT
axes[1,1].hist(low100_gram_mist_c, bins= 20, facecolor='cadetblue', edgecolor='black');
axes[1,1].set_xlabel('Percent of Grammatical Mistakes');
axes[1,1].set_ylabel('Frequency');
axes[1,1].set_title('Common Grammar Mistakes in ChatGPT Answers (Least 100)');


# In[ ]:




