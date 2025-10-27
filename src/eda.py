#!/usr/bin/env python
# coding: utf-8

# ## 4. Exploratory Data Analysis:

# #### Import statements

# In[ ]:


import json
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from wordcloud import WordCloud

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_excel('scraped_and_ai_data.xlsx')

# Load the preprocessed file
df = pd.read_excel('preprocessed_data.xlsx')
# Decode JSON strings back into Python lists
for col in ["Question", "Human Answer", "ChatGPT Answer"]:
    df[col] = df[col].apply(json.loads)


# ### A) Stop-Word Analysis

# #### Creating category-wise dictionary for stopword frequency

# In[ ]:


def stopword_cat(col, category):
    for i in range(len(category)):
        cat_count = data.Category.value_counts()
        for j in range(cat_count[i]):
            lc = cat_count[i]
            basic_tokens = col[(i*lc) + j].split()
            for k in basic_tokens:
                if k in stop_words and k in stopwords_cat[i]:
                    stopwords_cat[i][k] += 1
                elif k in stop_words and k not in stopwords_cat[i]:
                    stopwords_cat[i][k] = 1
                else:
                    continue
    return stopwords_cat


# #### Barh plot of top 20 frequently occured stopwords in each category

# In[ ]:


def plot_stopword_count(answer, category):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(len(category)):
        stopwords_cat = stopword_cat(answer, category)
        sorted_stopwords = sorted(stopwords_cat[i].items(), key=lambda x: x[1], reverse=True)[:20]
        axs[i//2, i%2].barh(range(len(sorted_stopwords)), [val[1] for val in sorted_stopwords], color = 'cadetblue')
        axs[i//2, i%2].set_yticks(range(len(sorted_stopwords)))
        axs[i//2, i%2].set_yticklabels([val[0] for val in sorted_stopwords])
        axs[i//2, i%2].set_title(category[i])

    plt.tight_layout()
    plt.show()


# In[ ]:


category = data['Category'].unique()

stopword_dict = []
stopwords_cat1, stopwords_cat2, stopwords_cat3, stopwords_cat4 = {}, {}, {}, {}
stopwords_cat = [stopwords_cat1, stopwords_cat2, stopwords_cat3, stopwords_cat4]

print("Frequency plot of top 20 stop words in Human answers:\n")
plot_stopword_count(data['Human Answer'], category)
print("\n\nFrequency plot of top 20 stop words in ChatGPT answers:\n")
plot_stopword_count(data['ChatGPT Answer'], category)


# ### B) Average Length of Answers

# In[ ]:


categories=['Movie/TV','Science/Tech','Philosophy','Indian Food']

Avg_len_human_answer = []
Avg_len_chatgpt_answer = []

for category in categories:
    category_df=df[df['Category'] == category].reset_index(drop=True)
    total_length_human_answer = 0
    total_length_chatgpt_answer = 0
    for i in range(len(category_df)):
        length_human_answer = len(category_df['Human Answer'][i])
        length_chatgpt_answer = len(category_df['ChatGPT Answer'][i])
        total_length_human_answer += length_human_answer
        total_length_chatgpt_answer += length_chatgpt_answer
    Avg_len_human_answer.append(total_length_human_answer/len(category_df))
    Avg_len_chatgpt_answer.append(total_length_chatgpt_answer/len(category_df))


# In[ ]:


X_axis = np.arange(len(categories))

plt.bar(X_axis - 0.2, Avg_len_human_answer, 0.4, label = 'Human', color = 'lightcoral')
plt.bar(X_axis + 0.2, Avg_len_chatgpt_answer, 0.4, label = 'ChatGPT', color = 'cadetblue')

plt.xticks(X_axis, categories)
plt.xlabel("Categories")
plt.ylabel("Average Answer Length")
plt.title("Average Answer Length in Each Category")
plt.legend()
plt.show()


# ### C) Lengths of Answers Histogram

# In[ ]:


human_ans_len = []
chatgpt_ans_len = []
for i in range(len(df)):
    human_ans_len.append(len(df['Human Answer'][i]))
    chatgpt_ans_len.append(len(df['ChatGPT Answer'][i]))


# In[ ]:


# Histogram of Human Answers Length
plt.hist(human_ans_len, bins = 100, facecolor = 'cadetblue', edgecolor='black', linewidth=0.5)
plt.xlabel('Length of Human Answers')
plt.ylabel('Frequency')
plt.title('Histogram of Human Answers Length')
plt.show()
print('\n\n')

# Histogram of ChatGPT Answers Length
plt.hist(chatgpt_ans_len, bins = 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
plt.xlabel('Length of ChatGPT Answers')
plt.ylabel('Frequency')
plt.title('Histogram of ChatGPT Answers Length')
plt.show()


# ### D) Ratio of questions AI Model cannot answer

# In[ ]:


# Set of words to search for
search_words = {'ai', 'language', 'model'}

threshold = 10
# Number of rows where the list contains the search words (represents generic answers)
generic_answer = len([i for i, row in df.iterrows() if search_words.issubset(row['ChatGPT Answer']) and len(row['ChatGPT Answer']) > threshold])

# Number of rows where the list contains the search words and is a short answer (represents questions that ChatGPT cannot answer)
cannot_answer = len([i for i, row in df.iterrows() if search_words.issubset(row['ChatGPT Answer']) and len(row['ChatGPT Answer']) < threshold])

# Number of answers that the ChatGPT is able to give an answer to
answer = len(df['ChatGPT Answer']) - generic_answer - cannot_answer


# In[ ]:


# Create a pie chart
plt.pie([generic_answer, cannot_answer, answer], labels = ['Generic Answer', 'Cannot Answer', 'Can Answer'], \
        autopct = '%1.1f%%', colors = ['thistle','lightcoral','lightgreen'], explode = [0.01, 0.01, 0.01])

plt.title('ChatGPT Generated Answers')
plt.show()


# In[ ]:


# Category of questions that generic ChatGPT answers
gen_answer_categories = [row['Category'] for i, row in df.iterrows() if search_words.issubset(row['ChatGPT Answer'])]
category_count = {}
for cat in set(df['Category']):
    category_count[cat] = gen_answer_categories.count(cat)

plt.pie(list(category_count.values()), labels =list(category_count.keys()), autopct='%1.1f%%', \
        colors = ['lightcoral', 'thistle', 'darkgray', 'lightblue'], explode = [0.01, 0.01, 0.01, 0.01])
plt.title('Generic ChatGPT Answers Categories')
plt.show()


# ### E) Word-Frequency Analysis

# In[ ]:


freq_dict_human = {}
freq_dict_chatgpt = {}

all_words = []
for tokens in df['Human Answer']:
    all_words += tokens

for word in all_words:
    if word in freq_dict_human:
        freq_dict_human[word] += 1
    else:
        freq_dict_human[word] = 1

all_words = []
for tokens in df['ChatGPT Answer']:
    all_words += tokens

for word in all_words:
    if word in freq_dict_chatgpt:
        freq_dict_chatgpt[word] += 1
    else:
        freq_dict_chatgpt[word] = 1


# In[ ]:


sorted_dict_human = {key: value for key, value in sorted(freq_dict_human.items(), key=lambda item: item[1], reverse=True)[:20]}
sorted_dict_chatgpt = {key: value for key, value in sorted(freq_dict_chatgpt.items(), key=lambda item: item[1], reverse=True)[:20]}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(sorted_dict_human.keys(), sorted_dict_human.values(), color = 'cadetblue')
axes[0].set_title("Word Frequency for Human Answer")
axes[0].set_xlabel("Words")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis='x', labelrotation=90)

axes[1].bar(sorted_dict_chatgpt.keys(), sorted_dict_chatgpt.values(), color = 'cadetblue')
axes[1].set_title("Word Frequency for ChatGPT Answer")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Frequency")
axes[1].tick_params(axis='x', labelrotation=90)

plt.show()


# ### F) Word Cloud

# In[ ]:


human_text = ''
chatGPT_text = ''

for ans in data['ChatGPT Answer']:
    chatGPT_text += ans

for ans in data['Human Answer']:
    human_text += ans

wordcloud_human = WordCloud(width=800, height=400, background_color="black", max_words=100, colormap="Blues").generate(human_text)
wordcloud_chatGPT = WordCloud(width=800, height=400, background_color="black", max_words=100, colormap="Blues").generate(chatGPT_text)
print('\nWord Cloud for Human Answers.\n')
plt.figure(figsize=(12,6))
plt.imshow(wordcloud_human, interpolation='bilinear')
plt.axis("off")
plt.show()

print('\n\nWord Cloud for ChatGPT Answers.\n')
plt.figure(figsize=(12,6))
plt.imshow(wordcloud_chatGPT, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### G) Sentiment Analysis

# In[ ]:


sia = SentimentIntensityAnalyzer()

sentiment_scores_human = []
sentiment_scores_human = data['Human Answer'].apply(lambda x: sia.polarity_scores(x))

sentiment_scores_chatgpt = []
sentiment_scores_chatgpt = data['ChatGPT Answer'].apply(lambda x: sia.polarity_scores(x))


# In[ ]:


compound_score_human = [score['compound'] for score in sentiment_scores_human]
pos_score_human = [score['pos'] for score in sentiment_scores_human]
neg_score_human = [score['neg'] for score in sentiment_scores_human]
neu_score_human = [score['neu'] for score in sentiment_scores_human]

compound_score_chatgpt = [score['compound'] for score in sentiment_scores_chatgpt]
pos_score_chatgpt = [score['pos'] for score in sentiment_scores_chatgpt]
neg_score_chatgpt = [score['neg'] for score in sentiment_scores_chatgpt]
neu_score_chatgpt = [score['neu'] for score in sentiment_scores_chatgpt]


fig, axes = plt.subplots(4, 2, figsize=(13, 20))

#compound_score_human
axes[0,0].hist(compound_score_human, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[0,0].set_xlabel('Values')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Distribution of Compound Score Human')

# compound_score_chatgpt
axes[0,1].hist(compound_score_chatgpt, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[0,1].set_xlabel('Values')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Distribution of Compound Score ChatGPT')

# neg_score_human
axes[1,0].hist(neg_score_human, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[1,0].set_xlabel('Values')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Distribution of Negative Score Human')

# neg_score_chatgpt
axes[1,1].hist(neg_score_chatgpt, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[1,1].set_xlabel('Values')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Negative Score ChatGPT')

# pos_score_human
axes[2,0].hist(pos_score_human, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[2,0].set_xlabel('Values')
axes[2,0].set_ylabel('Frequency')
axes[2,0].set_title('Distribution of Positive Score Human')

# pos_score_chatgpt
axes[2,1].hist(pos_score_chatgpt, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[2,1].set_xlabel('Values')
axes[2,1].set_ylabel('Frequency')
axes[2,1].set_title('Distribution of Positive Score ChatGPT')

#neu_score_human
axes[3,0].hist(neu_score_human, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[3,0].set_xlabel('Values')
axes[3,0].set_ylabel('Frequency')
axes[3,0].set_title('Distribution of Neutral Score Human')

# neu_score_chatgpt
axes[3,1].hist(neu_score_chatgpt, bins= 100, facecolor='cadetblue', edgecolor='black', linewidth=0.5)
axes[3,1].set_xlabel('Values')
axes[3,1].set_ylabel('Frequency')
axes[3,1].set_title('Distribution of Neutral Score ChatGPT')

plt.show()

