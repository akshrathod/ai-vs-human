#!/usr/bin/env python
# coding: utf-8

# ## 3. Data Pre-Processing
# 
# 

# #### Import statements

# In[ ]:


import pandas as pd
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import re
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# #### Inline print and go to the begining of line

# In[ ]:


def print_line(*args):
    args1 = [str(arg) for arg in args]
    str_ = ' '.join(args1)
    sys.stdout.write(str_ + '\r')
    sys.stdout.flush()


# #### Converting the text into tokens, where all types of spaces and punctuation are removed

# In[ ]:


def tokenize(text):
    tokens = [token.text for token in nlp(text) if not token.is_punct and not token.is_space]
    return tokens


# #### Adding column to store category of questions

# In[ ]:


def add_category(data, category):
    data["Category"] = category
    return data


# In[ ]:


def preprocess_data(data, remove_char_punc = True, lowercase = True, remove_stopword = True, remove_digits = True, lemmatization = True):
    if remove_char_punc:       
        # Remove Special Characters and Punctuation
        data = data.apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))

    if lowercase:
        # Lowercasing
        data = data.apply(lambda x: x.lower())

    # Tokenization
    data = data.apply(lambda x: nltk.word_tokenize(x))

    if remove_stopword:
        # Stopword Removal
        stop_words = set(stopwords.words('english'))
        data = data.apply(lambda x: [token for token in x if token not in stop_words])

    if remove_digits:
        # Digits Removal
        data = data.apply(lambda x: [token for token in x if not token.isdigit()])

    if lemmatization:
        # Lemmatization after POS Tagging
        lemmatizer = WordNetLemmatizer()
        # POS Tagging
        data = data.apply(lambda x: nltk.pos_tag(x))
        # Mapping Penn Treebank POS tags to WordNet POS tags
        def penn_to_wordnet(tag):
            if tag.startswith('JJ'):
                return wordnet.ADJ
            elif tag.startswith('VB'):
                return wordnet.VERB
            elif tag.startswith('NN'):
                return wordnet.NOUN
            elif tag.startswith('RB'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        data = data.apply(lambda x: [(word,penn_to_wordnet(tag)) for (word,tag) in x])
        # Lemmatization
        data = data.apply(lambda x: [lemmatizer.lemmatize(word,tag) for (word,tag) in x])

    return data


# In[ ]:


data = pd.read_excel('scraped_and_ai_data.xlsx')


# In[ ]:


# Preprocessing the data
df = data.copy()
df['Question'] = preprocess_data(df['Question'], remove_char_punc = True, lowercase = True, remove_stopword = True, remove_digits = False, lemmatization = True)
df['Human Answer'] = preprocess_data(df['Human Answer'], remove_char_punc = True, lowercase = True, remove_stopword = True, remove_digits = False, lemmatization = True)
df['ChatGPT Answer'] = preprocess_data(df['ChatGPT Answer'], remove_char_punc = True, lowercase = True, remove_stopword = True, remove_digits = False, lemmatization = True)


# In[ ]:


# Printing some examples
print('Question:')
for i in range(0,3):
    print('\n-Example', i)
    print('* Original: ', data['Question'].iloc[i])
    print('* Processed: ', df['Question'].iloc[i])

print('\n\nHuman Answer:')
for i in range(0,3):
    print('\n-Example', i)
    print('* Original: ', data['Human Answer'].iloc[i])
    print('* Processed: ', df['Human Answer'].iloc[i])

print('\n\nChatGPT Answer:')
for i in range(0,3):
    print('\n-Example', i)
    print('* Original: ', data['ChatGPT Answer'].iloc[i])
    print('* Processed: ', df['ChatGPT Answer'].iloc[i])


# In[ ]:


print(df.head())


# In[ ]:


# Storing the preprocessed data in excel file
df.to_excel("preprocessed_data.xlsx", index=False)

