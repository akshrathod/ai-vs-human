#!/usr/bin/env python
# coding: utf-8

# ## 2. Data Collection (Collect ChatGPT Answers)

# #### Import statements

# In[ ]:


import openai
import pandas as pd
openai.api_key = "Your_API_Key"


# In[ ]:


def get_ChatGPT_ans(questions):

    # Initializing list to store answers 
    answers = []

    # Send the questions to ChatGPT and store the responses
    for question in questions:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
        )
        answer = response['choices'][0]['message']['content']
        answers.append(answer)

    return answers


# In[ ]:


# Loading data
data = pd.read_excel("scraped_data.xlsx")
# Extracting questions column from data
questions = data["Question"].values
# Getting ChatGPT answers
chatgpt_ans = get_ChatGPT_ans(questions)
# Appending ChatGPT answers to data
data.insert(2, "ChatGPT Answer", chatgpt_ans)
# Storing data in excel file
data.to_excel("scraped_and_ai_data.xlsx", index=False)

