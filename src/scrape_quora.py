#!/usr/bin/env python
# coding: utf-8

# ## 1. Data Collection (Scraping Data From Quora)

# #### Import statements

# In[282]:


import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import spacy
import sys
import warnings

nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings('ignore')


# #### Saving data as an excel file

# In[ ]:


def store_excel(data, prev_data = None):
    if prev_data:
        # Loading old data into dataframe
        old_data = pd.read_excel(prev_data)
        # Concatenating the two dataframes vertically
        complete_data = pd.concat([old_data, data], ignore_index=True)
        # Storing the combined data to the excel file
        complete_data.to_excel('scraped_data.xlsx', index=False)
    else:
        # Storing the data to the excel file
        data.to_excel("scraped_data.xlsx", index=False)


# ### Scraping Data

# In[ ]:


def getData(page_urls, driver, min_ans_len = 15, limit = 10, scroll_num = 10, print_every = 10):

    # Empty dataframe to store the scraped content
    scraped_data = pd.DataFrame()

    # Initializing variable to track the number of data samples collected 
    len_data = 0

    # Initializing lists to store the scraped content
    questions = []
    answers = []

    # Count for webpages
    count = 1
    # Count for data being collected
    count1 = 0

    for page_url in page_urls:
        print(f"Page {count} of {len(page_urls)}")
        # Sending a get request to the web page (Navigating to the webpage)
        driver.get(page_url)
        # Wait
        driver.implicitly_wait(10)

        # Initializing variables to iterate through the try except block
        max_tries = 10
        retry = 0   

        # Initializing variable to check if we've reached the end of the page 
        old_content = None
        new_content = None

        # Scrolling to get enough answers
        for i in range(scroll_num):
            # Scrolling to access the next page of questions
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait for the content to be loaded        
            time.sleep(5) 
            # Checking if page is same before and after scrolling
            new_content = driver.page_source
            if new_content == old_content:
                break
            old_content = new_content

        # Used while loop to avoid "StaleElementReferenceException" error
        while retry<max_tries:
            try:
                # Scraping the question answer blocks on Quora
                data_elements = driver.find_elements(By.CSS_SELECTOR, "div.dom_annotate_multifeed_bundle_AnswersBundle")
                retry = 0
                break
            except:
                retry += 1


        # Used while loop to avoid "StaleElementReferenceException" error
        while retry<max_tries:
            try:
                for block in data_elements:

                    ### --- Questions --- ###
                    # Scraping question from the webpage 
                    ques = block.find_element(By.CSS_SELECTOR, "div.q-text.puppeteer_test_question_title span")

                    ### --- Answers --- ###
                    # Checking if "more" button is present for an answer
                    try: 
                        # Selecting the "more" button
                        read_more = block.find_element(By.CSS_SELECTOR, "div.q-absolute div.qt_read_more")
                        # Checking if the button is clickable
                        try:
                            # Expanding answer by clicking "more" button
                            read_more.click()
                        except:
                            # Discarding data where complete answer cannot be obtained
                            continue
                    except:
                        None
                    # Scraping answers from the webpage 
                    ans = block.find_element(By.CSS_SELECTOR, "div.q-box.spacing_log_answer_content.puppeteer_test_answer_content span.q-box")

                    if ques.text and ans.text:
                        # Skipping questions that are already present
                        if ques.text in questions:
                            continue
                        # Skipping the questions where length of answers are less than a given threshold
                        ans_tokens = len(tokenize(ans.text))
                        if ans_tokens<min_ans_len:
                            continue
                        # Appending the scraped question
                        questions.append(ques.text)
                        # Appending the scraped answer
                        answers.append(ans.text)
                        count1+=1
                        if count1 % print_every == 0 or count1 == limit:
                            print(f"{count1} of {limit}")
                        else:
                            print_line(f"{count1} of {limit}")
                    else:
                        continue

                    # Updating the number of data samples collected
                    len_data = len(questions) 
                    # Collecting data until limit is reached 
                    if len_data == limit:
                        break
                retry = 0
                break
            except:
                retry += 1 
        count+=1
        if len_data == limit:
            break

    # Warning to give more urls if desired amount of data is not scraped
    if len_data < limit:
        print("Warning: Need to provide more webpages to get desired amount of data!")

    # Storing the scraped information in a dataframe  
    scraped_data["Question"] = questions
    scraped_data["Answer"] = answers

    return scraped_data


# In[ ]:


# List of webpages to scrape data
page_urls = ["https://www.quora.com/topic/Philosophy"]

# Path to locate webdriver
executable_path = ".../chromedriver"
# Initializing webdriver
driver = webdriver.Chrome(executable_path=executable_path)
# Scraping data
data = getData(page_urls, driver, min_ans_len = 15, limit = 250, scroll_num = 20)
# Closing the webdriver
driver.quit()

# Category of data scraped
category = "Philosophy"
# Adding category column to the scraped data
data = add_category(data, category)
# Provide already existing file, if any, to which data must be appended
prev_data = None
# Storing scraped data in excel file
store_excel(data,prev_data)

