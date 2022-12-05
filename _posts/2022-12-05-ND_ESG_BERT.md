---
layout: single
title: "Text Analysis Code: whether a sentence is which ESG issues"
categories: [ESG, risk factors, Python, Text analysis]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
**[Notice]** [Journey to the academic researcher](https://ziofinlab.github.io/biography/bio/)
This is the story of how I became the insightful researcher.
{: .notice--info}

The text analysis code for isolating the 'risk factors' section from 10 years of 10-Ks.
This code comprises two sections. First, I Isolated 'risk factors' sections from 10 years of 10-Ks. The main idea of the isolation is finding patterns that represent where is risk factors section and utilizing the patterns in other remaining 10-Ks. Second, I categorized each sentence into ESG issues and counted the number of each category based on the BERT package established by professors Huang, Wang, and Yang.


```python
import re
import os    # import and read files
import os.path
import string # tokenization
import nltk # tokenization
from nltk.tokenize import MWETokenizer  #import tokenizer; making sentence to a list
tokenizer = MWETokenizer()
from nltk.corpus import stopwords  #import the list of stopwords
from nltk.stem.snowball import SnowballStemmer  #import stemmer module; finding root of words
stemmer = SnowballStemmer('english')
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()
from tqdm import tqdm
import csv
import multiprocessing
from multiprocessing import Pool
from time import time
import tweepy
import ssl
from tqdm import tqdm
ssl._create_default_https_context = ssl._create_unverified_context
import time
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
```


```python
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
```


```python
#define the following functions
def cmp(a, b):
    return (a > b) - (a < b)
```


```python
# Erasing the idcies that have one difference between other indicies
def list_cleansing(lst):
    for i in lst:
        for j in lst:
            if i - j == 1:
                lst.remove(j)
    return lst
```


```python
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\roman\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```python
column_names = ["file","total","e","s","g"]
esg_data_tot = pd.DataFrame(columns = column_names)
esg_data_90 = pd.DataFrame(columns = column_names)
esg_data_70 = pd.DataFrame(columns = column_names)

sent_names = ["file","label","score","contents"]
esg_sentence = pd.DataFrame(columns = sent_names)
```


```python
def error_print(basename, app_df, data_list, des):
    data_list = []
    data_list.append(basename)
    for i in range(len(app_df.columns)-1):
        data_list.append(des)
    app_df.loc[len(app_df)] = data_list
```


```python
path_base = 'C:/Users/roman/OneDrive/바탕 화면/Local_Codes/test'
filelist = os.listdir(path_base)

# get sub and sub-sub directories
path_sub = []
path_sub_sub = []
for i in filelist:
    path_sub.append(path_base + '/'+ i)
for i in path_sub:
    filelist_sub = os.listdir(i)
    for j in filelist_sub:
        path_sub_sub.append(i + '/' + j)

for op_path in path_sub_sub:
    # Reading files and calculating weights
    # setting the basement directory
#     op_path = 'C:/Users/roman/OneDrive/바탕 화면/Local_Codes/test'
    filelist = os.listdir(op_path)

    os.chdir(op_path)
    curDir = os.getcwd()

    print(op_path)

    for parent, dirnames, filenames in os.walk(curDir): # os.walk generates the file names in the directory
        # os.walk returns tuples that have three elements, paths, dir_names, and filenames
        for filename in tqdm(filenames):
            basename, extname = os.path.splitext(filename) # os.path.splitext can split (filename) to 'name' and 'extname'

            if((cmp(extname, '.txt') == 0) & ("10-K" in filename)): # cmp is comparison if the file is a txt file
                file_in = open(filename, encoding='utf-8', errors='ignore')
                docu = file_in.read()
                file_in.close()

                """Extracting indexes for isolating "Risk factors" from other items"""
                sentences = nltk.tokenize.sent_tokenize(docu)
                sentences_lower = [item.lower().strip() for item in sentences]

                start_w = ["\n risk factors \n", "risk factors\n", " risk factors\n","\nrisk factors", "\n risk factors" ,"risk factors\n","risk factors \n","risk factors.","r isk f actors"]
                # , "item 1a.","ITEM 1 A","item 1a","1a.", "1a", "1 a."
                end_w = ["item 1b.","item 1b","item 1 b.","item 1 b","IT EM 1B","1b.", "1 b.","it em 1b","unresolved staff comments","i tem 1b","u nresolved s taff c omments"]
                end_sub = ["2 properties", "2. properties", "\nproperties \n", "properties \n", "properties\n","p roperties","item 2","item 2."]
                # "item 2","item 2.", 
                s_idx = []
                e_idx = []
                des = ""

                try:
                    # setting start indexes (risk factors)
                    for start in start_w:
                        for idx, i in enumerate(sentences_lower):
                            if start in i:
                                s_idx.append(idx)

                    # setting end indexes (1b or item 2 right after risk factors)
                    for end in end_w:
                        for idx, i in enumerate(sentences_lower):
                            if end in i:
                                e_idx.append(idx)

                    # change end words list to set and then to list to erase duplicated vaules
                    e_idx_temp = set(e_idx)
                    e_idx_temp = list(e_idx_temp)
                    e_idx_temp.sort()

                    # if 1b item does not exist but is referred to other items
                    if (len(e_idx_temp) == 1) or (len(e_idx_temp) == 2) or (len(e_idx_temp) == 3):
                        if len(e_idx_temp) == 1:
                            if (e_idx_temp[0] <= 25) or (e_idx_temp[0] >= 1000):
                                e_idx = []
                        elif len(e_idx_temp) == 2:
                            if (e_idx_temp[0] <= 25) and (e_idx_temp[1] >= 1000):
                                e_idx = []
                            if (e_idx_temp[0] >= 1000) and (e_idx_temp[1] >= 1000):
                                e_idx = []
                            if (e_idx_temp[0] <= 25) and (e_idx_temp[1] <= 25):
                                e_idx = []
                        elif len(e_idx_temp) == 3:
                            if (e_idx_temp[1] <= 25) and (e_idx_temp[2] >= 1000):
                                e_idx = []
                            if (e_idx_temp[0] >= 1000) and (e_idx_temp[1] >= 1000) and (e_idx_temp[2] >= 1000):
                                e_idx = []

                    # if 1b item does not exist but item 2 exists
                    if len(e_idx) == 0:
                        for end_s in end_sub:
                            for idx, i in enumerate(sentences_lower):
                                if end_s in i:
                                    e_idx.append(idx)

                    if (len(s_idx) == 0) or (len(e_idx) == 0):
                        final_data, final_tot, final_70, final_90 = [],[],[],[]
                        des = "This 10-k does not have Risk Factors item"
                        error_print(basename, esg_sentence, final_data, des)
                        error_print(basename, esg_data_tot, final_tot, des)
                        error_print(basename, esg_data_70, final_70, des)
                        error_print(basename, esg_data_90, final_90, des)
                        final_data, final_tot, final_70, final_90 = [],[],[],[]
                        continue

                except:
                    final_data, final_tot, final_70, final_90 = [],[],[],[]
                    des = "This 10-k does not have Risk Factors item"
                    error_print(basename, esg_sentence, final_data, des)
                    error_print(basename, esg_data_tot, final_tot, des)
                    error_print(basename, esg_data_70, final_70, des)
                    error_print(basename, esg_data_90, final_90, des)
                    final_data, final_tot, final_70, final_90 = [],[],[],[]
                    continue


                """Cleansing indicies (s_idx and e_idx) to specify the risk factors section"""
                try:
                    # Exception procession
                    for i in e_idx:
                        if (i in s_idx) & (len(e_idx) == 1):
                            s_idx.remove(i)

                    # Erase duplicate indicies with converting the lists to the sets and then the sets to lists again
                    s_idx = set(s_idx)
                    e_idx = set(e_idx)
                    s_idx = list(s_idx)
                    e_idx = list(e_idx)
                    s_idx.sort()
                    e_idx.sort()

                    # Erase indicies that have one difference
                    list_cleansing(s_idx)
                    list_cleansing(e_idx)

                    fin_s = []
                    fin_e = []
                    temp_s = [x for x in s_idx]
                    temp_e = [x for x in e_idx]

                    if len(s_idx) >= len(e_idx):
                        for j in e_idx:
                            fin_e.append(j)
                            difference = lambda temp_s : abs(temp_s - j)
                            res = min(temp_s, key=difference, default=None)
                            fin_s.append(res)
                            if res == None:
                                continue
                            else:
                                while res > j:
                                    fin_s.remove(res)
                                    temp_s.remove(res)
                                    difference = lambda temp_s : abs(temp_s - j)
                                    res = min(temp_s, key=difference, default=None)
                                    fin_s.append(res)
                            temp_s = [x for x in s_idx]

                    else:
                        for i in s_idx:
                            fin_s.append(i)
                            difference = lambda temp_e : abs(temp_e - i)
                            res = min(temp_e, key=difference, default=None)
                            fin_e.append(res)
                            if res == None:
                                continue
                            else:
                                while res < i:
                                    fin_e.remove(res)
                                    temp_e.remove(res)
                                    difference = lambda temp_e : abs(temp_e - i)
                                    res = min(temp_e, key=difference, default=None)
                                    fin_e.append(res)
                            temp_e = [x for x in e_idx]

                except:
                    final_data, final_tot, final_70, final_90 = [],[],[],[]
                    des = "This 10-k can not cleanse the indicies"
                    error_print(basename, esg_sentence, final_data, des)
                    error_print(basename, esg_data_tot, final_tot, des)
                    error_print(basename, esg_data_70, final_70, des)
                    error_print(basename, esg_data_90, final_90, des)
                    final_data, final_tot, final_70, final_90 = [],[],[],[]
                    continue


                """Counting the number of sentences including natural disaster words"""
                try:
                    sen_count = 0
                    label = 0
                    score = 0
                    cont = 0

                    e_count = 0
                    s_count = 0
                    g_count = 0

                    e_conf = 0
                    s_conf = 0
                    g_conf = 0

                    e_90 = 0
                    s_90 = 0
                    g_90 = 0

                    e_70 = 0
                    s_70 = 0
                    g_70 = 0

                    for i in range(len(fin_s)):
                        for idx, j in enumerate(sentences_lower[fin_s[i]:fin_e[i]]):
                            results = nlp(j)
                            cont = j
                            sen_count += 1
                            label = results[0]['label']
                            score = results[0]['score']

                            if label == 'Environmental':
                                e_count += 1
                                e_conf = results[0]['score']
                                if e_conf >= 0.7:
                                    e_70 += 1
                                    if e_conf >= 0.9:
                                        e_90 += 1
                            if label == 'Social':
                                s_count += 1
                                s_conf = results[0]['score']
                                if s_conf >= 0.7:
                                    s_70 += 1
                                    if s_conf >= 0.9:
                                        s_90 += 1
                            if label == 'Governance':
                                g_count += 1
                                g_conf = results[0]['score']
                                if g_conf >= 0.7:
                                    g_70 += 1
                                    if g_conf >= 0.9:
                                        g_90 += 1

                            """stacking data to dataframe"""
                            final_data = []
                            final_data.append(basename)
                            final_data.append(label)
                            final_data.append(score)
                            final_data.append(cont)
                            esg_sentence.loc[len(esg_sentence)] = final_data

                    final_tot = []
                    final_tot.append(basename)
                    final_tot.append(sen_count)
                    final_tot.append(e_count)
                    final_tot.append(s_count)
                    final_tot.append(g_count)
                    esg_data_tot.loc[len(esg_data_tot)] = final_tot

                    final_70 = []
                    final_70.append(basename)
                    final_70.append(sen_count)
                    final_70.append(e_70)
                    final_70.append(s_70)
                    final_70.append(g_70)
                    esg_data_70.loc[len(esg_data_70)] = final_70

                    final_90 = []
                    final_90.append(basename)
                    final_90.append(sen_count)
                    final_90.append(e_90)
                    final_90.append(s_90)
                    final_90.append(g_90)
                    esg_data_90.loc[len(esg_data_90)] = final_90


                except:
                    final_data, final_tot, final_70, final_90 = [],[],[],[]
                    des = "This 10-k can not be categorized"
                    error_print(basename, esg_sentence, final_data, des)
                    error_print(basename, esg_data_tot, final_tot, des)
                    error_print(basename, esg_data_70, final_70, des)
                    error_print(basename, esg_data_90, final_90, des)
                    final_data, final_tot, final_70, final_90 = [],[],[],[]
                    continue

    esg_sentence.to_csv("esg_sentence.csv")
    esg_data_tot.to_csv("esg_cat_tot.csv")
    esg_data_70.to_csv("esg_cat_70.csv")
    esg_data_90.to_csv("esg_cat_90.csv")

    esg_data_tot = pd.DataFrame(columns = column_names)
    esg_data_90 = pd.DataFrame(columns = column_names)
    esg_data_70 = pd.DataFrame(columns = column_names)
    esg_sentence = pd.DataFrame(columns = sent_names)
```

    C:/Users/roman/OneDrive/바탕 화면/Local_Codes/test/2021/Q1
    C:/Users/roman/OneDrive/바탕 화면/Local_Codes/test/2021/Q1
    

    100%|██████████| 29/29 [00:48<00:00,  1.67s/it]
    

    C:/Users/roman/OneDrive/바탕 화면/Local_Codes/test/2021/Q2
    C:/Users/roman/OneDrive/바탕 화면/Local_Codes/test/2021/Q2
    

    100%|██████████| 25/25 [00:43<00:00,  1.76s/it]
    


```python

```
