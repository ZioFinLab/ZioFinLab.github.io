---
layout: single
title: "Twitter tweets scrapping model (Fall 2022)"
categories: [Twitter, Python, Text analysis]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
**[Notice]** [Journey to the academic researcher](https://ziofinlab.github.io/biography/bio/)
This is the story of how I became the insightful researcher.
{: .notice--info}

This code can scrape the tweets' data from the Twitter server. Before implementing this code, users need to register their accounts to developer accounts and get various keys to free access to the server. Users should explain their purposes for using Twitter API to get the keys. If they sufficiently explain their goals or the goals are not suitable to utilize the API, Twitter will deny their registration. Users without authorization can not use all API functions and will be strictly restricted by Twitter.
Furthermore, even though users get full authorization, they only can request 900 data per 15mins to mitigate Twitter servers' overworks.


```python
import pandas as pd
import tweepy
import ssl
from tqdm import tqdm
ssl._create_default_https_context = ssl._create_unverified_context
import time
```


```python
consumer_key = "ENTER YOUR KEY"
consumer_secret = "ENTER YOUR KEY"
access_key = "ENTER YOUR KEY"
access_secret = "ENTER YOUR KEY"
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
client  = tweepy.Client(bearer_token = 'ENTER YOUR KEY', wait_on_rate_limit=True)
```

## Search_all_tweets version


```python
database = pd.read_excel('Twitter_Official_1009_update22.xlsx', sheet_name='base')
database
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>news_date</th>
      <th>End</th>
      <th>Start</th>
      <th>Official Account</th>
      <th>Add</th>
      <th>Unnamed: 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-02-27</td>
      <td>2021-06-22</td>
      <td>2017-08-27</td>
      <td>cvspharmacy</td>
      <td>2019-05-03</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-03</td>
      <td>2020-10-21</td>
      <td>2017-10-03</td>
      <td>dominos</td>
      <td>2019-05-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-10</td>
      <td>2021-06-17</td>
      <td>2017-10-10</td>
      <td>Nordstrom</td>
      <td>2018-03-29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-02-01</td>
      <td>2021-06-16</td>
      <td>2017-08-01</td>
      <td>kroger</td>
      <td>2019-03-24</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-02-23</td>
      <td>2021-05-10</td>
      <td>2017-08-23</td>
      <td>Kohls</td>
      <td>2018-02-13</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-02-01</td>
      <td>2021-06-09</td>
      <td>2017-08-01</td>
      <td>Lowes</td>
      <td>2019-10-23</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2019-10-14</td>
      <td>2021-03-12</td>
      <td>2019-04-14</td>
      <td>lululemon</td>
      <td>2019-09-28</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-02-12</td>
      <td>2021-06-30</td>
      <td>2017-08-12</td>
      <td>Starbucks</td>
      <td>2019-09-02</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-01-23</td>
      <td>2021-06-21</td>
      <td>2017-07-23</td>
      <td>ATT</td>
      <td>2019-06-25</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-02-05</td>
      <td>2021-06-08</td>
      <td>2017-08-05</td>
      <td>TDBank_US</td>
      <td>2018-07-13</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-01-24</td>
      <td>2021-06-30</td>
      <td>2017-07-24</td>
      <td>Target</td>
      <td>2017-11-09</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018-02-06</td>
      <td>2021-03-24</td>
      <td>2017-08-06</td>
      <td>Tmobile</td>
      <td>2018-09-09</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2018-02-13</td>
      <td>2020-10-21</td>
      <td>2017-08-13</td>
      <td>ultabeauty</td>
      <td>2017-09-30</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018-03-15</td>
      <td>2020-08-11</td>
      <td>2017-09-15</td>
      <td>Walmart</td>
      <td>2018-12-19</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
column_names = ["Account","gen_date", "text", "in_url", "in_media", "hash_text", "hash_count", "ret_count", "fav_count"]
tw_data = pd.DataFrame(columns = column_names)
tw_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>gen_date</th>
      <th>text</th>
      <th>in_url</th>
      <th>in_media</th>
      <th>hash_text</th>
      <th>hash_count</th>
      <th>ret_count</th>
      <th>fav_count</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
req_counter = 0
for idx, data in tqdm(database.iterrows()):
    tweet_ex = client.search_all_tweets(query="from:"+data['Official Account']+" lang:en -is:retweet -is:reply -is:quote -is:nullcast", \
                                          start_time=data['Start'], end_time=data['Add'], max_results=500)
    acc_name = data['Official Account']
    if tweet_ex[0] == None:
        continue
    for tweet in tweet_ex[0]:
        # check whether the number of requests arrived at the limit
        req_counter += 1
        if req_counter == 900:
            time.sleep(901)
            req_counter = 0
            
        target_id = tweet.id
        ex_stat = api.get_status(target_id)

        gen_date = ex_stat._json['created_at']
        cont = ex_stat._json['text']
        try:
            urls = ex_stat.entities['urls'][0]
            in_url = urls['url']
        except:
            in_url = "No URL included"
        try:
            media = ex_stat.entities['media'][0]
            in_media = media['type']
        except:
            in_media = "No Media included"
        hash_cont = []
        for hashtag in ex_stat.entities['hashtags']:
            hash_cont.append(hashtag['text'])
        hash_num = len(ex_stat.entities['hashtags'])
        ret_num = ex_stat._json['retweet_count']
        fav_num = ex_stat._json['favorite_count']
#         for i in client.search_all_tweets(query="in_reply_to_status_id: "+str(target_id), \
#                                           start_time=data['Start'], end_time=data['End'], max_results=200):
#             rep_counter += 1
#         rep_num = rep_counter

        rows = [acc_name, gen_date, cont, in_url, in_media, hash_cont, hash_num, ret_num, fav_num]
        tw_data.loc[len(tw_data)] = rows
```

    14it [04:59, 21.43s/it]
    


```python
tw_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>gen_date</th>
      <th>text</th>
      <th>in_url</th>
      <th>in_media</th>
      <th>hash_text</th>
      <th>hash_count</th>
      <th>ret_count</th>
      <th>fav_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cvspharmacy</td>
      <td>Tue Jun 25 13:30:32 +0000 2019</td>
      <td>Using @SpaRoomProducts' therapeutic 100% Pure ...</td>
      <td>https://t.co/Qlq5WI3pFD</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cvspharmacy</td>
      <td>Fri Jun 21 19:38:29 +0000 2019</td>
      <td>☀️ 🖍️  Kick off summer with our free coloring ...</td>
      <td>https://t.co/vdzONMLBRp</td>
      <td>photo</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cvspharmacy</td>
      <td>Wed Jun 19 13:34:45 +0000 2019</td>
      <td>Up to 50% of Americans don’t take their medica...</td>
      <td>https://t.co/2M2B3fK3yF</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>4</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cvspharmacy</td>
      <td>Tue Jun 18 15:30:52 +0000 2019</td>
      <td>Slide into the season without sneezing. We del...</td>
      <td>https://t.co/CszOREMaZb</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cvspharmacy</td>
      <td>Sun Jun 16 13:30:01 +0000 2019</td>
      <td>Thanks for everything you do, dads! Happy Fath...</td>
      <td>No URL included</td>
      <td>photo</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>Walmart</td>
      <td>Fri Dec 21 18:49:02 +0000 2018</td>
      <td>Not only is @MartinaMcBride a country supersta...</td>
      <td>https://t.co/Gh4XcaI0ow</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>6</td>
      <td>22</td>
    </tr>
    <tr>
      <th>475</th>
      <td>Walmart</td>
      <td>Fri Dec 21 17:52:32 +0000 2018</td>
      <td>If you know, you know. https://t.co/F94z95mCGT</td>
      <td>https://t.co/F94z95mCGT</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>12</td>
      <td>52</td>
    </tr>
    <tr>
      <th>476</th>
      <td>Walmart</td>
      <td>Thu Dec 20 21:02:11 +0000 2018</td>
      <td>In case Santa doesn’t get your letter, just se...</td>
      <td>https://t.co/aXOWOFVH72</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>11</td>
      <td>33</td>
    </tr>
    <tr>
      <th>477</th>
      <td>Walmart</td>
      <td>Thu Dec 20 20:07:34 +0000 2018</td>
      <td>Note to self: check for Cheeto-fingers before ...</td>
      <td>https://t.co/MvG0Sa9Y9m</td>
      <td>No Media included</td>
      <td>[WalmartTopSeller]</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
    </tr>
    <tr>
      <th>478</th>
      <td>Walmart</td>
      <td>Thu Dec 20 18:08:46 +0000 2018</td>
      <td>It’s not over ‘til it’s over. And our 20 Days ...</td>
      <td>https://t.co/JarvS9CPuP</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
<p>479 rows × 9 columns</p>
</div>




```python
tw_data.to_csv("FinalCheck_cycle22.csv")
```

## PAGINATION & Search all tweets


```python
database = pd.read_excel('Twitter_Official_1012_update25.xlsx', sheet_name='base')
database
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>news_date</th>
      <th>End</th>
      <th>Start</th>
      <th>Official Account</th>
      <th>Add</th>
      <th>Unnamed: 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-02-15</td>
      <td>2018-08-15</td>
      <td>2017-08-15</td>
      <td>AMTDGroup</td>
      <td>2018-01-25</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-06</td>
      <td>2021-04-07</td>
      <td>2017-10-06</td>
      <td>BestBuy</td>
      <td>2017-11-15</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-02-22</td>
      <td>2021-01-01</td>
      <td>2017-08-22</td>
      <td>Avis</td>
      <td>2020-05-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-02-06</td>
      <td>2021-06-07</td>
      <td>2017-08-06</td>
      <td>ChipotleTweets</td>
      <td>2021-03-24</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-20</td>
      <td>2019-07-15</td>
      <td>2017-10-20</td>
      <td>Designer_Brands</td>
      <td>2018-06-19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-08-29</td>
      <td>2021-05-18</td>
      <td>2018-02-28</td>
      <td>DollarGeneral</td>
      <td>2018-03-18</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-03-01</td>
      <td>2019-12-18</td>
      <td>2017-09-01</td>
      <td>darden</td>
      <td>2017-09-12</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-10-04</td>
      <td>2021-06-22</td>
      <td>2020-04-04</td>
      <td>darden</td>
      <td>2021-01-26</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-05-27</td>
      <td>2021-06-28</td>
      <td>2017-11-27</td>
      <td>Ford</td>
      <td>2021-03-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-09-05</td>
      <td>2019-07-15</td>
      <td>2018-03-05</td>
      <td>FastenalCompany</td>
      <td>2018-04-01</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2019-01-08</td>
      <td>2020-06-21</td>
      <td>2018-07-08</td>
      <td>footlocker</td>
      <td>2018-08-13</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018-07-06</td>
      <td>2019-12-13</td>
      <td>2018-01-06</td>
      <td>Genesco_Inc</td>
      <td>2019-04-30</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2018-01-25</td>
      <td>2020-10-02</td>
      <td>2017-07-25</td>
      <td>Hyatt</td>
      <td>2017-08-15</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018-10-18</td>
      <td>2019-04-18</td>
      <td>2018-04-18</td>
      <td>habitburger</td>
      <td>2019-04-18</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-04-25</td>
      <td>2019-07-15</td>
      <td>2017-10-25</td>
      <td>HDSupply</td>
      <td>2017-11-01</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2018-01-11</td>
      <td>2021-02-27</td>
      <td>2017-07-11</td>
      <td>HiltonHotels</td>
      <td>2021-02-27</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2020-04-27</td>
      <td>2021-02-03</td>
      <td>2019-10-27</td>
      <td>HRBlock</td>
      <td>2019-11-13</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-08-23</td>
      <td>2021-06-01</td>
      <td>2018-02-23</td>
      <td>HSBC</td>
      <td>2018-03-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018-07-17</td>
      <td>2021-01-02</td>
      <td>2018-01-17</td>
      <td>Labcorp</td>
      <td>2020-03-19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019-11-12</td>
      <td>2020-05-12</td>
      <td>2019-05-12</td>
      <td>ElPolloLoco</td>
      <td>2019-07-06</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018-01-18</td>
      <td>2021-06-10</td>
      <td>2017-07-18</td>
      <td>lukoilengl</td>
      <td>2020-03-03</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018-02-05</td>
      <td>2018-08-05</td>
      <td>2017-08-05</td>
      <td>lululemon</td>
      <td>2017-08-13</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2018-04-01</td>
      <td>2021-04-05</td>
      <td>2017-10-01</td>
      <td>Macys</td>
      <td>2020-08-31</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2018-03-09</td>
      <td>2021-06-04</td>
      <td>2017-09-09</td>
      <td>Marriott</td>
      <td>2017-10-23</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2018-01-17</td>
      <td>2019-07-15</td>
      <td>2017-07-17</td>
      <td>MurphyUSA</td>
      <td>2017-08-09</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2018-01-17</td>
      <td>2019-02-27</td>
      <td>2017-08-31</td>
      <td>MurphyUSA</td>
      <td>2019-02-27</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2020-08-10</td>
      <td>2021-03-09</td>
      <td>2020-02-10</td>
      <td>MurphyUSA</td>
      <td>2020-10-02</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018-03-15</td>
      <td>2021-06-29</td>
      <td>2017-09-15</td>
      <td>Nike</td>
      <td>2021-06-29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2019-04-18</td>
      <td>2019-10-18</td>
      <td>2018-10-18</td>
      <td>OlliesOutlet</td>
      <td>2018-12-19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2019-07-26</td>
      <td>2020-01-26</td>
      <td>2019-01-26</td>
      <td>BankOZK</td>
      <td>2019-02-18</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018-04-02</td>
      <td>2021-05-19</td>
      <td>2017-10-02</td>
      <td>PolarisInc</td>
      <td>2017-10-09</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2018-04-10</td>
      <td>2021-06-22</td>
      <td>2017-10-10</td>
      <td>childrensplace</td>
      <td>2018-09-23</td>
      <td>True</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2018-11-30</td>
      <td>2019-09-06</td>
      <td>2018-05-30</td>
      <td>PlanetFitness</td>
      <td>2019-03-31</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2018-02-01</td>
      <td>2021-06-07</td>
      <td>2017-08-01</td>
      <td>PNCBank</td>
      <td>2017-08-30</td>
      <td>True</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2018-01-26</td>
      <td>2021-05-06</td>
      <td>2017-07-26</td>
      <td>PVHCorp</td>
      <td>2018-05-18</td>
      <td>True</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2018-03-19</td>
      <td>2018-10-20</td>
      <td>2018-03-19</td>
      <td>SportsmansWH</td>
      <td>2018-10-20</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2019-05-08</td>
      <td>2020-12-10</td>
      <td>2018-11-08</td>
      <td>SportsmansWH</td>
      <td>2019-10-29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2018-03-07</td>
      <td>2020-05-04</td>
      <td>2017-09-07</td>
      <td>TruistNews</td>
      <td>2020-01-29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020-12-02</td>
      <td>2021-06-02</td>
      <td>2020-06-02</td>
      <td>DelTaco</td>
      <td>2020-06-24</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2018-03-22</td>
      <td>2019-07-15</td>
      <td>2017-09-22</td>
      <td>TractorSupply</td>
      <td>2018-04-29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2018-03-15</td>
      <td>2021-04-04</td>
      <td>2017-09-15</td>
      <td>Wendys</td>
      <td>2020-02-19</td>
      <td>True</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2018-01-11</td>
      <td>2021-02-14</td>
      <td>2017-07-11</td>
      <td>WolverineWW</td>
      <td>2017-09-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2019-05-08</td>
      <td>2020-12-10</td>
      <td>2018-11-08</td>
      <td>SportsmansWH</td>
      <td>2019-10-29</td>
      <td>True</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2018-01-25</td>
      <td>2021-06-22</td>
      <td>2017-07-25</td>
      <td>riteaid</td>
      <td>2018-03-21</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
column_names = ["Account","gen_date", "text", "in_url", "in_media", "hash_text", "hash_count", "ret_count", "fav_count"]
tw_data = pd.DataFrame(columns = column_names)
tw_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>gen_date</th>
      <th>text</th>
      <th>in_url</th>
      <th>in_media</th>
      <th>hash_text</th>
      <th>hash_count</th>
      <th>ret_count</th>
      <th>fav_count</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
req_counter = 0
for idx, data in tqdm(database.iterrows()):
    acc_name = data['Official Account']
    for tweet in tweepy.Paginator(client.search_all_tweets, query= "from: "+data['Official Account']+ " lang:en -is:retweet -is:reply -is:quote -is:nullcast",
                                start_time=data['Start'], end_time=data['Add'], max_results=500).flatten(limit=2000):
        req_counter += 1
        if tweet == None:
            continue

        if req_counter == 900:
            time.sleep(901)
            req_counter = 0
        
        target_id = tweet.id
        ex_stat = api.get_status(target_id)

        gen_date = ex_stat._json['created_at']
        cont = ex_stat._json['text']
        try:
            urls = ex_stat.entities['urls'][0]
            in_url = urls['url']
        except:
            in_url = "No URL included"
        try:
            media = ex_stat.entities['media'][0]
            in_media = media['type']
        except:
            in_media = "No Media included"
        hash_cont = []
        for hashtag in ex_stat.entities['hashtags']:
            hash_cont.append(hashtag['text'])
        hash_num = len(ex_stat.entities['hashtags'])
        ret_num = ex_stat._json['retweet_count']
        fav_num = ex_stat._json['favorite_count']

        rows = [acc_name, gen_date, cont, in_url, in_media, hash_cont, hash_num, ret_num, fav_num]
        tw_data.loc[len(tw_data)] = rows
```

    1it [00:00,  5.46it/s]Rate limit exceeded. Sleeping for 900 seconds.
    4it [36:07, 665.03s/it]Rate limit exceeded. Sleeping for 822 seconds.
    6it [49:50, 530.64s/it]Rate limit exceeded. Sleeping for 900 seconds.
    8it [1:04:51, 495.97s/it]Rate limit exceeded. Sleeping for 900 seconds.
    Rate limit exceeded. Sleeping for 897 seconds.
    9it [1:34:52, 803.19s/it]Rate limit exceeded. Sleeping for 899 seconds.
    10it [1:49:51, 827.37s/it]Rate limit exceeded. Sleeping for 901 seconds.
    11it [2:05:37, 858.57s/it]Rate limit exceeded. Sleeping for 856 seconds.
    13it [2:19:53, 673.35s/it]Rate limit exceeded. Sleeping for 900 seconds.
    14it [2:34:54, 726.88s/it]Rate limit exceeded. Sleeping for 900 seconds.
    15it [2:49:54, 770.48s/it]Rate limit exceeded. Sleeping for 901 seconds.
    16it [3:06:34, 830.94s/it]Rate limit exceeded. Sleeping for 802 seconds.
    18it [3:19:56, 646.05s/it]Rate limit exceeded. Sleeping for 900 seconds.
    19it [3:34:56, 705.78s/it]Rate limit exceeded. Sleeping for 901 seconds.
    20it [3:49:57, 754.91s/it]Rate limit exceeded. Sleeping for 901 seconds.
    24it [4:24:02, 524.36s/it]Rate limit exceeded. Sleeping for 705 seconds.
    25it [4:35:47, 576.91s/it]Rate limit exceeded. Sleeping for 901 seconds.
    26it [4:50:49, 672.14s/it]Rate limit exceeded. Sleeping for 900 seconds.
    28it [5:06:14, 576.69s/it]Rate limit exceeded. Sleeping for 876 seconds.
    30it [5:20:51, 522.05s/it]Rate limit exceeded. Sleeping for 900 seconds.
    32it [5:35:51, 496.16s/it]Rate limit exceeded. Sleeping for 901 seconds.
    33it [5:50:55, 579.58s/it]Rate limit exceeded. Sleeping for 898 seconds.
    35it [6:05:53, 531.46s/it]Rate limit exceeded. Sleeping for 901 seconds.
    36it [6:20:54, 608.61s/it]Rate limit exceeded. Sleeping for 901 seconds.
    37it [6:35:56, 675.80s/it]Rate limit exceeded. Sleeping for 900 seconds.
    38it [6:50:56, 731.23s/it]Rate limit exceeded. Sleeping for 901 seconds.
    39it [7:05:57, 775.58s/it]Rate limit exceeded. Sleeping for 901 seconds.
    40it [7:20:58, 809.71s/it]Rate limit exceeded. Sleeping for 901 seconds.
    41it [7:52:41, 1115.17s/it]Rate limit exceeded. Sleeping for 850 seconds.
    43it [8:06:52, 808.84s/it] Rate limit exceeded. Sleeping for 900 seconds.
    44it [8:22:14, 684.88s/it]
    


```python
tw_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>gen_date</th>
      <th>text</th>
      <th>in_url</th>
      <th>in_media</th>
      <th>hash_text</th>
      <th>hash_count</th>
      <th>ret_count</th>
      <th>fav_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BestBuy</td>
      <td>Tue Nov 14 18:59:07 +0000 2017</td>
      <td>Gear up.\n\nGet the #StarWarsBattlefrontII Eli...</td>
      <td>https://t.co/qaVjKvyqv6</td>
      <td>No Media included</td>
      <td>[StarWarsBattlefrontII]</td>
      <td>1</td>
      <td>30</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BestBuy</td>
      <td>Tue Nov 14 15:00:01 +0000 2017</td>
      <td>.@saradietschy proves that no matter how big o...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>21</td>
      <td>129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BestBuy</td>
      <td>Mon Nov 13 18:00:12 +0000 2017</td>
      <td>They’ll be dashing like Dasher and dancing lik...</td>
      <td>https://t.co/bpibw0wXFH</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>19</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BestBuy</td>
      <td>Mon Nov 13 15:00:09 +0000 2017</td>
      <td>Got a fav song on JAY-Z’s 4:44 album?\nTell us...</td>
      <td>https://t.co/AuwgDRsUjB</td>
      <td>No Media included</td>
      <td>[BestBuyTicketsNY, Sweepstakes]</td>
      <td>2</td>
      <td>27</td>
      <td>93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BestBuy</td>
      <td>Sun Nov 12 15:00:09 +0000 2017</td>
      <td>The AMD Ryzen Processor with Radeon Vega Graph...</td>
      <td>https://t.co/2gOQRxYv6h</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>26</td>
      <td>52</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2954</th>
      <td>riteaid</td>
      <td>Mon Aug 07 01:01:52 +0000 2017</td>
      <td>The DreamShip takes flight at the @3RiversRega...</td>
      <td>https://t.co/wuKidTA8vY</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2955</th>
      <td>riteaid</td>
      <td>Sun Aug 06 12:00:02 +0000 2017</td>
      <td>Why wait? Get your flu shot now through August...</td>
      <td>No URL included</td>
      <td>photo</td>
      <td>[]</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2956</th>
      <td>riteaid</td>
      <td>Fri Aug 04 13:30:08 +0000 2017</td>
      <td>The more points you earn with wellness+Plenti,...</td>
      <td>https://t.co/qQhIuoId6G</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2957</th>
      <td>riteaid</td>
      <td>Wed Aug 02 20:54:03 +0000 2017</td>
      <td>It's never too early to get your flu shot, so ...</td>
      <td>https://t.co/uvjriPn3iv</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2958</th>
      <td>riteaid</td>
      <td>Tue Jul 25 12:00:03 +0000 2017</td>
      <td>Make healthy eating fun with a picnic lunch of...</td>
      <td>No URL included</td>
      <td>photo</td>
      <td>[]</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>2959 rows × 9 columns</p>
</div>




```python
tw_data.to_csv("FinalCheck_cycle25.csv")
```

## PAGINATION SECTION for one account


```python
column_names = ["Account","gen_date", "text", "in_url", "in_media", "hash_text", "hash_count", "ret_count", "fav_count"]
tw_data = pd.DataFrame(columns = column_names)
tw_data
```


```python
start_time = "2017-07-25"+"T00:00:01Z"
end_time = "2018-03-21"+"T00:00:01Z"
acc_name = "riteaid"
```


```python
req_counter = 0

for tweet in tqdm(tweepy.Paginator(client.search_all_tweets, query= "from: "+acc_name+ " lang:en -is:retweet -is:reply -is:quote -is:nullcast",
                            start_time=start_time, end_time=end_time, max_results=500).flatten(limit=2000)):
    req_counter += 1
    if req_counter == 900:
        time.sleep(901)
        req_counter = 0
    target_id = tweet.id
    ex_stat = api.get_status(target_id)
    
    gen_date = ex_stat._json['created_at']
    cont = ex_stat._json['text']
    try:
        urls = ex_stat.entities['urls'][0]
        in_url = urls['url']
    except:
        in_url = "No URL included"
    try:
        media = ex_stat.entities['media'][0]
        in_media = media['type']
    except:
        in_media = "No Media included"
    hash_cont = []
    for hashtag in ex_stat.entities['hashtags']:
        hash_cont.append(hashtag['text'])
    hash_num = len(ex_stat.entities['hashtags'])
    ret_num = ex_stat._json['retweet_count']
    fav_num = ex_stat._json['favorite_count']
    
#     for i in tweepy.Paginator(client.search_all_tweets, query="in_reply_to_status_id: "+str(target_id),
#                             start_time=start_time, end_time=end_time, max_results=500).flatten(limit=100000):
#     for i in client.search_all_tweets(query="in_reply_to_status_id: "+str(target_id), start_time=start_time, end_time=end_time, max_results=200):
#         idx += 1
#     rep_num = idx
    
    rows = [acc_name, gen_date, cont, in_url, in_media, hash_cont, hash_num, ret_num, fav_num]
    tw_data.loc[len(tw_data)] = rows

```


```python
tw_data
```


```python
tw_data.to_csv("FinalCheck_riteaid.csv")
```


```python
column_names = ["Account","gen_date", "text", "in_url", "in_media", "hash_text", "hash_count", "ret_count", "fav_count"]
tw_data = pd.DataFrame(columns = column_names)
tw_data
```


```python
start_time = "2018-11-08"+"T00:00:01Z"
end_time = "2019-10-29"+"T00:00:01Z"
acc_name = "SportsmansWH"
```


```python
req_counter = 0

for tweet in tqdm(tweepy.Paginator(client.search_all_tweets, query= "from: "+acc_name+ " lang:en -is:retweet -is:reply -is:quote -is:nullcast",
                            start_time=start_time, end_time=end_time, max_results=500).flatten(limit=2000)):
    req_counter += 1
    if req_counter == 900:
        time.sleep(901)
        req_counter = 0
    target_id = tweet.id
    ex_stat = api.get_status(target_id)
    
    gen_date = ex_stat._json['created_at']
    cont = ex_stat._json['text']
    try:
        urls = ex_stat.entities['urls'][0]
        in_url = urls['url']
    except:
        in_url = "No URL included"
    try:
        media = ex_stat.entities['media'][0]
        in_media = media['type']
    except:
        in_media = "No Media included"
    hash_cont = []
    for hashtag in ex_stat.entities['hashtags']:
        hash_cont.append(hashtag['text'])
    hash_num = len(ex_stat.entities['hashtags'])
    ret_num = ex_stat._json['retweet_count']
    fav_num = ex_stat._json['favorite_count']
    
#     for i in tweepy.Paginator(client.search_all_tweets, query="in_reply_to_status_id: "+str(target_id),
#                             start_time=start_time, end_time=end_time, max_results=500).flatten(limit=100000):
#     for i in client.search_all_tweets(query="in_reply_to_status_id: "+str(target_id), start_time=start_time, end_time=end_time, max_results=200):
#         idx += 1
#     rep_num = idx
    
    rows = [acc_name, gen_date, cont, in_url, in_media, hash_cont, hash_num, ret_num, fav_num]
    tw_data.loc[len(tw_data)] = rows

```


```python
tw_data
```


```python
tw_data.to_csv("FinalCheck_sportsmansWH.csv")
```


```python
column_names = ["Account","gen_date", "text", "in_url", "in_media", "hash_text", "hash_count", "ret_count", "fav_count"]
tw_data = pd.DataFrame(columns = column_names)
tw_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>gen_date</th>
      <th>text</th>
      <th>in_url</th>
      <th>in_media</th>
      <th>hash_text</th>
      <th>hash_count</th>
      <th>ret_count</th>
      <th>fav_count</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
start_time = "2019-04-14"+"T00:00:01Z"
end_time = "2019-09-28"+"T00:00:01Z"
acc_name = "lululemon"
```


```python
req_counter = 0

for tweet in tqdm(tweepy.Paginator(client.search_all_tweets, query= "from: "+acc_name+ " lang:en -is:retweet -is:reply -is:quote -is:nullcast",
                            start_time=start_time, end_time=end_time, max_results=500).flatten(limit=2000)):
    req_counter += 1
    if req_counter == 900:
        time.sleep(901)
        req_counter = 0
    target_id = tweet.id
    ex_stat = api.get_status(target_id)
    
    gen_date = ex_stat._json['created_at']
    cont = ex_stat._json['text']
    try:
        urls = ex_stat.entities['urls'][0]
        in_url = urls['url']
    except:
        in_url = "No URL included"
    try:
        media = ex_stat.entities['media'][0]
        in_media = media['type']
    except:
        in_media = "No Media included"
    hash_cont = ex_stat.entities['hashtags']
    hash_num = len(ex_stat.entities['hashtags'])
    ret_num = ex_stat._json['retweet_count']
    fav_num = ex_stat._json['favorite_count']
    
#     for i in tweepy.Paginator(client.search_all_tweets, query="in_reply_to_status_id: "+str(target_id),
#                             start_time=start_time, end_time=end_time, max_results=500).flatten(limit=100000):
#     for i in client.search_all_tweets(query="in_reply_to_status_id: "+str(target_id), start_time=start_time, end_time=end_time, max_results=200):
#         idx += 1
#     rep_num = idx
    
    rows = [acc_name, gen_date, cont, in_url, in_media, hash_cont, hash_num, ret_num, fav_num]
    tw_data.loc[len(tw_data)] = rows

```

    51it [00:36,  1.40it/s]
    


```python
tw_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account</th>
      <th>gen_date</th>
      <th>text</th>
      <th>in_url</th>
      <th>in_media</th>
      <th>hash_text</th>
      <th>hash_count</th>
      <th>ret_count</th>
      <th>fav_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lululemon</td>
      <td>Wed Aug 28 02:30:40 +0000 2019</td>
      <td>Vetted by @lululemonmen , our best men's worko...</td>
      <td>https://t.co/QZ8sqhYMXr</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>46</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lululemon</td>
      <td>Mon Aug 26 23:00:11 +0000 2019</td>
      <td>We’re going beyond the buzzwords and giving yo...</td>
      <td>https://t.co/dloPhlz2EY</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lululemon</td>
      <td>Sat Aug 17 04:10:55 +0000 2019</td>
      <td>Has anyone seen @craig_mcmorris fanny pack? ht...</td>
      <td>https://t.co/zeyycJs9ni</td>
      <td>photo</td>
      <td>[{'text': 'SeaWheeze', 'indices': [68, 78]}]</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lululemon</td>
      <td>Fri Aug 16 15:55:00 +0000 2019</td>
      <td>There are 10,000 people running #SeaWheeze. Bu...</td>
      <td>https://t.co/saxcECmR2L</td>
      <td>No Media included</td>
      <td>[{'text': 'SeaWheeze', 'indices': [32, 42]}]</td>
      <td>1</td>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lululemon</td>
      <td>Tue Aug 06 23:30:42 +0000 2019</td>
      <td>Her relationship with her boobs is complicated...</td>
      <td>https://t.co/LO3UDF7F9v</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>7</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>lululemon</td>
      <td>Tue Aug 06 19:07:31 +0000 2019</td>
      <td>Where there's boobs, there's truths...and duet...</td>
      <td>https://t.co/xzFESvHa57</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lululemon</td>
      <td>Wed Jul 31 00:08:23 +0000 2019</td>
      <td>Introducing Boob Truth Tuesdays—You’ll laugh, ...</td>
      <td>https://t.co/dcCXqjTmrm</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>6</td>
      <td>31</td>
    </tr>
    <tr>
      <th>7</th>
      <td>lululemon</td>
      <td>Tue Jul 23 23:00:02 +0000 2019</td>
      <td>Better, together—our full collection of Men's ...</td>
      <td>https://t.co/8JKFYfaeyS</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>4</td>
      <td>47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lululemon</td>
      <td>Thu Jul 18 00:00:20 +0000 2019</td>
      <td>Sign-up to be the first to know about the new ...</td>
      <td>https://t.co/9unfTAajwb</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>45</td>
    </tr>
    <tr>
      <th>9</th>
      <td>lululemon</td>
      <td>Sun Jul 14 00:10:01 +0000 2019</td>
      <td>Need healing? A confidence-boost? Rest? 3 reas...</td>
      <td>https://t.co/lkggudeIFi</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>10</td>
      <td>40</td>
    </tr>
    <tr>
      <th>10</th>
      <td>lululemon</td>
      <td>Sat Jul 06 22:37:57 +0000 2019</td>
      <td>Professional quarterback @NickFoles’ secret to...</td>
      <td>https://t.co/j6mPl1Wnqn</td>
      <td>No Media included</td>
      <td>[{'text': 'lululemon', 'indices': [93, 103]}]</td>
      <td>1</td>
      <td>32</td>
      <td>295</td>
    </tr>
    <tr>
      <th>11</th>
      <td>lululemon</td>
      <td>Fri Jul 05 19:02:33 +0000 2019</td>
      <td>More time for yoga–ICYMI, Elite Ambassador and...</td>
      <td>https://t.co/LlaKEFYaeB</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>39</td>
    </tr>
    <tr>
      <th>12</th>
      <td>lululemon</td>
      <td>Sun Jun 30 20:43:48 +0000 2019</td>
      <td>In honour of 50 years of #pride we’ve asked so...</td>
      <td>https://t.co/sIUMuGrBYb</td>
      <td>No Media included</td>
      <td>[{'text': 'pride', 'indices': [25, 31]}]</td>
      <td>1</td>
      <td>6</td>
      <td>70</td>
    </tr>
    <tr>
      <th>13</th>
      <td>lululemon</td>
      <td>Sat Jun 29 22:59:01 +0000 2019</td>
      <td>In any profession, work stress is real. Here a...</td>
      <td>https://t.co/Oyutp9qM0i</td>
      <td>No Media included</td>
      <td>[{'text': 'Chicago', 'indices': [56, 64]}]</td>
      <td>1</td>
      <td>12</td>
      <td>87</td>
    </tr>
    <tr>
      <th>14</th>
      <td>lululemon</td>
      <td>Sat Jun 22 02:00:00 +0000 2019</td>
      <td>“Yoga allows me to enjoy the present moment.” ...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>4</td>
      <td>36</td>
    </tr>
    <tr>
      <th>15</th>
      <td>lululemon</td>
      <td>Sat Jun 22 01:00:12 +0000 2019</td>
      <td>Peace Coleman from I Grow Chicago shares what ...</td>
      <td>https://t.co/nfLQG2spHP</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>lululemon</td>
      <td>Sat Jun 22 01:00:00 +0000 2019</td>
      <td>“Yoga is an intimate date with myself.” - Elit...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>9</td>
      <td>52</td>
    </tr>
    <tr>
      <th>17</th>
      <td>lululemon</td>
      <td>Sat Jun 22 00:30:00 +0000 2019</td>
      <td>“Yoga has taught me to make peace with the unk...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>5</td>
      <td>46</td>
    </tr>
    <tr>
      <th>18</th>
      <td>lululemon</td>
      <td>Sat Jun 22 00:00:00 +0000 2019</td>
      <td>"Yoga is a daily dose of energy, strength, goo...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>2</td>
      <td>32</td>
    </tr>
    <tr>
      <th>19</th>
      <td>lululemon</td>
      <td>Fri Jun 21 23:30:00 +0000 2019</td>
      <td>“Yoga puts me completely in control of my mood...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>20</th>
      <td>lululemon</td>
      <td>Fri Jun 21 23:00:12 +0000 2019</td>
      <td>Adria Moses from @DETBoxingGym took her trauma...</td>
      <td>https://t.co/jFcL3YEzEX</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>21</th>
      <td>lululemon</td>
      <td>Fri Jun 21 23:00:00 +0000 2019</td>
      <td>“Yoga helps me to be mindful.” - Elite Ambassa...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>22</th>
      <td>lululemon</td>
      <td>Fri Jun 21 22:30:00 +0000 2019</td>
      <td>“Yoga turned me from an inflexible jock into a...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>23</th>
      <td>lululemon</td>
      <td>Fri Jun 21 22:00:00 +0000 2019</td>
      <td>“Yoga helps create more space in my mind and b...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>3</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>lululemon</td>
      <td>Fri Jun 21 21:30:00 +0000 2019</td>
      <td>"Yoga clears my head and heals my body.” - Eli...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>2</td>
      <td>28</td>
    </tr>
    <tr>
      <th>25</th>
      <td>lululemon</td>
      <td>Fri Jun 21 21:00:00 +0000 2019</td>
      <td>“Yoga is the tool I use to delete my back pain...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>26</th>
      <td>lululemon</td>
      <td>Fri Jun 21 20:30:00 +0000 2019</td>
      <td>“Yoga exposed me completely.” - Elite Ambassad...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>27</th>
      <td>lululemon</td>
      <td>Fri Jun 21 20:07:14 +0000 2019</td>
      <td>See what helped @AlexMazerolle discover her pa...</td>
      <td>https://t.co/0Zo79KXgZO</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>28</th>
      <td>lululemon</td>
      <td>Fri Jun 21 20:00:00 +0000 2019</td>
      <td>“Yoga has extended my career.” - Elite Ambassa...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>3</td>
      <td>15</td>
    </tr>
    <tr>
      <th>29</th>
      <td>lululemon</td>
      <td>Fri Jun 21 19:00:00 +0000 2019</td>
      <td>“Yoga has taught me to embrace the moment.” - ...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>2</td>
      <td>15</td>
    </tr>
    <tr>
      <th>30</th>
      <td>lululemon</td>
      <td>Fri Jun 21 18:30:00 +0000 2019</td>
      <td>“Yoga makes me the most authentic and courageo...</td>
      <td>https://t.co/QSSYlBvucy</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
    </tr>
    <tr>
      <th>31</th>
      <td>lululemon</td>
      <td>Fri Jun 21 18:00:00 +0000 2019</td>
      <td>“Yoga helped me win medals at the highest leve...</td>
      <td>https://t.co/XbVZx7Tduj</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>4</td>
      <td>31</td>
    </tr>
    <tr>
      <th>32</th>
      <td>lululemon</td>
      <td>Fri Jun 21 17:30:00 +0000 2019</td>
      <td>“Yoga is my main method of recovery.” - Elite ...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>33</th>
      <td>lululemon</td>
      <td>Fri Jun 21 17:00:01 +0000 2019</td>
      <td>“Yoga has taught me to breathe through my chal...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>6</td>
      <td>18</td>
    </tr>
    <tr>
      <th>34</th>
      <td>lululemon</td>
      <td>Fri Jun 21 16:30:00 +0000 2019</td>
      <td>It's so much more than just poses. How has yog...</td>
      <td>No URL included</td>
      <td>No Media included</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>9</td>
      <td>33</td>
    </tr>
    <tr>
      <th>35</th>
      <td>lululemon</td>
      <td>Fri Jun 21 16:00:28 +0000 2019</td>
      <td>It’s more powerful than you think: https://t.c...</td>
      <td>https://t.co/HHsX6D1roo</td>
      <td>photo</td>
      <td>[{'text': 'internationalyogaday', 'indices': [...</td>
      <td>2</td>
      <td>5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>36</th>
      <td>lululemon</td>
      <td>Tue Jun 18 23:00:12 +0000 2019</td>
      <td>Gahh, we’re so excited! Made with good ingredi...</td>
      <td>https://t.co/rZhixCe3eg</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>21</td>
      <td>176</td>
    </tr>
    <tr>
      <th>37</th>
      <td>lululemon</td>
      <td>Tue Jun 18 16:52:12 +0000 2019</td>
      <td>Eau de burpees, sweaty hair, hot yoga face...i...</td>
      <td>https://t.co/LoLIne0jN1</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>1</td>
      <td>61</td>
    </tr>
    <tr>
      <th>38</th>
      <td>lululemon</td>
      <td>Wed Jun 05 21:13:00 +0000 2019</td>
      <td>323,577 collective kilometers down, how may mo...</td>
      <td>https://t.co/mDHcER2NFa</td>
      <td>No Media included</td>
      <td>[{'text': 'GlobalRunningDay', 'indices': [65, ...</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
    </tr>
    <tr>
      <th>39</th>
      <td>lululemon</td>
      <td>Thu May 30 23:00:06 +0000 2019</td>
      <td>Grab your run crew, hit the pavement and crush...</td>
      <td>https://t.co/i7xaAwpvpB</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>7</td>
      <td>38</td>
    </tr>
    <tr>
      <th>40</th>
      <td>lululemon</td>
      <td>Fri May 24 16:01:01 +0000 2019</td>
      <td>On #GlobalRunningDay, let’s show the world the...</td>
      <td>https://t.co/cuEbAnMZxP</td>
      <td>No Media included</td>
      <td>[{'text': 'GlobalRunningDay', 'indices': [3, 2...</td>
      <td>1</td>
      <td>8</td>
      <td>46</td>
    </tr>
    <tr>
      <th>41</th>
      <td>lululemon</td>
      <td>Tue May 21 22:18:01 +0000 2019</td>
      <td>Vancouver's own, @robbiedxc is our newest Glob...</td>
      <td>https://t.co/xE6zPUg2Jh</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>5</td>
      <td>64</td>
    </tr>
    <tr>
      <th>42</th>
      <td>lululemon</td>
      <td>Sun May 12 12:51:05 +0000 2019</td>
      <td>Happy #MothersDay We’re celebrating our global...</td>
      <td>https://t.co/4XxxgbNmCV</td>
      <td>No Media included</td>
      <td>[{'text': 'MothersDay', 'indices': [6, 17]}]</td>
      <td>1</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>43</th>
      <td>lululemon</td>
      <td>Sat May 11 18:30:20 +0000 2019</td>
      <td>Recover from your run to keep moving and feeli...</td>
      <td>https://t.co/GHNABwSQ3i</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>2</td>
      <td>35</td>
    </tr>
    <tr>
      <th>44</th>
      <td>lululemon</td>
      <td>Sat May 04 18:20:20 +0000 2019</td>
      <td>What sound runners eat and when?  Find out in ...</td>
      <td>https://t.co/4g4RkkhYSq</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>35</td>
    </tr>
    <tr>
      <th>45</th>
      <td>lululemon</td>
      <td>Fri May 03 22:45:02 +0000 2019</td>
      <td>Congratulations Sun Choe, lululemon’s Chief Pr...</td>
      <td>https://t.co/tyQY8tKQ86</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>11</td>
      <td>73</td>
    </tr>
    <tr>
      <th>46</th>
      <td>lululemon</td>
      <td>Wed May 01 04:18:02 +0000 2019</td>
      <td>Reflecting on growing up, falling down, and fo...</td>
      <td>https://t.co/4uixGLRtH1</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>11</td>
      <td>46</td>
    </tr>
    <tr>
      <th>47</th>
      <td>lululemon</td>
      <td>Sat Apr 27 18:06:20 +0000 2019</td>
      <td>Learn how trail running can help build strengt...</td>
      <td>https://t.co/KcjS0pbKm3</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>7</td>
      <td>32</td>
    </tr>
    <tr>
      <th>48</th>
      <td>lululemon</td>
      <td>Thu Apr 18 00:32:00 +0000 2019</td>
      <td>Running is how he brings people together to ex...</td>
      <td>https://t.co/jjsd4scINo</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>23</td>
    </tr>
    <tr>
      <th>49</th>
      <td>lululemon</td>
      <td>Wed Apr 17 20:14:00 +0000 2019</td>
      <td>Our newest Global Run Ambassador opens up and ...</td>
      <td>https://t.co/8nE3o8mv3V</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>7</td>
      <td>29</td>
    </tr>
    <tr>
      <th>50</th>
      <td>lululemon</td>
      <td>Sun Apr 14 20:18:19 +0000 2019</td>
      <td>Learn how the track can improve pace, efficien...</td>
      <td>https://t.co/yY2xmADYQl</td>
      <td>No Media included</td>
      <td>[]</td>
      <td>0</td>
      <td>3</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>




```python
tw_data.to_csv("FinalCheck_lululemon.csv")
```
