---
layout: single
title: "How to scrappe Google Trends with Python"
categories: [Google Trends, Python, Text analysis]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

Google trend is an excellent measurement of consumers' attention to specific keywords. Professor Hye Seung Lee wanted to get Google Trends data to identify consumers' attention to negative ESG issues in companies.
The problem with using Google API is that Google has an auto bot to restrict users who request too much data and that the API does not have an extended range of functions to get extensive data set. To solve those problems, I used headers for the bot to consider my IP as everyday users, and I applied season adjustment skills that I learned from EY to shift semi-annual data to three years data. I attached a helpful website to understand how to write 'headers' below.


```python
import pandas as pd                        
from pytrends.request import TrendReq
# from pytrends.request import TrendReq as UTrendReq
GET_METHOD='get'
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
from tqdm import tqdm
```

https://stackoverflow.com/questions/50571317/pytrends-the-request-failed-google-returned-a-response-with-code-429


```python
# headers = {
#     'authority': 'trends.google.com',
#     'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
#     'accept-language': 'en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7',
#     # Requests sorts cookies= alphabetically
#     # 'cookie': '__utmc=10102256; __utmz=10102256.1663271349.4.4.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utma=10102256.946941554.1663014720.1663271349.1663272274.5; __utmt=1; __utmb=10102256.144.9.1663279083347; OGPC=19031016-1:; AEC=AakniGPN80DfaL7odQOG0szxLSbG7HdkLoKuiT5hx7Uctu3_G266ApfSkQ; 1P_JAR=2022-9-15-21; SID=Oggqc8Fgb7hAwjxKRrlI4cj239ozDUFvlhrEYtCgXDcVjwtiPd-Ise8unpj4V2bKx_ObCg.; __Secure-1PSID=Oggqc8Fgb7hAwjxKRrlI4cj239ozDUFvlhrEYtCgXDcVjwticdMt05WF_9O7AAduLIa3-Q.; __Secure-3PSID=Oggqc8Fgb7hAwjxKRrlI4cj239ozDUFvlhrEYtCgXDcVjwtiFoy4XYvNSuDiQLTfga2MHQ.; HSID=AQlz9SZYHUemwUJba; SSID=AyfEb_MsLnEhBn6jG; APISID=rlLvghcqUd1g3sUr/ARSKX93TE17cs-31h; SAPISID=3zlT18C7_l0YzwMk/AW7xTjIEkFGPwsOwN; __Secure-1PAPISID=3zlT18C7_l0YzwMk/AW7xTjIEkFGPwsOwN; __Secure-3PAPISID=3zlT18C7_l0YzwMk/AW7xTjIEkFGPwsOwN; NID=511=s4FRdz8zDBR9s5VoxT32yPsRAcjtptfFFH6YNPXBgPe3OTn0v9o7i4meWmdxew_u88pDqwBPKP6lRhzcYGtcBjNhKef82MSkId_TOTjfP6mA3V7CuAkPQjmt3ALaTjKzWpGy4LxQdlooJEn2P0LiJOAbE1G15Ne4k-kc740kWexIXztLQjNBAJDzDT2oCdnlK3JsjsWCqpcmsvufqAUEHs6QecQGyEH44csUWPFD2lmnO3sjlBNEuw; SIDCC=AEf-XMTGu6FI7hU4on6Gg-R3BVVwgR7Zws1bqy52ofXGjSgSz63JLGT_KliH7zDuSmnaqItouA; __Secure-1PSIDCC=AEf-XMTyaIZsg5ihkgXH-jumlXF0sIa5trW3sw1YcxlCfL9RgL8AwGZlZsmn1HgTD8_4ANnPng; __Secure-3PSIDCC=AEf-XMSd_5Ly3mqByGFlTxblZCTEok4kGQ0kiGVGJU7IomAM2PZcaUT9bfs4xbNw23vK86NyMg',
#     'referer': 'https://trends.google.com/trends/?geo=US',
#     'sec-ch-ua': '"Google Chrome";v="105", "Not)A;Brand";v="8", "Chromium";v="105"',
#     'sec-ch-ua-arch': '"x86"',
#     'sec-ch-ua-bitness': '"64"',
#     'sec-ch-ua-full-version': '"105.0.5195.102"',
#     'sec-ch-ua-mobile': '?0',
#     'sec-ch-ua-model': '""',
#     'sec-ch-ua-platform': '"Windows"',
#     'sec-fetch-dest': 'document',
#     'sec-fetch-mode': 'navigate',
#     'sec-fetch-site': 'same-origin',
#     'sec-fetch-user': '?1',
#     'upgrade-insecure-requests': '1',
#     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
#     'x-client-data': 'CKW1yQEIhbbJAQijtskBCMG2yQEIqZ3KAQjgk8sBCJOhywEI57LMAQiEvMwBCNy8zAEI8sHMAQi7ycwBCOLLzAEIodLMAQj918wBCOXZzAEIwtvMAQio3cwB',
# }

```


```python
# class TrendReq(UTrendReq):
#     def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
#         return super()._get_data(url, method=GET_METHOD, trim_chars=trim_chars, headers=headers, **kwargs)
```


```python
pytrends = TrendReq(hl='en-US', tz=300) # set the time zone as Eastern Standard Time
```


```python
names = pd.read_excel('Samplefirms_updated.xlsx', sheet_name='List') # can also index sheet by name or fetch all sheets
temp_tickers = names['ticker'].tolist()

# because pytrends always parse a string, I should make a list the list which is double covered
tickers = []
for i in temp_tickers:
    tickers += [[i]]
    
temp_com_names = names['Company name'].tolist()
com_names = []
for i in temp_com_names:
    com_names += [[i]]
```


```python
years = ['2018', '2019', '2020']
```


```python
trend_ticker = pd.DataFrame()
temp_ticker = pd.DataFrame()

for i, j in tqdm(enumerate(tickers)):   
    pytrends.build_payload(j, timeframe= '2018-01-01 2018-06-30',geo='',gprop='')
    trend_temp1 = pytrends.interest_over_time()
    trend_temp1 = trend_temp1.drop(columns = 'isPartial')
    time.sleep(2)

    pytrends.build_payload(j, timeframe= '2018-07-01 2018-12-31',geo='',gprop='')
    trend_temp2 = pytrends.interest_over_time()
    trend_temp2 = trend_temp2.drop(columns = 'isPartial')
    time.sleep(1)
    
    if i == 0:
        trend_ticker = pd.concat([trend_temp1, trend_temp2], axis=0)
    else:
        # concat semi annual dataframes
        temp_ticker = pd.concat([trend_temp1, trend_temp2], axis=0)
        # merge another ticker's trends
        trend_ticker = pd.merge(trend_ticker, temp_ticker, left_index=True, right_index=True)
```

    249it [25:30,  6.15s/it]
    


```python
print(trend_ticker)
trend_ticker.to_csv("2018_ticker.csv")
```

                AAP  AAPL  AEO  ALL  AMC  AMTD  AMZN  AN  ANF  ASB  ...  WH  WING  \
    date                                                            ...             
    2018-01-01   60     6   88   74   64     0     3  79   47   60  ...  66    63   
    2018-01-02   61    23  100   71   51    25    15  80   42  100  ...  75    54   
    2018-01-03   67    28   65   68   36     0    18  81   46   82  ...  83    55   
    2018-01-04   59    25   86   69   34     0    18  85   46   64  ...  80    54   
    2018-01-05   68    28   62   73   53    17    17  78   46   53  ...  86    54   
    ...         ...   ...  ...  ...  ...   ...   ...  ..  ...  ...  ...  ..   ...   
    2018-12-27   73    36   59   78   73     0    46  73   80   50  ...  73    68   
    2018-12-28   79    26   59   75   75     0    37  75   81   50  ...  64    74   
    2018-12-29   85     8   43   80   83     0    10  75   84   41  ...  66    81   
    2018-12-30   81     6   27   83   75    37     6  77   76   46  ...  57    87   
    2018-12-31   91    25   47   82   58     0    31  88   78   68  ...  62    85   
    
                WMT  WOOF  WSM  WWW  XOM  XPO  YUM  ZION  
    date                                                  
    2018-01-01    9    49  100   86   45   22   80    17  
    2018-01-02   17    51   56   89   75  100   86    16  
    2018-01-03   20    41   36   89   74   86   80    16  
    2018-01-04   28    43   48   83   67   83   87    19  
    2018-01-05   22    49   35   86   69   75   80    17  
    ...         ...   ...  ...  ...  ...  ...  ...   ...  
    2018-12-27   24    56   77   70   92   42   87    27  
    2018-12-28   21    75   73   69   86   34   87    24  
    2018-12-29    5    53  100   73   64   16   92    27  
    2018-12-30    4    60   81   64   51   10   84    28  
    2018-12-31   17    70   86   63   65   19   86    28  
    
    [365 rows x 249 columns]
    


```python
trend_ticker = pd.DataFrame()
temp_ticker = pd.DataFrame()

for i, j in tqdm(enumerate(tickers)):   
    pytrends.build_payload(j, timeframe= '2019-01-01 2019-06-30',geo='',gprop='')
    trend_temp1 = pytrends.interest_over_time()
    trend_temp1 = trend_temp1.drop(columns = 'isPartial')
    time.sleep(2)

    pytrends.build_payload(j, timeframe= '2019-07-01 2019-12-31',geo='',gprop='')
    trend_temp2 = pytrends.interest_over_time()
    trend_temp2 = trend_temp2.drop(columns = 'isPartial')
    time.sleep(1)
    
    if i == 0:
        trend_ticker = pd.concat([trend_temp1, trend_temp2], axis=0)
    else:
        # concat semi annual dataframes
        temp_ticker = pd.concat([trend_temp1, trend_temp2], axis=0)
        # merge another ticker's trends
        trend_ticker = pd.merge(trend_ticker, temp_ticker, left_index=True, right_index=True)
```

    249it [17:20,  4.18s/it]
    


```python
print(trend_ticker)
trend_ticker.to_csv("2019_ticker.csv")
```

                AAP  AAPL  AEO  ALL  AMC  AMTD  AMZN   AN  ANF  ASB  ...  WH  \
    date                                                             ...       
    2019-01-01   63     7   50   74   71     0    16   80   78   68  ...  76   
    2019-01-02   57    41   87   70   37    49    42   79   80  100  ...  83   
    2019-01-03   59   100   68   74   33     0    54   82   78   70  ...  81   
    2019-01-04   58    60   61   74   43    51    49   81   87   61  ...  84   
    2019-01-05   65    13   54   76   59     0    15   74   81   43  ...  87   
    ...         ...   ...  ...  ...  ...   ...   ...  ...  ...  ...  ...  ..   
    2019-12-27   72    45   68   83   86     0    44   70   39   27  ...  77   
    2019-12-28   78    12   36   88   87     0    11   67   40   22  ...  72   
    2019-12-29   80     6   38   91   83     0     7   69   38   26  ...  67   
    2019-12-30   80    35   60   86   69     0    25   72   42   36  ...  71   
    2019-12-31   88    39   44   88   68     0    26  100   37   34  ...  70   
    
                WING  WMT  WOOF  WSM  WWW  XOM  XPO  YUM  ZION  
    date                                                        
    2019-01-01    43   10    76   55   94   54   15   88     5  
    2019-01-02    33   25    77   29   95   93   64   80     4  
    2019-01-03    37   29    77   32   94   91   68   85     4  
    2019-01-04    40   22    79   30   96   87   53   84     4  
    2019-01-05    42   10    92   22  100   60   25   82     4  
    ...          ...  ...   ...  ...  ...  ...  ...  ...   ...  
    2019-12-27    69   27    69   65   54   57   56   77    28  
    2019-12-28    78   11    66   62   54   44   24   76    26  
    2019-12-29    78    6    55   69   53   46   24   81    28  
    2019-12-30    69   20    57   66   53   55   70   80    24  
    2019-12-31    79   22    53   54   50   61   44   84    24  
    
    [365 rows x 249 columns]
    


```python
trend_ticker = pd.DataFrame()
temp_ticker = pd.DataFrame()

for i, j in tqdm(enumerate(tickers)):   
    pytrends.build_payload(j, timeframe= '2020-01-01 2020-06-30',geo='',gprop='')
    trend_temp1 = pytrends.interest_over_time()
    trend_temp1 = trend_temp1.drop(columns = 'isPartial')
    time.sleep(2)

    pytrends.build_payload(j, timeframe= '2020-07-01 2020-12-31',geo='',gprop='')
    trend_temp2 = pytrends.interest_over_time()
    trend_temp2 = trend_temp2.drop(columns = 'isPartial')
    time.sleep(1)
    
    if i == 0:
        trend_ticker = pd.concat([trend_temp1, trend_temp2], axis=0)
    else:
        # concat semi annual dataframes
        temp_ticker = pd.concat([trend_temp1, trend_temp2], axis=0)
        # merge another ticker's trends
        trend_ticker = pd.merge(trend_ticker, temp_ticker, left_index=True, right_index=True)
```

    249it [18:39,  4.50s/it]
    


```python
print(trend_ticker)
trend_ticker.to_csv("2020_ticker.csv")
```

                AAP  AAPL  AEO  ALL  AMC  AMTD  AMZN  AN  ANF  ASB  ...  WH  WING  \
    date                                                            ...             
    2020-01-01   24    10   53   64  100     0     8  70   58   88  ...  49    56   
    2020-01-02   22    48   63   65   67    51    23  71   56  100  ...  55    58   
    2020-01-03   21    47   57   67   76    14    24  69   63   71  ...  56    54   
    2020-01-04   24    13   51   66   91     0     8  67   61   46  ...  51    60   
    2020-01-05   24     9   46   68   66     0     7  67   65   54  ...  56    58   
    ...         ...   ...  ...  ...  ...   ...   ...  ..  ...  ...  ...  ..   ...   
    2020-12-27   69     5   45   87   60     0     6  62   70   24  ...  61    78   
    2020-12-28   68    34   57   83   63     0    28  64   65   32  ...  72    72   
    2020-12-29   69    35   72   84   60     0    32  68   58   34  ...  64    70   
    2020-12-30   69    31   70   81   63     0    29  64   62   32  ...  64    75   
    2020-12-31   76    28   61   89   59     0    25  86   53   30  ...  60    95   
    
                WMT  WOOF  WSM  WWW  XOM  XPO  YUM  ZION  
    date                                                  
    2020-01-01    9    59  100   16   17   17   88    11  
    2020-01-02   18    85   71   17   26   56   82    11  
    2020-01-03   18    76   63   17   36   49   88    11  
    2020-01-04    9    67   57   17   20   19   88    15  
    2020-01-05    7    64   61   17   20   17   93    12  
    ...         ...   ...  ...  ...  ...  ...  ...   ...  
    2020-12-27    5    57   31   66   43   16   80    46  
    2020-12-28   26    63   33   66   71   65   76    44  
    2020-12-29   26    67   33   69   67   69   82    40  
    2020-12-30   18    65   25   66   70   61   85    47  
    2020-12-31   19    64   33   66   73   44   94    47  
    
    [366 rows x 249 columns]
    


```python
trend_name = pd.DataFrame()
temp_name = pd.DataFrame()

for i, j in tqdm(enumerate(com_names)):   
    pytrends.build_payload(j, timeframe= '2018-01-01 2018-06-30',geo='',gprop='')
    trend_temp1 = pytrends.interest_over_time()
    trend_temp1 = trend_temp1.drop(columns = 'isPartial')
    time.sleep(2)

    pytrends.build_payload(j, timeframe= '2018-07-01 2018-12-31',geo='',gprop='')
    trend_temp2 = pytrends.interest_over_time()
    trend_temp2 = trend_temp2.drop(columns = 'isPartial')
    time.sleep(1)
    
    if i == 0:
        trend_name = pd.concat([trend_temp1, trend_temp2], axis=0)
    else:
        # concat semi annual dataframes
        temp_name = pd.concat([trend_temp1, trend_temp2], axis=0)
        # merge another ticker's trends
        trend_name = pd.merge(trend_name, temp_name, left_index=True, right_index=True)
```

    249it [16:16,  3.92s/it]
    


```python
print(trend_name)
trend_name.to_csv("2018_name.csv")
```

                ADVANCE AUTO PARTS  APPLE  AMERICAN EAGLE OUTFITTERS  ALLSTATE  \
    date                                                                         
    2018-01-01                  43    100                         67        36   
    2018-01-02                  39     99                         80       100   
    2018-01-03                  43     98                         79        65   
    2018-01-04                  33     94                         55        63   
    2018-01-05                  40     94                         63        62   
    ...                        ...    ...                        ...       ...   
    2018-12-27                  73     45                         43        87   
    2018-12-28                  66     41                         32        98   
    2018-12-29                  83     42                         20        65   
    2018-12-30                  76     38                         24        42   
    2018-12-31                  70     34                         27        82   
    
                AMC ENTERTAINMENT HOLDINGS  AMTD IDEA GROUP  AMAZON  AUTONATION   \
    date                                                                           
    2018-01-01                           0                0      94           64   
    2018-01-02                          43                0      99           88   
    2018-01-03                           0                0      99           91   
    2018-01-04                           0               13      97           81   
    2018-01-05                           0               10      93           85   
    ...                                ...              ...     ...          ...   
    2018-12-27                           0                0      61           95   
    2018-12-28                           0                0      54          100   
    2018-12-29                           0                0      57           84   
    2018-12-30                          35                0      53           66   
    2018-12-31                          21                0      47           90   
    
                ABERCROMBIE & FITCH  ASSOCIATED BANC  ...  WYNDHAM HOTELS  \
    date                                              ...                   
    2018-01-01                   57                0  ...              37   
    2018-01-02                   28               27  ...              55   
    2018-01-03                   31                0  ...              49   
    2018-01-04                   48               62  ...              54   
    2018-01-05                   50               34  ...              60   
    ...                         ...              ...  ...             ...   
    2018-12-27                   21                0  ...              47   
    2018-12-28                   24               76  ...              51   
    2018-12-29                   31                0  ...              64   
    2018-12-30                   17                0  ...              44   
    2018-12-31                   18                0  ...              44   
    
                WINGSTOP  WALMART  PETCO HEALTH WELLNESS  WILLIAMS SONOMA  \
    date                                                                    
    2018-01-01        47       98                      0               91   
    2018-01-02        33       68                      0               75   
    2018-01-03        27       70                      0               66   
    2018-01-04        26       69                      0               72   
    2018-01-05        33       65                      0               72   
    ...              ...      ...                    ...              ...   
    2018-12-27        54       35                      0               64   
    2018-12-28        62       34                      0               60   
    2018-12-29        83       36                      0               53   
    2018-12-30        87       34                      0               54   
    2018-12-31        84       39                      0               49   
    
                WOLVERINE WORLD WIDE  EXXON MOBIL  XPO LOGISTICS  YUM BRANDS  \
    date                                                                       
    2018-01-01                    51           31             24          27   
    2018-01-02                    31           45            100          47   
    2018-01-03                    24           55             73           0   
    2018-01-04                     0           44             81          36   
    2018-01-05                     0           45             76          20   
    ...                          ...          ...            ...         ...   
    2018-12-27                     0           54             33          33   
    2018-12-28                     0           56             23           0   
    2018-12-29                     0           39             14          33   
    2018-12-30                     0           27             10           0   
    2018-12-31                     0           45             13          42   
    
                ZIONS BANCORPORATION  
    date                              
    2018-01-01                     0  
    2018-01-02                     0  
    2018-01-03                     0  
    2018-01-04                     0  
    2018-01-05                     0  
    ...                          ...  
    2018-12-27                   100  
    2018-12-28                     0  
    2018-12-29                     0  
    2018-12-30                     0  
    2018-12-31                     0  
    
    [365 rows x 249 columns]
    


```python
trend_name = pd.DataFrame()
temp_name = pd.DataFrame()

for i, j in tqdm(enumerate(com_names)):   
    pytrends.build_payload(j, timeframe= '2019-01-01 2019-06-30',geo='',gprop='')
    trend_temp1 = pytrends.interest_over_time()
    trend_temp1 = trend_temp1.drop(columns = 'isPartial')
    time.sleep(2)

    pytrends.build_payload(j, timeframe= '2019-07-01 2019-12-31',geo='',gprop='')
    trend_temp2 = pytrends.interest_over_time()
    trend_temp2 = trend_temp2.drop(columns = 'isPartial')
    time.sleep(1)
    
    if i == 0:
        trend_name = pd.concat([trend_temp1, trend_temp2], axis=0)
    else:
        # concat semi annual dataframes
        temp_name = pd.concat([trend_temp1, trend_temp2], axis=0)
        # merge another ticker's trends
        trend_name = pd.merge(trend_name, temp_name, left_index=True, right_index=True)
        if i % 50 == 0:
            time.sleep(60)
```

    249it [21:28,  5.17s/it]
    


```python
print(trend_name)
trend_name.to_csv("2019_name.csv")
```

                ADVANCE AUTO PARTS  APPLE  AMERICAN EAGLE OUTFITTERS  ALLSTATE  \
    date                                                                         
    2019-01-01                  71     81                         60        45   
    2019-01-02                  59     86                         51       100   
    2019-01-03                  56     92                         78        69   
    2019-01-04                  65     89                         60        69   
    2019-01-05                  83     83                         77        45   
    ...                        ...    ...                        ...       ...   
    2019-12-27                  70     38                         36        87   
    2019-12-28                  78     36                         37        63   
    2019-12-29                  64     34                         39        51   
    2019-12-30                  56     33                         35        98   
    2019-12-31                  57     30                         38        83   
    
                AMC ENTERTAINMENT HOLDINGS  AMTD IDEA GROUP  AMAZON  AUTONATION   \
    date                                                                           
    2019-01-01                          36                0      91           64   
    2019-01-02                           0               21     100           82   
    2019-01-03                           0                0      95           91   
    2019-01-04                           0                0      92           88   
    2019-01-05                           0                0      97           84   
    ...                                ...              ...     ...          ...   
    2019-12-27                           0                0      61           72   
    2019-12-28                           0                0      60           67   
    2019-12-29                           0                0      57           63   
    2019-12-30                          22               56      57           76   
    2019-12-31                           0                0      49           68   
    
                ABERCROMBIE & FITCH  ASSOCIATED BANC  ...  WYNDHAM HOTELS  \
    date                                              ...                   
    2019-01-01                   46                0  ...              42   
    2019-01-02                   29                0  ...              51   
    2019-01-03                   53               71  ...              47   
    2019-01-04                   45                0  ...              47   
    2019-01-05                   31               30  ...              41   
    ...                         ...              ...  ...             ...   
    2019-12-27                   43                0  ...              55   
    2019-12-28                   62                0  ...              54   
    2019-12-29                   47                0  ...              61   
    2019-12-30                   58               51  ...              58   
    2019-12-31                   40                0  ...              56   
    
                WINGSTOP  WALMART  PETCO HEALTH WELLNESS  WILLIAMS SONOMA  \
    date                                                                    
    2019-01-01        44       95                      0              100   
    2019-01-02        29       63                      0               82   
    2019-01-03        23       63                      0               83   
    2019-01-04        28       64                      0               87   
    2019-01-05        41       71                      0               98   
    ...              ...      ...                    ...              ...   
    2019-12-27        57       36                      0               54   
    2019-12-28        70       34                     36               53   
    2019-12-29        68       33                      0               50   
    2019-12-30        51       30                      0               46   
    2019-12-31        64       37                     51               44   
    
                WOLVERINE WORLD WIDE  EXXON MOBIL  XPO LOGISTICS  YUM BRANDS  \
    date                                                                       
    2019-01-01                     0           23             22          23   
    2019-01-02                     0           84             71          37   
    2019-01-03                     0           92             78          28   
    2019-01-04                    99           78             71          34   
    2019-01-05                    86           60             31           0   
    ...                          ...          ...            ...         ...   
    2019-12-27                     0           50             56          47   
    2019-12-28                    24           31             34           0   
    2019-12-29                     0           28             13          15   
    2019-12-30                     0           44             66          34   
    2019-12-31                     0           51             37          33   
    
                ZIONS BANCORPORATION  
    date                              
    2019-01-01                     0  
    2019-01-02                     0  
    2019-01-03                     0  
    2019-01-04                     0  
    2019-01-05                     0  
    ...                          ...  
    2019-12-27                    19  
    2019-12-28                     0  
    2019-12-29                    64  
    2019-12-30                     0  
    2019-12-31                     0  
    
    [365 rows x 249 columns]
    


```python
trend_name = pd.DataFrame()
temp_name = pd.DataFrame()

for i, j in tqdm(enumerate(com_names)):   
    pytrends.build_payload(j, timeframe= '2020-01-01 2020-06-30',geo='',gprop='')
    trend_temp1 = pytrends.interest_over_time()
    trend_temp1 = trend_temp1.drop(columns = 'isPartial')
    time.sleep(2)

    pytrends.build_payload(j, timeframe= '2020-07-01 2020-12-31',geo='',gprop='')
    trend_temp2 = pytrends.interest_over_time()
    trend_temp2 = trend_temp2.drop(columns = 'isPartial')
    time.sleep(1)
    
    if i == 0:
        trend_name = pd.concat([trend_temp1, trend_temp2], axis=0)
    else:
        # concat semi annual dataframes
        temp_name = pd.concat([trend_temp1, trend_temp2], axis=0)
        # merge another ticker's trends
        trend_name = pd.merge(trend_name, temp_name, left_index=True, right_index=True)
        if i % 50 == 0:
            time.sleep(60)
```

    249it [21:09,  5.10s/it]
    


```python
print(trend_name)
trend_name.to_csv("2020_name.csv")
```

                ADVANCE AUTO PARTS  APPLE  AMERICAN EAGLE OUTFITTERS  ALLSTATE  \
    date                                                                         
    2020-01-01                  71     87                         47        42   
    2020-01-02                  62     88                         63        78   
    2020-01-03                  66     79                         57        70   
    2020-01-04                  70     84                         85        43   
    2020-01-05                  71     85                         73        32   
    ...                        ...    ...                        ...       ...   
    2020-12-27                  65     61                         45        47   
    2020-12-28                  62     56                         27        80   
    2020-12-29                  57     54                         25        85   
    2020-12-30                  63     51                         31        81   
    2020-12-31                  76     52                         29        80   
    
                AMC ENTERTAINMENT HOLDINGS  AMTD IDEA GROUP  AMAZON  AUTONATION   \
    date                                                                           
    2020-01-01                           0              100      68           56   
    2020-01-02                           0                0      77           79   
    2020-01-03                           0                0      74           84   
    2020-01-04                           0                0      71           82   
    2020-01-05                           0                0      73           62   
    ...                                ...              ...     ...          ...   
    2020-12-27                           0                0      72           54   
    2020-12-28                           0                0      70           95   
    2020-12-29                          61                0      67           92   
    2020-12-30                           0                0      65           82   
    2020-12-31                           0                0      56           97   
    
                ABERCROMBIE & FITCH  ASSOCIATED BANC  ...  WYNDHAM HOTELS  \
    date                                              ...                   
    2020-01-01                   86                0  ...              42   
    2020-01-02                   90                0  ...              48   
    2020-01-03                   44                0  ...              53   
    2020-01-04                   67                0  ...              45   
    2020-01-05                   82                0  ...              43   
    ...                         ...              ...  ...             ...   
    2020-12-27                   48                0  ...              55   
    2020-12-28                   55                0  ...              45   
    2020-12-29                   48                0  ...              61   
    2020-12-30                   51                0  ...              64   
    2020-12-31                   50               61  ...              70   
    
                WINGSTOP  WALMART  PETCO HEALTH WELLNESS  WILLIAMS SONOMA  \
    date                                                                    
    2020-01-01        49       85                      0               85   
    2020-01-02        33       58                      0               74   
    2020-01-03        33       57                     38               82   
    2020-01-04        41       65                      0               83   
    2020-01-05        43       60                      0               77   
    ...              ...      ...                    ...              ...   
    2020-12-27        76       43                      0               65   
    2020-12-28        60       38                      0               49   
    2020-12-29        55       37                      0               47   
    2020-12-30        58       39                     79               44   
    2020-12-31       100       48                      0               45   
    
                WOLVERINE WORLD WIDE  EXXON MOBIL  XPO LOGISTICS  YUM BRANDS  \
    date                                                                       
    2020-01-01                     0           10             14          19   
    2020-01-02                     0           23             48          28   
    2020-01-03                    41           28             50          27   
    2020-01-04                     0           21             25           0   
    2020-01-05                    70           14             20           8   
    ...                          ...          ...            ...         ...   
    2020-12-27                     0           13             18          12   
    2020-12-28                    48           37             60          32   
    2020-12-29                     0           47             75          30   
    2020-12-30                     0           39             61          43   
    2020-12-31                    51           47             47          33   
    
                ZIONS BANCORPORATION  
    date                              
    2020-01-01                    32  
    2020-01-02                    49  
    2020-01-03                    32  
    2020-01-04                    26  
    2020-01-05                     0  
    ...                          ...  
    2020-12-27                     0  
    2020-12-28                     0  
    2020-12-29                     0  
    2020-12-30                    34  
    2020-12-31                     0  
    
    [366 rows x 249 columns]
    
