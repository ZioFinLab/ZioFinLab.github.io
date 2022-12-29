---
layout: single
title: "Stock construction model with Neural Networks (Fall 2022)"
categories: [Prediction model, Python, Stock selection, Portfolio management, Neural Networks]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
**[Notice]** [Journey to the academic researcher](https://ziofinlab.github.io/biography/bio/)
This is the story of how I became the insightful researcher.
{: .notice--info}


This is a stock selection model through neural networks technology. Based on more than ten years of previous data, the model returns the set of stocks that can have the highest returns through the neural networks.

## Fall 2022 | Instructor: Lin Tong | Student: Jihong Yu

In this example, we will randomly select 100 stocks from the database "rpsdata_rfs_cleaned_2000.csv" using the random seed 2021. We will do some data preparations for the subsequent procesures. 

## Data Preparation


```python
import pandas as pd
df = pd.read_csv("rpsdata_rfs_cleaned_2000.csv")
df["datadate"]=pd.to_datetime(df["datadate"],format="%Y-%m-%d")
df["DATE"]=pd.to_datetime(df["DATE"],format="%Y-%m-%d")
df.head()
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
      <th>permno</th>
      <th>fyear</th>
      <th>sic2</th>
      <th>spi</th>
      <th>mve_f</th>
      <th>bm</th>
      <th>ep</th>
      <th>cashpr</th>
      <th>dy</th>
      <th>lev</th>
      <th>...</th>
      <th>baspread</th>
      <th>std_dolvol</th>
      <th>std_turn</th>
      <th>ill</th>
      <th>zerotrade</th>
      <th>BETA</th>
      <th>betasq</th>
      <th>rsq1</th>
      <th>pricedelay</th>
      <th>idiovol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10001</td>
      <td>1999</td>
      <td>49</td>
      <td>0.000000</td>
      <td>20.99325</td>
      <td>0.644588</td>
      <td>0.075596</td>
      <td>-24.025440</td>
      <td>0.054541</td>
      <td>1.416217</td>
      <td>...</td>
      <td>0.012581</td>
      <td>1.078260</td>
      <td>0.597444</td>
      <td>4.016382e-06</td>
      <td>3.818182e+00</td>
      <td>0.062780</td>
      <td>0.003941</td>
      <td>-0.004141</td>
      <td>1.905185</td>
      <td>0.025807</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10002</td>
      <td>1998</td>
      <td>60</td>
      <td>0.000000</td>
      <td>115.71000</td>
      <td>0.509429</td>
      <td>0.048181</td>
      <td>-6.261051</td>
      <td>0.021191</td>
      <td>3.844966</td>
      <td>...</td>
      <td>0.035375</td>
      <td>1.163444</td>
      <td>0.172986</td>
      <td>3.703263e-06</td>
      <td>7.636364e+00</td>
      <td>0.487761</td>
      <td>0.237910</td>
      <td>0.033295</td>
      <td>-0.025434</td>
      <td>0.050570</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10009</td>
      <td>1998</td>
      <td>60</td>
      <td>0.000000</td>
      <td>50.61000</td>
      <td>0.757597</td>
      <td>0.095673</td>
      <td>-28.390070</td>
      <td>0.022782</td>
      <td>10.058840</td>
      <td>...</td>
      <td>0.031896</td>
      <td>1.046077</td>
      <td>4.829668</td>
      <td>2.535039e-07</td>
      <td>2.881086e-08</td>
      <td>0.299177</td>
      <td>0.089507</td>
      <td>0.022911</td>
      <td>0.391768</td>
      <td>0.035277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10012</td>
      <td>1998</td>
      <td>36</td>
      <td>-0.101781</td>
      <td>34.62894</td>
      <td>0.130151</td>
      <td>-0.105057</td>
      <td>20.815490</td>
      <td>0.000000</td>
      <td>0.166681</td>
      <td>...</td>
      <td>0.112558</td>
      <td>0.712583</td>
      <td>23.904390</td>
      <td>1.265005e-08</td>
      <td>3.353590e-09</td>
      <td>2.389729</td>
      <td>5.710806</td>
      <td>0.088599</td>
      <td>-0.264245</td>
      <td>0.145356</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10016</td>
      <td>1998</td>
      <td>38</td>
      <td>-0.003628</td>
      <td>300.37280</td>
      <td>0.183991</td>
      <td>0.031607</td>
      <td>2.865464</td>
      <td>0.000000</td>
      <td>0.692200</td>
      <td>...</td>
      <td>0.043094</td>
      <td>1.005188</td>
      <td>2.595967</td>
      <td>8.944278e-08</td>
      <td>3.859758e-08</td>
      <td>0.619844</td>
      <td>0.384207</td>
      <td>0.093318</td>
      <td>-0.096680</td>
      <td>0.039299</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 147 columns</p>
</div>



Randomly sample stocks. 


```python
df.head()
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
      <th>permno</th>
      <th>fyear</th>
      <th>sic2</th>
      <th>spi</th>
      <th>mve_f</th>
      <th>bm</th>
      <th>ep</th>
      <th>cashpr</th>
      <th>dy</th>
      <th>lev</th>
      <th>...</th>
      <th>baspread</th>
      <th>std_dolvol</th>
      <th>std_turn</th>
      <th>ill</th>
      <th>zerotrade</th>
      <th>BETA</th>
      <th>betasq</th>
      <th>rsq1</th>
      <th>pricedelay</th>
      <th>idiovol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10001</td>
      <td>1999</td>
      <td>49</td>
      <td>0.000000</td>
      <td>20.99325</td>
      <td>0.644588</td>
      <td>0.075596</td>
      <td>-24.025440</td>
      <td>0.054541</td>
      <td>1.416217</td>
      <td>...</td>
      <td>0.012581</td>
      <td>1.078260</td>
      <td>0.597444</td>
      <td>4.016382e-06</td>
      <td>3.818182e+00</td>
      <td>0.062780</td>
      <td>0.003941</td>
      <td>-0.004141</td>
      <td>1.905185</td>
      <td>0.025807</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10002</td>
      <td>1998</td>
      <td>60</td>
      <td>0.000000</td>
      <td>115.71000</td>
      <td>0.509429</td>
      <td>0.048181</td>
      <td>-6.261051</td>
      <td>0.021191</td>
      <td>3.844966</td>
      <td>...</td>
      <td>0.035375</td>
      <td>1.163444</td>
      <td>0.172986</td>
      <td>3.703263e-06</td>
      <td>7.636364e+00</td>
      <td>0.487761</td>
      <td>0.237910</td>
      <td>0.033295</td>
      <td>-0.025434</td>
      <td>0.050570</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10009</td>
      <td>1998</td>
      <td>60</td>
      <td>0.000000</td>
      <td>50.61000</td>
      <td>0.757597</td>
      <td>0.095673</td>
      <td>-28.390070</td>
      <td>0.022782</td>
      <td>10.058840</td>
      <td>...</td>
      <td>0.031896</td>
      <td>1.046077</td>
      <td>4.829668</td>
      <td>2.535039e-07</td>
      <td>2.881086e-08</td>
      <td>0.299177</td>
      <td>0.089507</td>
      <td>0.022911</td>
      <td>0.391768</td>
      <td>0.035277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10012</td>
      <td>1998</td>
      <td>36</td>
      <td>-0.101781</td>
      <td>34.62894</td>
      <td>0.130151</td>
      <td>-0.105057</td>
      <td>20.815490</td>
      <td>0.000000</td>
      <td>0.166681</td>
      <td>...</td>
      <td>0.112558</td>
      <td>0.712583</td>
      <td>23.904390</td>
      <td>1.265005e-08</td>
      <td>3.353590e-09</td>
      <td>2.389729</td>
      <td>5.710806</td>
      <td>0.088599</td>
      <td>-0.264245</td>
      <td>0.145356</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10016</td>
      <td>1998</td>
      <td>38</td>
      <td>-0.003628</td>
      <td>300.37280</td>
      <td>0.183991</td>
      <td>0.031607</td>
      <td>2.865464</td>
      <td>0.000000</td>
      <td>0.692200</td>
      <td>...</td>
      <td>0.043094</td>
      <td>1.005188</td>
      <td>2.595967</td>
      <td>8.944278e-08</td>
      <td>3.859758e-08</td>
      <td>0.619844</td>
      <td>0.384207</td>
      <td>0.093318</td>
      <td>-0.096680</td>
      <td>0.039299</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 147 columns</p>
</div>




```python
import random
num_stock=100   #The number of stocks.
random.seed(2021)
stocklist =  random.sample(list(df.permno.unique()),num_stock) 
df = df[df.permno.isin(stocklist)].copy()
del df["datadate"]
del df["fyear"]
df.sort_values(by=["permno","DATE"],inplace=True)
df.reset_index(drop=True,inplace=True)
```


```python
df.head()
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
      <th>permno</th>
      <th>sic2</th>
      <th>spi</th>
      <th>mve_f</th>
      <th>bm</th>
      <th>ep</th>
      <th>cashpr</th>
      <th>dy</th>
      <th>lev</th>
      <th>sp</th>
      <th>...</th>
      <th>baspread</th>
      <th>std_dolvol</th>
      <th>std_turn</th>
      <th>ill</th>
      <th>zerotrade</th>
      <th>BETA</th>
      <th>betasq</th>
      <th>rsq1</th>
      <th>pricedelay</th>
      <th>idiovol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10325</td>
      <td>56</td>
      <td>-0.010241</td>
      <td>83.51094</td>
      <td>0.936201</td>
      <td>0.024356</td>
      <td>-17.92364</td>
      <td>0.010083</td>
      <td>2.94395</td>
      <td>6.996401</td>
      <td>...</td>
      <td>0.069598</td>
      <td>0.712721</td>
      <td>3.341473</td>
      <td>1.221359e-07</td>
      <td>1.587679e-08</td>
      <td>1.801611</td>
      <td>3.245802</td>
      <td>0.150400</td>
      <td>0.177043</td>
      <td>0.086112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10325</td>
      <td>56</td>
      <td>-0.010241</td>
      <td>83.51094</td>
      <td>0.936201</td>
      <td>0.024356</td>
      <td>-17.92364</td>
      <td>0.010083</td>
      <td>2.94395</td>
      <td>6.996401</td>
      <td>...</td>
      <td>0.070863</td>
      <td>0.803676</td>
      <td>2.309144</td>
      <td>2.448277e-07</td>
      <td>2.721601e-08</td>
      <td>1.846191</td>
      <td>3.408423</td>
      <td>0.151738</td>
      <td>0.186343</td>
      <td>0.088105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10325</td>
      <td>56</td>
      <td>-0.010241</td>
      <td>83.51094</td>
      <td>0.936201</td>
      <td>0.024356</td>
      <td>-17.92364</td>
      <td>0.010083</td>
      <td>2.94395</td>
      <td>6.996401</td>
      <td>...</td>
      <td>0.063259</td>
      <td>0.721935</td>
      <td>4.526667</td>
      <td>9.009622e-08</td>
      <td>1.925423e-08</td>
      <td>1.792151</td>
      <td>3.211805</td>
      <td>0.144251</td>
      <td>0.250596</td>
      <td>0.087058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10325</td>
      <td>56</td>
      <td>-0.010241</td>
      <td>83.51094</td>
      <td>0.936201</td>
      <td>0.024356</td>
      <td>-17.92364</td>
      <td>0.010083</td>
      <td>2.94395</td>
      <td>6.996401</td>
      <td>...</td>
      <td>0.061893</td>
      <td>0.781739</td>
      <td>4.661429</td>
      <td>7.737564e-08</td>
      <td>1.564298e-08</td>
      <td>1.818809</td>
      <td>3.308067</td>
      <td>0.155762</td>
      <td>0.111497</td>
      <td>0.087586</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10325</td>
      <td>56</td>
      <td>-0.010241</td>
      <td>83.51094</td>
      <td>0.936201</td>
      <td>0.024356</td>
      <td>-17.92364</td>
      <td>0.010083</td>
      <td>2.94395</td>
      <td>6.996401</td>
      <td>...</td>
      <td>0.055615</td>
      <td>0.968370</td>
      <td>2.274174</td>
      <td>1.641843e-07</td>
      <td>5.060460e-08</td>
      <td>1.853151</td>
      <td>3.434170</td>
      <td>0.164967</td>
      <td>0.159384</td>
      <td>0.087659</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 145 columns</p>
</div>



When the data table contains many NaNs, we must be very careful in dropping NaNs in order to keep as many rows and columns as possible.

1. Remove all columns except the ones with less than 20% NaNs
2. Apply imputation to the remaining columns.

We can change 20% to other threshold to optimize our performance. The following code applies this procedure to a copy of **df**.


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
nanpercent = 0.2
dftemp = df.copy()
#Remove all columns except the ones with less than nanpercent NaNs
dftemp = dftemp[dftemp.columns[dftemp.isna().mean(axis=0)<nanpercent]]  
#Impute the remaining columns. 
dftemp.loc[:,dftemp.columns[dftemp.isna().sum(axis=0)>0]] = imputer.fit_transform(dftemp.loc[:,dftemp.columns[dftemp.isna().sum(axis=0)>0]]) 
print("Before")
print(df.shape)
print("After")
print(dftemp.shape)
```

    Before
    (8422, 145)
    After
    (8422, 121)



```python
dftemp = dftemp.set_index('permno')
```


```python
featurename=list(dftemp.columns)
featurename.remove("DATE")
featurename.remove("RET")
targetname="RET"
```

## Building Neural Network


```python
from sklearn import preprocessing
import datetime
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import random as python_random
#Set up the random seeds to reproduce results.
import statistics
import matplotlib.pyplot as plt
```


```python
def data_dividor(test_start_date, test_end_date):
    """divide the data based on the target_date"""
    train_x = dftemp[dftemp.DATE<test_start_date].loc[:,featurename]
    train_y = dftemp[dftemp.DATE<test_start_date].loc[:,targetname]
    test_x = dftemp[(dftemp.DATE>=test_start_date) & (dftemp.DATE<test_end_date)].loc[:,featurename]
    test_y = dftemp[(dftemp.DATE>=test_start_date) & (dftemp.DATE<test_end_date)].loc[:,targetname]
    
    # scaling the data from 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    test_x = min_max_scaler.transform(test_x)
    
    # returns training and test sets
    return train_x, train_y, test_x, test_y
```


```python
def nn_builder(train_x, train_y, test_x):
    """Building the neural network model to predict stock prices"""
    keras.backend.clear_session()  #Clean the session to reset the model/layer ID
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(128, activation='sigmoid',input_shape=(train_x.shape[1],)),
      tf.keras.layers.Dense(16, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='linear')
    ])
    # model.summary()

    # Set the concrete random numbers
    os.environ['PYTHONHASHSEED']=str(0)
    np.random.seed(2021)
    python_random.seed(2021)
    tf.random.set_seed(2021)

    #Define an optimizer, for example, Adam
    opt = keras.optimizers.Adam(learning_rate=0.01)

    #Compile the model
    model.compile(optimizer=opt,   #Set optimizer='adam' if you want to use default learning rate.
                  loss='mean_squared_error',
                  metrics='mean_squared_error')

    #Train and record how the performance metrics changes during training. 
    model.fit(train_x, train_y, 
                    verbose=0,
                    batch_size=1000, 
                    epochs=50
                   )
    
    forecaster= model.predict(test_x)
    return forecaster
```


```python
start_date = "2010-01-01"
terminal_date = "2020-01-01"
s_date = pd.to_datetime(start_date)
e_date = pd.to_datetime(terminal_date)
iter_num = int(relativedelta(e_date, s_date).years)
```


```python
high_mean = []
high_stv = []
low_mean = []
low_stv = []
temp_end = s_date

for i in range(1,iter_num+1):
    if temp_end == e_date:
        temp_end = s_date
        break
    if i == 1:
        temp_end = s_date + relativedelta(years=i)
        print("="*100)
        print("NN model construction")
        print("start date: ",s_date)
        print("end date: ",temp_end)
        
        train_x, train_y, test_x, test_y = data_dividor(s_date, temp_end)
        dftemp.loc[(dftemp.DATE>=s_date) & (dftemp.DATE<temp_end),"Signal"] = nn_builder(train_x, train_y, test_x)
        
        for j in range(1,13):
            if j == 1:
                temp_month_start = s_date
                temp_month_end = s_date + relativedelta(months = j)
                print("-"*100)
                print("selecting monthly highest and lowest")
                print("start date: ",temp_month_start)
                print("end date: ",temp_month_end)
                
                highest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nlargest(10, "Signal")
                lowest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nsmallest(10, "Signal")

                high_mean.append(statistics.mean(highest_df.RET))
                high_stv.append(statistics.stdev(highest_df.RET))        
                low_mean.append(statistics.mean(lowest_df.RET))
                low_stv.append(statistics.stdev(lowest_df.RET))
            
            else:
                temp_month_start = temp_month_end
                temp_month_end = temp_month_start + relativedelta(months = 1)
                print("-"*100)
                print("selecting monthly highest and lowest")
                print("start date: ",temp_month_start)
                print("end date: ",temp_month_end)
                
                highest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nlargest(10, "Signal")
                lowest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nsmallest(10, "Signal")

                high_mean.append(statistics.mean(highest_df.RET))
                high_stv.append(statistics.stdev(highest_df.RET))
                low_mean.append(statistics.mean(lowest_df.RET))
                low_stv.append(statistics.stdev(lowest_df.RET))
                
            
    else:
        temp_start = temp_end
        temp_end = temp_start + relativedelta(years=1)
        print("="*100)
        print("NN model construction")
        print("start date: ",temp_start)
        print("end date: ",temp_end)
        train_x, train_y, test_x, test_y = data_dividor(temp_start, temp_end)
        dftemp.loc[(dftemp.DATE>=temp_start) & (dftemp.DATE<temp_end),"Signal"] = nn_builder(train_x, train_y, test_x)
        
        for j in range(1,13):
            if j == 1:
                temp_month_start = temp_start
                temp_month_end = temp_start + relativedelta(months = j)
                print("-"*100)
                print("selecting monthly highest and lowest")
                print("start date: ",temp_month_start)
                print("end date: ",temp_month_end)
                
                highest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nlargest(10, "Signal")
                lowest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nsmallest(10, "Signal")

                high_mean.append(statistics.mean(highest_df.RET))
                high_stv.append(statistics.stdev(highest_df.RET))        
                low_mean.append(statistics.mean(lowest_df.RET))
                low_stv.append(statistics.stdev(lowest_df.RET))
                
            else:
                temp_month_start = temp_month_end
                temp_month_end = temp_month_start + relativedelta(months = 1)
                print("-"*100)
                print("selecting monthly highest and lowest")
                print("start date: ",temp_month_start)
                print("end date: ",temp_month_end)
                
                highest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nlargest(10, "Signal")
                lowest_df = dftemp.loc[(dftemp.DATE>=temp_month_start) & (dftemp.DATE<temp_month_end)].nsmallest(10, "Signal")

                high_mean.append(statistics.mean(highest_df.RET))
                high_stv.append(statistics.stdev(highest_df.RET))
                low_mean.append(statistics.mean(lowest_df.RET))
                low_stv.append(statistics.stdev(lowest_df.RET))
```

    ====================================================================================================
    NN model construction
    start date:  2010-01-01 00:00:00
    end date:  2011-01-01 00:00:00
    11/11 [==============================] - 0s 850us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-01-01 00:00:00
    end date:  2010-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-02-01 00:00:00
    end date:  2010-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-03-01 00:00:00
    end date:  2010-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-04-01 00:00:00
    end date:  2010-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-05-01 00:00:00
    end date:  2010-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-06-01 00:00:00
    end date:  2010-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-07-01 00:00:00
    end date:  2010-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-08-01 00:00:00
    end date:  2010-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-09-01 00:00:00
    end date:  2010-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-10-01 00:00:00
    end date:  2010-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-11-01 00:00:00
    end date:  2010-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2010-12-01 00:00:00
    end date:  2011-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2011-01-01 00:00:00
    end date:  2012-01-01 00:00:00
    11/11 [==============================] - 0s 649us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-01-01 00:00:00
    end date:  2011-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-02-01 00:00:00
    end date:  2011-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-03-01 00:00:00
    end date:  2011-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-04-01 00:00:00
    end date:  2011-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-05-01 00:00:00
    end date:  2011-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-06-01 00:00:00
    end date:  2011-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-07-01 00:00:00
    end date:  2011-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-08-01 00:00:00
    end date:  2011-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-09-01 00:00:00
    end date:  2011-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-10-01 00:00:00
    end date:  2011-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-11-01 00:00:00
    end date:  2011-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2011-12-01 00:00:00
    end date:  2012-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2012-01-01 00:00:00
    end date:  2013-01-01 00:00:00
    11/11 [==============================] - 0s 932us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-01-01 00:00:00
    end date:  2012-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-02-01 00:00:00
    end date:  2012-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-03-01 00:00:00
    end date:  2012-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-04-01 00:00:00
    end date:  2012-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-05-01 00:00:00
    end date:  2012-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-06-01 00:00:00
    end date:  2012-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-07-01 00:00:00
    end date:  2012-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-08-01 00:00:00
    end date:  2012-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-09-01 00:00:00
    end date:  2012-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-10-01 00:00:00
    end date:  2012-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-11-01 00:00:00
    end date:  2012-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2012-12-01 00:00:00
    end date:  2013-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2013-01-01 00:00:00
    end date:  2014-01-01 00:00:00
    11/11 [==============================] - 0s 689us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-01-01 00:00:00
    end date:  2013-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-02-01 00:00:00
    end date:  2013-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-03-01 00:00:00
    end date:  2013-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-04-01 00:00:00
    end date:  2013-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-05-01 00:00:00
    end date:  2013-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-06-01 00:00:00
    end date:  2013-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-07-01 00:00:00
    end date:  2013-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-08-01 00:00:00
    end date:  2013-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-09-01 00:00:00
    end date:  2013-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-10-01 00:00:00
    end date:  2013-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-11-01 00:00:00
    end date:  2013-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2013-12-01 00:00:00
    end date:  2014-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2014-01-01 00:00:00
    end date:  2015-01-01 00:00:00
    11/11 [==============================] - 0s 885us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-01-01 00:00:00
    end date:  2014-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-02-01 00:00:00
    end date:  2014-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-03-01 00:00:00
    end date:  2014-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-04-01 00:00:00
    end date:  2014-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-05-01 00:00:00
    end date:  2014-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-06-01 00:00:00
    end date:  2014-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-07-01 00:00:00
    end date:  2014-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-08-01 00:00:00
    end date:  2014-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-09-01 00:00:00
    end date:  2014-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-10-01 00:00:00
    end date:  2014-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-11-01 00:00:00
    end date:  2014-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2014-12-01 00:00:00
    end date:  2015-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2015-01-01 00:00:00
    end date:  2016-01-01 00:00:00
    12/12 [==============================] - 0s 818us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-01-01 00:00:00
    end date:  2015-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-02-01 00:00:00
    end date:  2015-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-03-01 00:00:00
    end date:  2015-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-04-01 00:00:00
    end date:  2015-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-05-01 00:00:00
    end date:  2015-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-06-01 00:00:00
    end date:  2015-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-07-01 00:00:00
    end date:  2015-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-08-01 00:00:00
    end date:  2015-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-09-01 00:00:00
    end date:  2015-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-10-01 00:00:00
    end date:  2015-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-11-01 00:00:00
    end date:  2015-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2015-12-01 00:00:00
    end date:  2016-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2016-01-01 00:00:00
    end date:  2017-01-01 00:00:00
    13/13 [==============================] - 0s 935us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-01-01 00:00:00
    end date:  2016-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-02-01 00:00:00
    end date:  2016-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-03-01 00:00:00
    end date:  2016-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-04-01 00:00:00
    end date:  2016-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-05-01 00:00:00
    end date:  2016-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-06-01 00:00:00
    end date:  2016-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-07-01 00:00:00
    end date:  2016-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-08-01 00:00:00
    end date:  2016-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-09-01 00:00:00
    end date:  2016-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-10-01 00:00:00
    end date:  2016-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-11-01 00:00:00
    end date:  2016-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2016-12-01 00:00:00
    end date:  2017-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2017-01-01 00:00:00
    end date:  2018-01-01 00:00:00
    12/12 [==============================] - 0s 1ms/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-01-01 00:00:00
    end date:  2017-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-02-01 00:00:00
    end date:  2017-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-03-01 00:00:00
    end date:  2017-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-04-01 00:00:00
    end date:  2017-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-05-01 00:00:00
    end date:  2017-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-06-01 00:00:00
    end date:  2017-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-07-01 00:00:00
    end date:  2017-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-08-01 00:00:00
    end date:  2017-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-09-01 00:00:00
    end date:  2017-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-10-01 00:00:00
    end date:  2017-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-11-01 00:00:00
    end date:  2017-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2017-12-01 00:00:00
    end date:  2018-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2018-01-01 00:00:00
    end date:  2019-01-01 00:00:00
    12/12 [==============================] - 0s 809us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-01-01 00:00:00
    end date:  2018-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-02-01 00:00:00
    end date:  2018-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-03-01 00:00:00
    end date:  2018-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-04-01 00:00:00
    end date:  2018-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-05-01 00:00:00
    end date:  2018-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-06-01 00:00:00
    end date:  2018-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-07-01 00:00:00
    end date:  2018-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-08-01 00:00:00
    end date:  2018-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-09-01 00:00:00
    end date:  2018-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-10-01 00:00:00
    end date:  2018-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-11-01 00:00:00
    end date:  2018-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2018-12-01 00:00:00
    end date:  2019-01-01 00:00:00
    ====================================================================================================
    NN model construction
    start date:  2019-01-01 00:00:00
    end date:  2020-01-01 00:00:00
    13/13 [==============================] - 0s 937us/step
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-01-01 00:00:00
    end date:  2019-02-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-02-01 00:00:00
    end date:  2019-03-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-03-01 00:00:00
    end date:  2019-04-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-04-01 00:00:00
    end date:  2019-05-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-05-01 00:00:00
    end date:  2019-06-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-06-01 00:00:00
    end date:  2019-07-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-07-01 00:00:00
    end date:  2019-08-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-08-01 00:00:00
    end date:  2019-09-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-09-01 00:00:00
    end date:  2019-10-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-10-01 00:00:00
    end date:  2019-11-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-11-01 00:00:00
    end date:  2019-12-01 00:00:00
    ----------------------------------------------------------------------------------------------------
    selecting monthly highest and lowest
    start date:  2019-12-01 00:00:00
    end date:  2020-01-01 00:00:00



```python
full_dates = dftemp.loc[(dftemp.DATE>="2010-01-01") & (dftemp.DATE<"2020-01-01")].DATE
full_dates = full_dates.to_numpy()
full_dates = np.unique(full_dates)
print(len(full_dates))
```

    120



```python
high_m = pd.DataFrame(high_mean, full_dates)
high_s = pd.DataFrame(high_stv, full_dates)
low_m = pd.DataFrame(low_mean, full_dates)
low_s = pd.DataFrame(low_stv, full_dates)

high_m.to_csv("long_portfolio_ret.csv")
high_s.to_csv("long_portfolio_stv.csv")
low_m.to_csv("short_portfolio_ret.csv")
low_s.to_csv("short_portfolio_stv.csv")
```


```python
def cumulative_ret(returns):
    plus_one = []
    cum_ret = []
    
    for i in returns:
        plus_one.append(i+1)
    
    for idx, i in enumerate(plus_one):
        if idx == 0:
            cum_ret.append(i)
        else:
            cum_ret.append(cum_ret[idx-1]*i)
    
    return cum_ret
```


```python
cum_long = cumulative_ret(high_mean)
cum_short = cumulative_ret(low_mean)
```


```python
plt.plot(full_dates, cum_long, label = "Long Portfolio")
plt.plot(full_dates, cum_short, label = "Short Portfolio")
plt.legend()
plt.show()
```



<img src="https://ZioFinLab.github.io/images/Portfolio_Construction_NN/output_23_0.png" alt="family_1" style="zoom:100%;" />



```python
dftemp.to_csv("full_dataset.csv")
```


```python
long_port = pd.concat([high_m,high_s], axis =1)
long_port.columns = ["l_mean","l_stv"]

short_port = pd.concat([low_m,low_s], axis =1)
short_port.columns = ["s_mean","s_stv"]

cum_l = pd.DataFrame(cum_long, full_dates)
cum_l.columns = ["l_cum_ret"]
cum_s = pd.DataFrame(cum_short, full_dates)
cum_s.columns = ["s_cum_ret"]
```


```python
fulldeck = pd.concat([long_port,cum_l,short_port,cum_s], axis =1)
fulldeck
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
      <th>l_mean</th>
      <th>l_stv</th>
      <th>l_cum_ret</th>
      <th>s_mean</th>
      <th>s_stv</th>
      <th>s_cum_ret</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-29</th>
      <td>-0.004422</td>
      <td>0.085028</td>
      <td>0.995578</td>
      <td>0.039711</td>
      <td>0.214839</td>
      <td>1.039711</td>
    </tr>
    <tr>
      <th>2010-02-26</th>
      <td>-0.009013</td>
      <td>0.097267</td>
      <td>0.986605</td>
      <td>-0.038077</td>
      <td>0.092174</td>
      <td>1.000122</td>
    </tr>
    <tr>
      <th>2010-03-31</th>
      <td>0.055456</td>
      <td>0.078477</td>
      <td>1.041318</td>
      <td>0.036467</td>
      <td>0.106512</td>
      <td>1.036594</td>
    </tr>
    <tr>
      <th>2010-04-30</th>
      <td>0.073332</td>
      <td>0.098117</td>
      <td>1.117680</td>
      <td>0.153541</td>
      <td>0.231966</td>
      <td>1.195753</td>
    </tr>
    <tr>
      <th>2010-05-28</th>
      <td>-0.119772</td>
      <td>0.084221</td>
      <td>0.983813</td>
      <td>-0.074951</td>
      <td>0.099705</td>
      <td>1.106130</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-08-30</th>
      <td>0.002401</td>
      <td>0.174070</td>
      <td>3.031183</td>
      <td>-0.172382</td>
      <td>0.154210</td>
      <td>1.109405</td>
    </tr>
    <tr>
      <th>2019-09-30</th>
      <td>0.015143</td>
      <td>0.069278</td>
      <td>3.077083</td>
      <td>0.066460</td>
      <td>0.054727</td>
      <td>1.183136</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>-0.007247</td>
      <td>0.124325</td>
      <td>3.054783</td>
      <td>-0.051360</td>
      <td>0.105104</td>
      <td>1.122370</td>
    </tr>
    <tr>
      <th>2019-11-29</th>
      <td>0.030594</td>
      <td>0.164779</td>
      <td>3.148242</td>
      <td>0.092424</td>
      <td>0.181769</td>
      <td>1.226104</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>0.057558</td>
      <td>0.191250</td>
      <td>3.329449</td>
      <td>0.096272</td>
      <td>0.110828</td>
      <td>1.344144</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 6 columns</p>
</div>




```python
fulldeck.to_csv("portfolio_info.csv")
```

