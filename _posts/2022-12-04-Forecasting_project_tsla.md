---
layout: single
title: "Forecasting the stock price of Tesla with cryptocurrencies (Fall 2021)"
categories: [Stock price forecasting, Python, Neural Networks]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
**[Notice]** [Journey to the academic researcher](https://ziofinlab.github.io/biography/bio/)
This is the story of how I became the insightful researcher.
{: .notice--info}

The subject of this report is whether the fluctuation of cryptocurrency can explain the change in the stock price of Tesla. Since Tesla announced that they accept cryptocurrency to buy their products Tesla and holds about 2 billion dollars, some financial analysts have claimed that the effect of the volatility of cryptocurrency is huge for the stock price of Tesla. Furthermore, As the stock price of Tesla has skyrocketed since the pandemic, Elon Musk, CEO of Tesla, has influenced not only the stock price but also the price of cryptocurrency especially the DOGE coin. We will use two methods for forecasting 1) the traditional approach: forecasting the stock price through VAR(Vector auto-regression with whole variables). 2) LASSO approach: forecasting it through VAR with LASSO eliminating the variables that have low coefficients.

## Research purpose
Whether it is possible that the fluctuation of cryptocurrency explains the change of the stock price of Tesla and which approach can forecast the stock price better.

## Methods of analysis
1. Setting up variables 
The following 10 cryptocurrencies are exogenous variables based on the market value.

-	Bitcoin, Ethereum, Binance, Cardano, Tether, Ripple, Solana, Kraken, USD Coin, and Dogecoin

In addition, we also control for the following variables:

i)	Treasury bond yield: as an opportunity cost of the stock price, it has a negative effect on the stock price

ii)	Fed Funds Rate: Higher rates indicate that the economy is heating up (more consumption) and vice-versa.

iii)	Inflation (inflation expectation and 10-year break-even inflation): relating the purchasing power of customers and stimulating the increase of interest rates

iv)	Trade Weighted Dollar Index: TWD can influence the export and import of products along with the fluctuation of stock price

v)	Oil price: A higher oil price may cause people to switch to buy an electric car in which case, the price of Tesla's company would rise.
vi)	Gold and Silver prices: As a safety asset, the prices of both silver and gold can have a negative relationship with the stock price

the independent variable is the stock price of Tesla


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
import warnings
warnings.simplefilter('ignore')
from sklearn import (linear_model, metrics, neural_network, pipeline, preprocessing, model_selection)
import seaborn as sns
sns.set()
import datetime as dt
from statsmodels.tsa.stattools import grangercausalitytests
```


```python
dset = pd.read_csv('TSLA_ds_1022.csv')
dset = dset.drop(0)
dset = dset.set_index("Date")
dset.head()
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
      <th>TSLA</th>
      <th>10CMR</th>
      <th>FFR</th>
      <th>infexp</th>
      <th>TWD</th>
      <th>10YBE</th>
      <th>oil</th>
      <th>XAU</th>
      <th>XAG</th>
      <th>BTC</th>
      <th>...</th>
      <th>ETH_diff</th>
      <th>BNB_diff</th>
      <th>ADA_diff</th>
      <th>Tether_diff</th>
      <th>XRP_diff</th>
      <th>SOL_diff</th>
      <th>pDOTn_diff</th>
      <th>USDC_diff</th>
      <th>DOGE_diff</th>
      <th>const</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-05</th>
      <td>44.69</td>
      <td>2.25</td>
      <td>0.36</td>
      <td>1.81</td>
      <td>114.2649</td>
      <td>1.56</td>
      <td>35.97</td>
      <td>1077.66</td>
      <td>13.98</td>
      <td>431.8</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>43.81</td>
      <td>2.18</td>
      <td>0.36</td>
      <td>1.80</td>
      <td>114.6177</td>
      <td>1.53</td>
      <td>33.97</td>
      <td>1094.45</td>
      <td>14.01</td>
      <td>428.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>43.13</td>
      <td>2.16</td>
      <td>0.36</td>
      <td>1.75</td>
      <td>114.6517</td>
      <td>1.50</td>
      <td>33.29</td>
      <td>1109.25</td>
      <td>14.31</td>
      <td>459.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>42.20</td>
      <td>2.13</td>
      <td>0.36</td>
      <td>1.73</td>
      <td>115.0097</td>
      <td>1.48</td>
      <td>33.20</td>
      <td>1104.24</td>
      <td>13.95</td>
      <td>454.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-01-11</th>
      <td>41.57</td>
      <td>2.17</td>
      <td>0.36</td>
      <td>1.71</td>
      <td>115.0141</td>
      <td>1.45</td>
      <td>31.42</td>
      <td>1094.26</td>
      <td>13.86</td>
      <td>449.3</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24.6883</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



## Analyzing the Granger Causality
In the result of Granger Causality analysis, all cryptocurrency variables have a probability greater than 5%. Therefore, the variables of cryptocurrency do not have a significant effect on the traditional approach.


```python
maxlag=10
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
```


```python
Granger = dset.loc[:, "TSLA_diff":"DOGE_diff"]
grangers_causation_matrix(Granger, variables = Granger.columns)
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
      <th>TSLA_diff_x</th>
      <th>10CMR_diff_x</th>
      <th>FFR_diff_x</th>
      <th>infexp_diff_x</th>
      <th>TWD_diff_x</th>
      <th>10YBE_diff_x</th>
      <th>oil_diff_x</th>
      <th>XAU_diff_x</th>
      <th>XAG_diff_x</th>
      <th>BTC_diff_x</th>
      <th>ETH_diff_x</th>
      <th>BNB_diff_x</th>
      <th>ADA_diff_x</th>
      <th>Tether_diff_x</th>
      <th>XRP_diff_x</th>
      <th>SOL_diff_x</th>
      <th>pDOTn_diff_x</th>
      <th>USDC_diff_x</th>
      <th>DOGE_diff_x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TSLA_diff_y</th>
      <td>1.0000</td>
      <td>0.6571</td>
      <td>0.1963</td>
      <td>0.4127</td>
      <td>0.2082</td>
      <td>0.1380</td>
      <td>0.8005</td>
      <td>0.0007</td>
      <td>0.0096</td>
      <td>0.6192</td>
      <td>0.4485</td>
      <td>0.9632</td>
      <td>0.5514</td>
      <td>0.9211</td>
      <td>0.3492</td>
      <td>0.8981</td>
      <td>0.7859</td>
      <td>0.8515</td>
      <td>0.8469</td>
    </tr>
    <tr>
      <th>10CMR_diff_y</th>
      <td>0.0091</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0010</td>
      <td>0.0000</td>
      <td>0.0063</td>
      <td>0.0000</td>
      <td>0.0001</td>
      <td>0.0000</td>
      <td>0.0002</td>
      <td>0.2052</td>
      <td>0.0001</td>
      <td>0.6290</td>
      <td>0.0214</td>
      <td>0.1545</td>
      <td>0.0431</td>
      <td>0.7412</td>
      <td>0.4359</td>
    </tr>
    <tr>
      <th>FFR_diff_y</th>
      <td>0.0879</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0005</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4487</td>
      <td>0.0000</td>
      <td>0.9813</td>
      <td>0.0001</td>
      <td>0.5253</td>
      <td>0.8308</td>
      <td>0.8973</td>
      <td>0.8941</td>
    </tr>
    <tr>
      <th>infexp_diff_y</th>
      <td>0.4964</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0002</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0067</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.3346</td>
      <td>0.0000</td>
      <td>0.0065</td>
      <td>0.1274</td>
      <td>0.3326</td>
      <td>0.5260</td>
      <td>0.4813</td>
      <td>0.6607</td>
    </tr>
    <tr>
      <th>TWD_diff_y</th>
      <td>0.4741</td>
      <td>0.0009</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.2020</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0007</td>
      <td>0.0000</td>
      <td>0.6265</td>
      <td>0.0104</td>
      <td>0.3737</td>
      <td>0.3939</td>
      <td>0.6338</td>
      <td>0.3210</td>
      <td>0.2664</td>
      <td>0.8202</td>
    </tr>
    <tr>
      <th>10YBE_diff_y</th>
      <td>0.7924</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0154</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.7873</td>
      <td>0.0000</td>
      <td>0.8048</td>
      <td>0.0012</td>
      <td>0.4042</td>
      <td>0.3702</td>
      <td>0.5009</td>
      <td>0.8122</td>
    </tr>
    <tr>
      <th>oil_diff_y</th>
      <td>0.6599</td>
      <td>0.0000</td>
      <td>0.0107</td>
      <td>0.0000</td>
      <td>0.0259</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0001</td>
      <td>0.2883</td>
      <td>0.4544</td>
      <td>0.3266</td>
      <td>0.6329</td>
      <td>0.7146</td>
      <td>0.9673</td>
      <td>0.5308</td>
      <td>0.9539</td>
      <td>0.7794</td>
      <td>0.8809</td>
      <td>0.9600</td>
    </tr>
    <tr>
      <th>XAU_diff_y</th>
      <td>0.0261</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0390</td>
      <td>0.5379</td>
      <td>0.0000</td>
      <td>0.0409</td>
      <td>1.0000</td>
      <td>0.0050</td>
      <td>0.0009</td>
      <td>0.0118</td>
      <td>0.1037</td>
      <td>0.3502</td>
      <td>0.5957</td>
      <td>0.5645</td>
      <td>0.4666</td>
      <td>0.7449</td>
      <td>0.4006</td>
      <td>0.8543</td>
    </tr>
    <tr>
      <th>XAG_diff_y</th>
      <td>0.3175</td>
      <td>0.0000</td>
      <td>0.0082</td>
      <td>0.0337</td>
      <td>0.1683</td>
      <td>0.0000</td>
      <td>0.5411</td>
      <td>0.0126</td>
      <td>1.0000</td>
      <td>0.0051</td>
      <td>0.0107</td>
      <td>0.5948</td>
      <td>0.0260</td>
      <td>0.4045</td>
      <td>0.4021</td>
      <td>0.0001</td>
      <td>0.3679</td>
      <td>0.6546</td>
      <td>0.4319</td>
    </tr>
    <tr>
      <th>BTC_diff_y</th>
      <td>0.7210</td>
      <td>0.0136</td>
      <td>0.0491</td>
      <td>0.3187</td>
      <td>0.0040</td>
      <td>0.0734</td>
      <td>0.4174</td>
      <td>0.0213</td>
      <td>0.1035</td>
      <td>1.0000</td>
      <td>0.0116</td>
      <td>0.2759</td>
      <td>0.0452</td>
      <td>0.6021</td>
      <td>0.0607</td>
      <td>0.1768</td>
      <td>0.5764</td>
      <td>0.4664</td>
      <td>0.7431</td>
    </tr>
    <tr>
      <th>ETH_diff_y</th>
      <td>0.0000</td>
      <td>0.0363</td>
      <td>0.0885</td>
      <td>0.1287</td>
      <td>0.0091</td>
      <td>0.0976</td>
      <td>0.3597</td>
      <td>0.4238</td>
      <td>0.6921</td>
      <td>0.0062</td>
      <td>1.0000</td>
      <td>0.3661</td>
      <td>0.1468</td>
      <td>0.3435</td>
      <td>0.0898</td>
      <td>0.7666</td>
      <td>0.4659</td>
      <td>0.7766</td>
      <td>0.6625</td>
    </tr>
    <tr>
      <th>BNB_diff_y</th>
      <td>0.8345</td>
      <td>0.7036</td>
      <td>0.3885</td>
      <td>0.4279</td>
      <td>0.7411</td>
      <td>0.6453</td>
      <td>0.9371</td>
      <td>0.6083</td>
      <td>0.8366</td>
      <td>0.3796</td>
      <td>0.5876</td>
      <td>1.0000</td>
      <td>0.1206</td>
      <td>0.6467</td>
      <td>0.0745</td>
      <td>0.0412</td>
      <td>0.4296</td>
      <td>0.3453</td>
      <td>0.6194</td>
    </tr>
    <tr>
      <th>ADA_diff_y</th>
      <td>0.7034</td>
      <td>0.0912</td>
      <td>0.3040</td>
      <td>0.0432</td>
      <td>0.5556</td>
      <td>0.0284</td>
      <td>0.3320</td>
      <td>0.0443</td>
      <td>0.4985</td>
      <td>0.0004</td>
      <td>0.0477</td>
      <td>0.0245</td>
      <td>1.0000</td>
      <td>0.5514</td>
      <td>0.0004</td>
      <td>0.0650</td>
      <td>0.1236</td>
      <td>0.4002</td>
      <td>0.2660</td>
    </tr>
    <tr>
      <th>Tether_diff_y</th>
      <td>0.9104</td>
      <td>0.8093</td>
      <td>0.9815</td>
      <td>0.7463</td>
      <td>0.0559</td>
      <td>0.6725</td>
      <td>0.9786</td>
      <td>0.2917</td>
      <td>0.5475</td>
      <td>0.0700</td>
      <td>0.1434</td>
      <td>0.0008</td>
      <td>0.0003</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.9261</td>
      <td>0.9519</td>
      <td>0.7658</td>
      <td>0.9077</td>
    </tr>
    <tr>
      <th>XRP_diff_y</th>
      <td>0.6536</td>
      <td>0.5104</td>
      <td>0.2915</td>
      <td>0.2571</td>
      <td>0.3972</td>
      <td>0.2159</td>
      <td>0.8442</td>
      <td>0.2406</td>
      <td>0.6087</td>
      <td>0.0201</td>
      <td>0.0903</td>
      <td>0.0270</td>
      <td>0.0012</td>
      <td>0.1630</td>
      <td>1.0000</td>
      <td>0.5952</td>
      <td>0.2933</td>
      <td>0.6438</td>
      <td>0.5779</td>
    </tr>
    <tr>
      <th>SOL_diff_y</th>
      <td>0.7142</td>
      <td>0.0926</td>
      <td>0.4007</td>
      <td>0.3217</td>
      <td>0.7922</td>
      <td>0.4763</td>
      <td>0.9589</td>
      <td>0.1414</td>
      <td>0.0322</td>
      <td>0.3276</td>
      <td>0.2857</td>
      <td>0.3227</td>
      <td>0.0004</td>
      <td>0.8986</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>0.9475</td>
      <td>0.9911</td>
    </tr>
    <tr>
      <th>pDOTn_diff_y</th>
      <td>0.8101</td>
      <td>0.1768</td>
      <td>0.4263</td>
      <td>0.8178</td>
      <td>0.1299</td>
      <td>0.5545</td>
      <td>0.9439</td>
      <td>0.0013</td>
      <td>0.0000</td>
      <td>0.2988</td>
      <td>0.1380</td>
      <td>0.7665</td>
      <td>0.3767</td>
      <td>0.9649</td>
      <td>0.0016</td>
      <td>0.0024</td>
      <td>1.0000</td>
      <td>0.9831</td>
      <td>0.9593</td>
    </tr>
    <tr>
      <th>USDC_diff_y</th>
      <td>0.8319</td>
      <td>0.4363</td>
      <td>0.9091</td>
      <td>0.5988</td>
      <td>0.2546</td>
      <td>0.5857</td>
      <td>0.9543</td>
      <td>0.4587</td>
      <td>0.5843</td>
      <td>0.0244</td>
      <td>0.2662</td>
      <td>0.1003</td>
      <td>0.0029</td>
      <td>0.8840</td>
      <td>0.6250</td>
      <td>0.9224</td>
      <td>0.9979</td>
      <td>1.0000</td>
      <td>0.9930</td>
    </tr>
    <tr>
      <th>DOGE_diff_y</th>
      <td>0.7456</td>
      <td>0.6685</td>
      <td>0.9377</td>
      <td>0.7775</td>
      <td>0.8012</td>
      <td>0.6029</td>
      <td>0.9161</td>
      <td>0.5163</td>
      <td>0.5112</td>
      <td>0.3599</td>
      <td>0.2325</td>
      <td>0.4398</td>
      <td>0.3318</td>
      <td>0.9068</td>
      <td>0.5919</td>
      <td>0.9506</td>
      <td>0.9652</td>
      <td>0.9761</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



## Confirm training period and test period
We created three time periods of the training set. The first set starts with the beginning of 2016 when cryptocurrencies became famous. The second set starts with the beginning of 2017 when cryptocurrency skyrocketed the first time. The third set starts with the beginning of 2020 when both Tesla and cryptocurrencies skyrocketed.
Because the third set only represents that LASSO does not eliminate the coefficient of cryptocurrencies and we realized that the current relationship between the stock price of Tesla and the fluctuations of cryptocurrencies is stronger than before, we decide to choose the Third set as a training set. Furthermore, we confirmed that the test period from Jul-01-2021 to Oct-18-2021.


```python
TSLA = dset["TSLA_diff"]
Exogen = dset.loc[:, "10CMR_diff":"DOGE_diff"]
TSLA_OLS = sm.OLS(TSLA, Exogen)
result_OLS = TSLA_OLS.fit()

names = dset.columns.values[1:19]
plt.figure(figsize=(15,5))
plt.bar(names,result_OLS.params[0:])
plt.title('Relationship 2016-01-01 ~ 2021-10-19')
plt.xlabel('Intercept and Coefficients')
plt.ylabel('OLS Estimates');
print(result_OLS.params[0:])
```

    10CMR_diff     0.014131
    FFR_diff       0.092686
    infexp_diff    0.087457
    TWD_diff      -1.309505
    10YBE_diff     0.015982
    oil_diff      -0.011928
    XAU_diff      -1.087257
    XAG_diff       0.463873
    BTC_diff       0.020326
    ETH_diff       0.017720
    BNB_diff      -0.004339
    ADA_diff       0.013301
    Tether_diff   -0.002773
    XRP_diff      -0.020046
    SOL_diff       0.026030
    pDOTn_diff     0.036460
    USDC_diff      0.026032
    DOGE_diff      0.000118
    dtype: float64

<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_9_1.png" alt="family_1" style="zoom:100%;" />

```python
TSLA_EN = LassoCV(cv=10, random_state=1)
TSLA_EN.fit(Exogen, TSLA)

plt.figure(figsize=(15,5))
plt.bar(names, TSLA_EN.coef_)
plt.title('Relationship 2016-01-01 ~ 2021-10-19')
plt.xlabel('Intercept and Coefficients')
plt.ylabel('Lasso Estimates');
print(TSLA_EN.coef_)
```

    [ 0.          0.05427374  0.         -0.          0.         -0.
     -0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.        ]

<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_10_1.png" alt="family_1" style="zoom:100%;" />

```python
TSLA = dset.loc["2017-01-01": ,"TSLA_diff"]
Exogen = dset.loc["2017-01-01": , "10CMR_diff":"DOGE_diff"]
TSLA_OLS = sm.OLS(TSLA, Exogen)
result_OLS = TSLA_OLS.fit()

names = dset.columns.values[1:19]
plt.figure(figsize=(15,5))
plt.bar(names,result_OLS.params[0:])
plt.title('Relationship 2017-01-01 ~ 2021-10-19')
plt.xlabel('Intercept and Coefficients')
plt.ylabel('OLS Estimates');
print(result_OLS.params[0:])
```

    10CMR_diff     0.080909
    FFR_diff       0.018976
    infexp_diff   -0.210880
    TWD_diff      -1.256933
    10YBE_diff     0.298315
    oil_diff      -0.006214
    XAU_diff      -0.544770
    XAG_diff       0.468780
    BTC_diff       0.009931
    ETH_diff       0.021716
    BNB_diff      -0.005463
    ADA_diff      -0.009273
    Tether_diff    0.006178
    XRP_diff       0.017648
    SOL_diff       0.021012
    pDOTn_diff     0.026336
    USDC_diff      0.027537
    DOGE_diff      0.000150
    dtype: float64

<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_11_1.png" alt="family_1" style="zoom:100%;" />



```python
TSLA_EN = LassoCV(cv=10, random_state=1)
TSLA_EN.fit(Exogen, TSLA)

plt.figure(figsize=(15,5))
plt.bar(names, TSLA_EN.coef_)
plt.title('Relationship 2017-01-01 ~ 2021-10-19')
plt.xlabel('Intercept and Coefficients')
plt.ylabel('Lasso Estimates');
print(TSLA_EN.coef_)
```

    [ 0.04315125  0.03434962  0.         -0.          0.06338159 -0.00461966
      0.          0.05125903  0.          0.02463747 -0.          0.
      0.          0.0127104   0.02291635  0.00664378  0.          0.00013601]

<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_12_1.png" alt="family_1" style="zoom:100%;" />



```python
TSLA = dset.loc["2020-01-02": ,"TSLA_diff"]
Exogen = dset.loc["2020-01-02": , "10CMR_diff":"DOGE_diff"]
TSLA_OLS = sm.OLS(TSLA, Exogen)
result_OLS = TSLA_OLS.fit()

names = dset.columns.values[1:19]
plt.figure(figsize=(15,5))
plt.bar(names,result_OLS.params[0:])
plt.title('Relationship 2020-01-02 ~ 2021-10-19')
plt.xlabel('Intercept and Coefficients')
plt.ylabel('OLS Estimates');
print(result_OLS.params[0:])
```

    10CMR_diff     0.068849
    FFR_diff       0.017995
    infexp_diff   -0.398306
    TWD_diff      -2.191456
    10YBE_diff     0.380097
    oil_diff      -0.006267
    XAU_diff      -0.312150
    XAG_diff       0.398949
    BTC_diff       0.071159
    ETH_diff       0.036119
    BNB_diff      -0.023011
    ADA_diff      -0.021280
    Tether_diff    3.290954
    XRP_diff       0.046905
    SOL_diff       0.012452
    pDOTn_diff     0.000428
    USDC_diff      2.495569
    DOGE_diff      0.000170
    dtype: float64

<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_13_1.png" alt="family_1" style="zoom:100%;" />



```python
TSLA_EN = LassoCV(cv=10, random_state=1)
TSLA_EN.fit(Exogen, TSLA)

plt.figure(figsize=(15,5))
plt.bar(names, TSLA_EN.coef_)
plt.title('Relationship 2020-01-02 ~ 2021-10-19')
plt.xlabel('Intercept and Coefficients')
plt.ylabel('Lasso Estimates');
print(TSLA_EN.coef_)
```

    [ 0.00260532  0.04136176  0.         -0.          0.00862294 -0.00686096
      0.          0.          0.00785282  0.07127208 -0.          0.
      0.          0.03075644  0.01217944  0.         -0.          0.00015222]




<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_14_1.png" alt="family_1" style="zoom:100%;" />



```python
"""alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
for a in alphas:
    model = ElasticNet(alpha=a).fit(Exogen,TSLA)   
    score = model.score(Exogen,TSLA)
    pred_y = model.predict(Exogen)
    mse = mean_squared_error(TSLA, pred_y)
    print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
       .format(a, score, mse, np.sqrt(mse))) """
```




    'alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]\nfor a in alphas:\n    model = ElasticNet(alpha=a).fit(Exogen,TSLA)   \n    score = model.score(Exogen,TSLA)\n    pred_y = model.predict(Exogen)\n    mse = mean_squared_error(TSLA, pred_y)\n    print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"\n       .format(a, score, mse, np.sqrt(mse))) '




```python
from statsmodels.tsa.vector_ar.var_model import VAR
```


```python
dset1 = pd.read_csv('TSLA_ds_1022.csv')
dset1 = dset1.drop(0)
dset1 = dset1.set_index("Date")
names1 = ["TSLA_diff", "10CMR_diff", "FFR_diff", "infexp_diff", "TWD_diff", "10YBE_diff",\
          "oil_diff", "XAU_diff", "XAG_diff", "BTC_diff", "ETH_diff", "BNB_diff",\
          "ADA_diff", "Tether_diff", "XRP_diff", "SOL_diff", "pDOTn_diff", "USDC_diff", "DOGE_diff"]
names2 = ["10CMR_diff", "FFR_diff", "infexp_diff", "TWD_diff", "10YBE_diff",\
          "oil_diff", "XAU_diff", "XAG_diff", "BTC_diff", "ETH_diff", "BNB_diff",\
          "ADA_diff", "Tether_diff", "XRP_diff", "SOL_diff", "pDOTn_diff", "USDC_diff", "DOGE_diff"]

dset2 = dset1.loc[:,names1]
#data_t = dset2.loc["2016-01-04":"2021-06-21", :]

#col_mask=dset2.isnull().any(axis=0) 
#row_mask=dset2.isnull().any(axis=1)
#print(dset2.loc[row_mask,col_mask])

#dset2.fillna(dset2.mean(), inplace=True)
#dset2 = dset2.fillna(dset2.mean())
#dset2._is_view
```


```python
start_date = "2021-06-21"
end_date = "2021-10-19"

dset2.index.names = ["Date"]
dset2.index = pd.to_datetime(dset.index)
dset2.to_period("D")

data_train = dset2.loc["2016-01-05":"2021-06-21", :]
var_train = VAR(data_train)
results = var_train.fit(25)
lag_order = results.k_ar
forecasted = pd.DataFrame(results.forecast(data_train.values[-lag_order:], 120)) # Forecast 120 months


# Rename forecasted columns
forecasted_names = list(forecasted.columns.values)
data_train_names = list(data_train.columns.values)

var_dict = dict(zip(forecasted_names, data_train_names))
for f,t in var_dict.items():
    forecasted = forecasted.rename(columns={f:t + "_fcast"})
    
forecasted.index= pd.DatetimeIndex(pd.date_range(start_date, periods=forecasted.shape[0]))
forecasted.index.names = ["Date"]

# Parse together forecasted data with original dataset
final_data = pd.merge(forecasted, dset2, left_index=True, right_index=True)
final_data = final_data.sort_index(axis=0, ascending=True)
final_data = pd.concat([data_train, final_data], sort=True, axis=0)
final_data = final_data.sort_index(axis=0, ascending=True)

TSLA_fs = final_data.loc["2021-06-22":"2021-10-18","TSLA_diff_fcast"]
TSLA_r = final_data.loc["2021-06-22":"2021-10-18","TSLA_diff"]

print(TSLA_fs)
var_mse1 = metrics.mean_squared_error(TSLA_fs, TSLA_r)
```

    Date
    2021-06-22    0.101006
    2021-06-23    5.433423
    2021-06-24   -7.929520
    2021-06-25    0.569414
    2021-06-28    2.115058
                    ...   
    2021-10-12   -0.075105
    2021-10-13    0.118774
    2021-10-14    0.335843
    2021-10-15   -0.108588
    2021-10-18    0.121682
    Name: TSLA_diff_fcast, Length: 83, dtype: float64



```python
print(f"The mean squared error between the forecasted and actual values is {var_mse1}")
```

    The mean squared error between the forecasted and actual values is 17.311018428934627


## Results of forecasting
The MSE(Mean squared error) of forecasting the stock price through VAR(Vector auto -regression) with the traditional model is 17.3110.
The MSE of forecasting the stock price through VAR with the LASSO model is 10.6772.


```python
fig, ax = plt.subplots(figsize=(14,6))
colors = sns.color_palette("deep", 8)

TSLA_rplot = final_data.loc["2021-01-02":"2021-10-18","TSLA_diff"]

TSLA_fs.plot(ax=ax, legend=True, linewidth=2.5, linestyle="dashed")
TSLA_rplot.plot(ax=ax, legend=True, alpha=0.6, linestyle="solid")

ax.set_title("VAR in-sample forecast, traditional approach", fontsize=16, fontweight="bold", fontname="Verdana", loc="left")
ax.set_ylabel("First differences", fontname="Verdana")
ax.legend([f"VAR Forecast, MSE={var_mse1}", "TSLA Real Fluctuations"])
```




    <matplotlib.legend.Legend at 0x272c5e19550>

<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_21_1.png" alt="family_1" style="zoom:100%;" />



```python
def train_test_plot(model, X_train, X_test):
    """
    This will plot the actual values of CPI against the one fitted by the model
    We train the model until 2009 and then use it from 2009 onwards on the test features dataset
    """
    fig, ax = plt.subplots(figsize=(12,4))
    colors = sns.color_palette("deep", 8)
    
    yvalues = pd.DataFrame(y_test)
    
    forecasted = list(model.predict(X_test)) # Use the model fit on features data from 2009 onwards
    df_fcast = pd.DataFrame({"date": list(yvalues.index), "TSLA_fcast": forecasted})
    df_fcast = df_fcast.set_index("date")
    
    df = pd.merge(yvalues, df_fcast, left_index=True, right_index=True)

    df["TSLA_fcast"].plot(ax=ax, legend=True, linewidth=2.5, linestyle="dashed", color="forestgreen") # TSLA fitted
    df["TSLA_diff"].plot(ax=ax, legend=True, linewidth=1.5, linestyle="solid", color="salmon") # Actual TSLA values
    
    ax.set_title("TSLA vs. Model's TSLA")
    ax.set_ylabel("First differences")
    ax.legend(["Fitted TSLA","Actual TSLA"])
```


```python
x_train = dset2.loc["2020-01-05":"2021-06-21", names2]
x_test = dset2.loc["2021-06-21":,names2]
y_train = dset2.loc["2020-01-05":"2021-06-21", "TSLA_diff"]
y_test = dset2.loc["2021-06-21":, "TSLA_diff"]

lasso = linear_model.LassoCV(cv=model_selection.TimeSeriesSplit(n_splits=5), 
                             alphas=None, tol = 10000, normalize=True) 

fred_lasso = lasso.fit(x_train, y_train)
optimal_alpha = fred_lasso.alpha_

lasso2 = linear_model.Lasso(alpha=optimal_alpha, normalize=True)
lasso2.fit(x_train, y_train)

lasso2.coef_
#train_test_plot(dset_EN, x_train, x_test) 
```




    array([ 7.20902440e-02,  2.00701734e-02, -4.02847424e-01, -2.21122815e+00,
            3.77903319e-01, -6.91098494e-03, -3.34285652e-01,  4.24181205e-01,
            8.54914242e-02,  3.52091625e-02, -2.50758876e-02, -3.56472136e-02,
            3.52150559e+00,  6.06896275e-02,  8.61617708e-03,  7.33200359e-03,
            3.14226828e+00,  2.28882758e-04])




```python
fig, ax = plt.subplots(figsize=(12,4))
colors = sns.color_palette("deep", 8)

yvalues = pd.DataFrame(y_test)
    
forecasted = list(lasso2.predict(x_test)) # Use the model fit on features data from 2009 onwards
df_fcast = pd.DataFrame({"date": list(yvalues.index), "TSLA_fcast": forecasted})
df_fcast = df_fcast.set_index("date")
    
df = pd.merge(yvalues, df_fcast, left_index=True, right_index=True)


df["TSLA_fcast"].plot(ax=ax, legend=True, linewidth=2.5, linestyle="dashed", color="forestgreen") # TSLA fitted
df["TSLA_diff"].plot(ax=ax, legend=True, linewidth=1.5, linestyle="solid", color="salmon") # Actual TSLA values
    
ax.set_title("TSLA vs. Model's TSLA")
ax.set_ylabel("First differences")
ax.legend(["Fitted TSLA","Actual TSLA"])
```




    <matplotlib.legend.Legend at 0x272c414f5e0>




<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_24_1.png" alt="family_1" style="zoom:100%;" />



```python
metrics.mean_squared_error(y_test, dset_EN.predict(x_test))
```




    3.8741380753790273




```python
lasso_coefs = pd.DataFrame({"features":list(x_train), "coef": lasso2.coef_})
lasso_coefs = lasso_coefs[lasso_coefs.coef != 0.0]
lasso_coefs.sort_values("coef", ascending=False)
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
      <th>features</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Tether_diff</td>
      <td>3.521506</td>
    </tr>
    <tr>
      <th>16</th>
      <td>USDC_diff</td>
      <td>3.142268</td>
    </tr>
    <tr>
      <th>7</th>
      <td>XAG_diff</td>
      <td>0.424181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10YBE_diff</td>
      <td>0.377903</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BTC_diff</td>
      <td>0.085491</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10CMR_diff</td>
      <td>0.072090</td>
    </tr>
    <tr>
      <th>13</th>
      <td>XRP_diff</td>
      <td>0.060690</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ETH_diff</td>
      <td>0.035209</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FFR_diff</td>
      <td>0.020070</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SOL_diff</td>
      <td>0.008616</td>
    </tr>
    <tr>
      <th>15</th>
      <td>pDOTn_diff</td>
      <td>0.007332</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DOGE_diff</td>
      <td>0.000229</td>
    </tr>
    <tr>
      <th>5</th>
      <td>oil_diff</td>
      <td>-0.006911</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BNB_diff</td>
      <td>-0.025076</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ADA_diff</td>
      <td>-0.035647</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XAU_diff</td>
      <td>-0.334286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>infexp_diff</td>
      <td>-0.402847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TWD_diff</td>
      <td>-2.211228</td>
    </tr>
  </tbody>
</table>
</div>




```python
names3 = ["TSLA_diff","FFR_diff", "ETH_diff", "XRP_diff",\
          "10YBE_diff", "BTC_diff", "SOL_diff"]
dset3 = dset1.loc[:,names3]
#mse2, df2 = var_create(columns=names, data=dset)
#print(f"The mean squared error between the forecasted and actual values is {mse2}")
```


```python
start_date = "2021-06-21"
end_date = "2021-10-19"

dset3.index.names = ["Date"]
dset3.index = pd.to_datetime(dset.index)
dset3.to_period("D")

data_train = dset3.loc["2020-01-02":"2021-06-21", :]
var_train = VAR(data_train)
results = var_train.fit(25)
lag_order = results.k_ar
forecasted = pd.DataFrame(results.forecast(data_train.values[-lag_order:], 120)) # Forecast 120 months

# Rename forecasted columns
forecasted_names = list(forecasted.columns.values)
data_train_names = list(data_train.columns.values)

var_dict = dict(zip(forecasted_names, data_train_names))
for f,t in var_dict.items():
    forecasted = forecasted.rename(columns={f:t + "_fcast"})
    
forecasted.index= pd.DatetimeIndex(pd.date_range(start_date, periods=forecasted.shape[0]))
forecasted.index.names = ["Date"]

# Parse together forecasted data with original dataset
final_data = pd.merge(forecasted, dset3, left_index=True, right_index=True)
final_data = final_data.sort_index(axis=0, ascending=True)
final_data = pd.concat([data_train, final_data], sort=True, axis=0)
final_data = final_data.sort_index(axis=0, ascending=True)

TSLA_fs = final_data.loc["2021-06-22":"2021-10-18","TSLA_diff_fcast"]
TSLA_r = final_data.loc["2021-06-22":"2021-10-18","TSLA_diff"]

print(TSLA_fs)
var_mse2 = metrics.mean_squared_error(TSLA_fs, TSLA_r)
```

    Date
    2021-06-22     9.055724
    2021-06-23    12.795725
    2021-06-24     0.255226
    2021-06-25    -4.068157
    2021-06-28    -1.563721
                    ...    
    2021-10-12    -0.436792
    2021-10-13     0.067200
    2021-10-14    -0.005382
    2021-10-15     0.128331
    2021-10-18    -0.751468
    Name: TSLA_diff_fcast, Length: 83, dtype: float64



```python
fig, ax = plt.subplots(figsize=(14,6))
colors = sns.color_palette("deep", 8)

TSLA_rplot = final_data.loc["2021-01-02":"2021-10-18","TSLA_diff"]

TSLA_fs.plot(ax=ax, legend=True, linewidth=2.5, linestyle="dashed")
TSLA_rplot.plot(ax=ax, legend=True, alpha=0.6, linestyle="solid")

ax.set_title("VAR in-sample forecast, LASSO approach", fontsize=16, fontweight="bold", fontname="Verdana", loc="left")
ax.set_ylabel("First differences", fontname="Verdana")
ax.legend([f"VAR Forecast, MSE={var_mse2}", "TSLA Real Fluctuations"])
```




    <matplotlib.legend.Legend at 0x272c71231f0>


<img src="https://ZioFinLab.github.io/images/2022-12-04-Forecasting_project_tsla/output_29_1.png" alt="family_1" style="zoom:100%;" />
    


## Result and inference
With the traditional approach, we can not find any significant coefficient relationship between the stock price of Tesla and the price of cryptocurrencies. However, with the LASSO, we figure out that even the coefficients of cryptocurrencies do not have statistical significance, the fluctuations of cryptocurrencies can support to explain more the change of stock price of Tesla because the MSE with the LASSO is lower than that with the traditional approach.
Furthermore, we can confirm that even though the change of price of cryptocurrencies did not affect the fluctuations of the stock price of Tesla, nowadays the effect has become larger than before.
