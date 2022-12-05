---
layout: single
title: "Binomial Asset Pricing Model Based on Goldman Sachs theory"
categories: [Binomial Asset Pricing Model, Python, Option pricing, Goldman Sachs]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---
**[Notice]** [Journey to the academic researcher](https://ziofinlab.github.io/biography/bio/)
This is the story of how I became the insightful researcher.
{: .notice--info}

Implementation of the Goldman Sachs' option pricing model in python. I treated binomial tree as a network with nodes (i,j) with i representing the time steps and j representing the number of ordered price outcome (lowest - or bottom of tree - to highest). 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.stats import bernoulli
from scipy.optimize import fsolve
from scipy.stats import norm
from tabulate import tabulate
%matplotlib inline
```

# Binomial Tree Representation

Stock tree can be represented using nodes (i,j) and intial stock price S0

S_{i,j} = S_0u^{j}d^{i-j}

C_{i,j} represents contract price at each node (i,j). Where C_{N,j} represents final payoff function that we can define.

# American Option Characteristics

For an american put option: if T = t_N then at the terminal nodes, C^{j}_N = (K-S^{j}_N)^{+}

For all other parts of the tree at nodes (i,j):

- Max of exercise value or continuation/hold value

- C^{j}_i = max[ (K-S^{j}_i)^{+} , exp^{-r\Delta t} { q^{j}_i C^{j+1}_{i+1} + (1 - q^{j}_i)C^{j-1}_{i-1}} ]


```python
# Initialise parameters
S0 = 10000      # initial stock price
K = 10000       # strike price
T = 3         # time to maturity in years
N = 3         # number of time steps
sigma = 0.4  # volatility

dt = T/N
u = np.exp(sigma * np.sqrt(dt))       # up-factor in binomial models
d = 1/u       # ensure recombining tree

opttype = 'C' # Option Type 'C' or 'P'
rates = pd.read_csv('rates_ex.csv')
put_v = pd.read_csv('put_v_ex.csv')
call_v = pd.read_csv('call_v_ex.csv')
coupon = 200 # 액면이자
```

# Stock Tree


```python
#  Create some empty matrices to hold our stock and call prices.
stock_prices = np.zeros( (N+1, N+1) )

#  Put our initial price in the matrix
stock_prices[0,0] = S0

#  Fill out the remaining values
for i in range(1, N+1 ):
    M = i + 1
    stock_prices[0, i] = u * stock_prices[0, i-1]
    for j in range(1, M ):
        stock_prices[j, i] = d * stock_prices[j - 1, i - 1]
 
plt.spy(stock_prices)
```




    <matplotlib.image.AxesImage at 0x2497f67ec70>




![output_6_1](../images/2022-12-04-Goldman_Basic/output_6_1.png)



```python
print(tabulate(stock_prices.round(0)))
#DF = pd.DataFrame(stock_prices)
#DF.to_csv("stock_prices_ex.csv")
```

    -----  -----  -----  -----
    10000  14918  22255  33201
        0   6703  10000  14918
        0      0   4493   6703
        0      0      0   3012
    -----  -----  -----  -----


# Strike Prices Tree


```python
#  Create some empty matrices to hold our stock and call prices.
K_prices = np.zeros( (N+1, N+1) )

#  Put our initial price in the matrix
K_prices[0,0] = K

#  Fill out the remaining values
for i in range(1, N+1 ):
    M = i + 1
    if i < 4: # 한 달 동안은 리픽싱을 적용하지 않음
        K_prices[0, i] = K_prices[0, i-1]
        for j in range(1, M ):
            K_prices[j, i] = K_prices[j-1, i-1]
            
    else:  # 한 달 이후에는 모두 적용됨
        if stock_prices[0, i] <= K:
            K_prices[0, i] = max(stock_prices[0, i], K*70/100)
            for j in range(1, M ):
                if stock_prices[j, i] <= K:
                    K_prices[j, i] = max(stock_prices[j, i], K*70/100)
                else:
                    K_prices[j, i] = K_prices[j-1, i-1]
        else:
            K_prices[0, i] = K_prices[0, i-1]
            for j in range(1, M ):
                if stock_prices[j, i] <= K:
                    K_prices[j, i] = max(stock_prices[j, i], K*70/100)
                else:
                    K_prices[j, i] = K_prices[j-1, i-1]

plt.spy(K_prices)
```




    <matplotlib.image.AxesImage at 0x2497f6ea6d0>




![output_9_1](../images/2022-12-04-Goldman_Basic/output_9_1.png)



```python
print(tabulate(K_prices.round(0)))
#DF = pd.DataFrame(K_prices)
#DF.to_csv("K_prices.csv")
```

    -----  -----  -----  -----
    10000  10000  10000  10000
        0  10000  10000  10000
        0      0  10000  10000
        0      0      0  10000
    -----  -----  -----  -----


# Put bond price


```python
#  Create some empty matrices to hold our stock and call prices.
p_prices = np.zeros( (N+1, N+1) )
put_ar = put_v.iloc[0,1:].to_numpy()
p_interval = 0
print(put_ar)

#  Fill out the remaining values
for i in range(1, N+1):
    M = i + 1
    if i >= 2: # 상환권 행사 가능 이후에만 고려함
        #if i == 2: # 상환권 행사할 수 있는 최초 시기
        p_prices[0,i] = put_ar[i] # 다시 바꾸면 put_ar[i]를 put_ar[p_interval]로
        #p_interval += 1 예제에서는 그냥 바로 적용하니까 안씀
        for j in range(1, M ):
            p_prices[j, i] = p_prices[j-1 , i]
        """elif i % 13 == 1:  # 상환권 행사가 13주 마다 가능함
            p_prices[0,i] = put_ar[p_interval]
            p_interval += 1
            for j in range(1, M ):
                p_prices[j, i] = p_prices[j-1 , i]"""
        
plt.spy(p_prices)
```

    [0 0 10800 11100]





    <matplotlib.image.AxesImage at 0x2497f8fc520>

![output_12_2](../images/2022-12-04-Goldman_Basic/output_12_2.png)



```python
print(tabulate(p_prices.round(0)))
#DF = pd.DataFrame(p_prices)
#DF.to_csv("p_prices.csv")
```

    -  -  -----  -----
    0  0  10800  11100
    0  0  10800  11100
    0  0  10800  11100
    0  0      0  11100
    -  -  -----  -----


# Call bond price


```python
#  Create some empty matrices to hold our stock and call prices.
c_prices = np.zeros( (N+1, N+1) )
call_ar = call_v.iloc[0,1:].to_numpy()
c_interval = 0
print(call_ar)

#  Fill out the remaining values
for i in range(1, N+1):
    M = i + 1
    if i >= 2: # 상환권 행사 가능 이후에만 고려함
        #if i == 2: # 상환권 행사할 수 있는 최초 시기
        c_prices[0,i] = call_ar[i] # 다시 바꾸면 put_ar[i]를 put_ar[p_interval]로
        #p_interval += 1 예제에서는 그냥 바로 적용하니까 안씀
        for j in range(1, M ):
            c_prices[j, i] = c_prices[j-1 , i]
        """elif i % 13 == 1:  # 상환권 행사가 13주 마다 가능함
            p_prices[0,i] = put_ar[p_interval]
            p_interval += 1
            for j in range(1, M ):
                p_prices[j, i] = p_prices[j-1 , i]"""
        
plt.spy(c_prices)
```

    [0 0 10800 0]





    <matplotlib.image.AxesImage at 0x2497fc44c10>




![output_15_2](../images/2022-12-04-Goldman_Basic/output_15_2.png)



```python
print(tabulate(c_prices.round(0)))
#DF = pd.DataFrame(p_prices)
#DF.to_csv("p_prices.csv")
```

    -  -  -----  -
    0  0  10800  0
    0  0  10800  0
    0  0  10800  0
    0  0      0  0
    -  -  -----  -


# Interest Rates Trees


```python
rd = rates.iloc[0,1:].to_numpy()
rf = rates.iloc[1,1:].to_numpy()

rd_matrix = np.zeros( (N+1, N+1) )
rd_matrix[0,0] = rd[0]

for i in range(1, N+1 ):
    M = i + 1
    rd_matrix[0, i] = rd[i]
    for j in range(1, M ):
        rd_matrix[j, i] = rd_matrix[j - 1, i]
 
plt.spy(rd_matrix)
print(tabulate(rd_matrix.round(4)))
#DF = pd.DataFrame(rd_matrix)
#DF.to_csv("rd_matrix.csv")

rf_matrix = np.zeros( (N+1, N+1) )
rf_matrix[0,0] = rf[0]

for i in range(1, N+1 ):
    M = i + 1
    rf_matrix[0, i] = rf[i]
    for j in range(1, M ):
        rf_matrix[j, i] = rf_matrix[j - 1, i]
```

    ---  ---  ---  ---
    0.1  0.1  0.1  0.1
    0    0.1  0.1  0.1
    0    0    0.1  0.1
    0    0    0    0.1
    ---  ---  ---  ---

![output_18_1](../images/2022-12-04-Goldman_Basic/output_18_1.png)



```python
plt.spy(rf_matrix)
print(tabulate(rf_matrix.round(4)))
#DF = pd.DataFrame(rf_matrix)
#DF.to_csv("rf_matrix.csv")
#print(tabulate(rf_matrix.round(4)))
```

    ----  ----  ----  ----
    0.02  0.02  0.02  0.02
    0     0.02  0.02  0.02
    0     0     0.02  0.02
    0     0     0     0.02
    ----  ----  ----  ----

![output_19_1](../images/2022-12-04-Goldman_Basic/output_19_1.png)


# Q-Value Tree


```python
q_value = np.zeros( (N+1, N+1) )

#  Fill out the remaining values
for i in range(1, N+1 ):
    M = i + 1
    q_value[0, i] = (np.exp(rf_matrix[0, i]*dt) - d)/(u-d)
    for j in range(1, M ):
        q_value[j, i] = (np.exp(rf_matrix[j, i]*dt) - d)/(u-d)
 
plt.spy(q_value)
```




    <matplotlib.image.AxesImage at 0x2497fb1bd00>

![output_21_1](../images/2022-12-04-Goldman_Basic/output_21_1.png)

```python
print(tabulate(q_value.round(4)))
#DF = pd.DataFrame(q_value)
#DF.to_csv("q_value.csv")
```

    -  ------  ------  ------
    0  0.4259  0.4259  0.4259
    0  0.4259  0.4259  0.4259
    0  0       0.4259  0.4259
    0  0       0       0.4259
    -  ------  ------  ------


# Holding Value & Convertible Bond


```python
hp_value = np.zeros( (N+1, N+1) )
hp_value[0, N] = put_ar[-1]

cb_value = np.zeros( (N+1, N+1) )
cb_value[0, N] = max(hp_value[0,N], (stock_prices[0,N]*K)/K_prices[0,N])

#  Fill out the remaining values
for i in range( N, -1, -1 ):
    print("i",i)
    for j in range( 0, i+1 ):
        print("j",j)
        if i == N:
            if j == N:
                cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
                continue;
            hp_value[j+1, i] = hp_value[j, i] # 만기의 상환금액 일괄 적용
            if c_prices[j, i] != 0: cb_value[j, i] = max(min(hp_value[j,i], c_prices[j, i]), (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
            else: cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
        else:
            if i==0 and j==0: coupon = 0 # 기초에는 coupon을 받지 않음
            if cb_value[j, i+1] == (stock_prices[j,i+1]*K)/K_prices[j,i+1]: # 바로 오른쪽 트리가 주식으로 전환한 경우
                if cb_value[j+1, i+1] == (stock_prices[j+1,i+1]*K)/K_prices[j+1,i+1]: # 이 전 두 가지 상황 모두 주식으로 전환한 경우
                    hp_value[j, i] = ( cb_value[j,i+1]*q_value[j,i+1] + cb_value[j+1,i+1]*(1-q_value[j+1,i+1]) )*np.exp(-rf_matrix[j+1,i+1]*dt) + coupon
                    if c_prices[j, i] != 0: cb_value[j, i] = max(min(hp_value[j,i], c_prices[j, i]), (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
                    else: cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i])
                    
                else:  # 한 번은 주식으로 전환하고 한 번은 전환하지 않은 경우
                    hp_value[j, i] = ( cb_value[j,i+1]*q_value[j,i+1] + cb_value[j+1,i+1]*(1-q_value[j+1,i+1]) )*np.exp(-(rf_matrix[j,i+1]*q_value[j,i+1] + rd_matrix[j+1,i+1]*(1-q_value[j+1,i+1]))*dt) + coupon
                    if c_prices[j, i] != 0: cb_value[j, i] = max(min(hp_value[j,i], c_prices[j, i]), (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
                    else: cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i])
                    
            else: # 바로 오른쪽 트리가 주식으로 전환하지 않은 경우
                if cb_value[j+1, i+1] == (stock_prices[j+1,i+1]*K)/K_prices[j+1,i+1]: # 대각선은 주식으로 전환한 경우
                    hp_value[j, i] = ( cb_value[j,i+1]*q_value[j,i+1] + cb_value[j+1, i+1]*(1-q_value[j+1,i+1]) )*np.exp(-(rd_matrix[j,i+1]*(1-q_value[j,i+1]) + rf_matrix[j+1,i+1]*q_value[j+1,i+1])*dt) + coupon
                    if c_prices[j, i] != 0: cb_value[j, i] = max(min(hp_value[j,i], c_prices[j, i]), (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
                    else: cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i])
                    
                else:  # 한 번도 전환하지 않은 경우
                    hp_value[j, i] = ( cb_value[j, i+1]*q_value[j, i+1] + cb_value[j+1, i+1]*(1-q_value[j+1, i+1]) )*np.exp(-rd_matrix[j+1, i+1]*dt) + coupon
                    if c_prices[j, i] != 0: cb_value[j, i] = max(min(hp_value[j,i], c_prices[j, i]), (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i]) # 보유, 주식, 상환 가격 모두 비교
                    else: cb_value[j, i] = max(hp_value[j,i], (stock_prices[j,i]*K)/K_prices[j,i], p_prices[j,i])

                    
coupon = 200 # 재검증의 오류를 방지하기 위해 쿠폰을 재설정함
plt.spy(hp_value)
```

    i 3
    j 0
    j 1
    j 2
    j 3
    i 2
    j 0
    j 1
    j 2
    i 1
    j 0
    j 1
    i 0
    j 0





    <matplotlib.image.AxesImage at 0x2497ffb7130>

![output_24_2](../images/2022-12-04-Goldman_Basic/output_24_2.png)



```python
print(tabulate(hp_value.round(0)))
#DF = pd.DataFrame(hp_value)
#DF.to_csv("hp_value.csv")
```

    -----  -----  -----  -----
    11308  14879  22455  11100
        0   9972  12114  11100
        0      0  10244  11100
        0      0      0  11100
    -----  -----  -----  -----



```python
plt.spy(cb_value)
print(tabulate(cb_value.round(0)))
#DF = pd.DataFrame(cb_value)
#DF.to_csv("cb_value.csv")
```

    -----  -----  -----  -----
    11308  14918  22255  33201
        0   9972  10800  14918
        0      0  10800  11100
        0      0      0  11100
    -----  -----  -----  -----

![output_26_1](../images/2022-12-04-Goldman_Basic/output_26_1.png)



```python
#print(tabulate(cb_value.round(0)))
```
