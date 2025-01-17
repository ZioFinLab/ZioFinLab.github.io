---
layout: single
title: "META's Secret in ESG issues"
categories: [ESG, Python]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# META's Secret in ESG issues
Among 10 companies which experienced two-notch downgrade, only two companies became laggard ones. I wanted to point out Meta Platforms because the stock price of Meta plummeted by 24.15% in a day.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df=pd.read_csv('msci_timeseries_20210901.csv')
```


```python
fb = df.query('ISSUER_ISIN == "US30303M1027"')
fb.reset_index(drop=True, inplace=True)
fb = fb.loc[0, :]
fb
```




    ISSUER_NAME                          FACEBOOK, INC.
    ISSUERID                         IID000000002638948
    ISSUER_TICKER                                    FB
    ISSUER_CUSIP                              30303M102
    ISSUER_SEDOL                                B7TL820
                                            ...        
    CORP_GOVERNANCE_GOV_PILLAR_SD                  -3.3
    ACCOUNTING_GOV_PILLAR_SD                      -0.29
    BOARD_GOV_PILLAR_SD                           -0.86
    OWNERSHIP_GOV_PILLAR_SD                       -1.41
    PAY_GOV_PILLAR_SD                             -0.78
    Name: 0, Length: 206, dtype: object




```python
finder = fb.filter(regex = 'WEIGHT')
weight = finder[(finder > 0)]
weight
```




    WEIGHTED_AVERAGE_SCORE          3.7
    ENVIRONMENTAL_PILLAR_WEIGHT       5
    SOCIAL_PILLAR_WEIGHT             53
    GOVERNANCE_PILLAR_WEIGHT         42
    CLIMATE_CHANGE_THEME_WEIGHT     5.0
    HUMAN_CAPITAL_THEME_WEIGHT       24
    PRODUCT_SAFETY_THEME_WEIGHT    29.0
    CARBON_EMISSIONS_WEIGHT         5.0
    HUMAN_CAPITAL_DEV_WEIGHT       24.0
    PRIVACY_DATA_SEC_WEIGHT        29.0
    Name: 0, dtype: object




```python
inter_ind = df[(df['IVA_INDUSTRY'] == 'Interactive Media & Services')]
inter_ind[['WEIGHTED_AVERAGE_SCORE','INDUSTRY_ADJUSTED_SCORE']].describe().round(3)
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
      <th>WEIGHTED_AVERAGE_SCORE</th>
      <th>INDUSTRY_ADJUSTED_SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>23.000</td>
      <td>23.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.104</td>
      <td>4.391</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.952</td>
      <td>2.296</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.100</td>
      <td>0.800</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.450</td>
      <td>2.850</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.100</td>
      <td>4.100</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.750</td>
      <td>6.050</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.500</td>
      <td>10.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
inter_ind['IVA_COMPANY_RATING'].value_counts()[['CCC','B','BB','BBB','A','AA','AAA']].plot(kind='bar')
count = inter_ind['IVA_COMPANY_RATING'].value_counts()
count
```




    BB     6
    B      5
    BBB    4
    A      4
    AA     2
    AAA    1
    CCC    1
    Name: IVA_COMPANY_RATING, dtype: int64




![output_6_1](../images/2022-12-02-ESG-META/output_6_1.png)


I used Bottom-up analysis based on MSCI ESG rating methodology. I thought four numbers are extreme. The first one is Human Capital Development.
Meta’s exposure score is high, but its management seems not enough. However, I figured out the exposure scores of companies in Interactive Media & Services are generally high as Meta’s and Meta’s management score is higher than industry’s average. Like HCD, Ownership and Control subjection points is a little bit lower than industry’s average, but it is not extremely bad.


```python
hcd_ind = inter_ind['HUMAN_CAPITAL_DEV_MGMT_SCORE']
plt.hist(hcd_ind, bins=7)
```




    (array([1., 0., 3., 1., 3., 8., 7.]),
     array([0.7       , 1.42857143, 2.15714286, 2.88571429, 3.61428571,
            4.34285714, 5.07142857, 5.8       ]),
     <BarContainer object of 7 artists>)

![output_8_1](../images/2022-12-02-ESG-META/output_8_1.png)



```python
hcd_ind.describe()
```




    count    23.000000
    mean      4.334783
    std       1.245942
    min       0.700000
    25%       4.100000
    50%       4.600000
    75%       5.100000
    max       5.800000
    Name: HUMAN_CAPITAL_DEV_MGMT_SCORE, dtype: float64




```python
hcd_ind = inter_ind['HUMAN_CAPITAL_DEV_EXP_SCORE']
plt.hist(hcd_ind, bins=7)
```




    (array([ 1.,  0.,  1.,  1., 14.,  1.,  5.]),
     array([6.2       , 6.64285714, 7.08571429, 7.52857143, 7.97142857,
            8.41428571, 8.85714286, 9.3       ]),
     <BarContainer object of 7 artists>)




![output_10_1](../images/2022-12-02-ESG-META/output_10_1.png)



```python
hcd_ind.describe()
```




    count    23.000000
    mean      8.343478
    std       0.699887
    min       6.200000
    25%       8.200000
    50%       8.300000
    75%       8.450000
    max       9.300000
    Name: HUMAN_CAPITAL_DEV_EXP_SCORE, dtype: float64




```python
hcd = df['HUMAN_CAPITAL_DEV_MGMT_SCORE']
plt.hist(hcd, bins=10)
```




    (array([ 74.,  96., 158., 291., 427., 344., 321., 303., 155.,  48.]),
     array([0.  , 0.92, 1.84, 2.76, 3.68, 4.6 , 5.52, 6.44, 7.36, 8.28, 9.2 ]),
     <BarContainer object of 10 artists>)



![output_12_1](../images/2022-12-02-ESG-META/output_12_1.png)

```python
hcd.describe()
```




    count    2217.000000
    mean        4.800496
    std         1.913621
    min         0.000000
    25%         3.500000
    50%         4.800000
    75%         6.300000
    max         9.200000
    Name: HUMAN_CAPITAL_DEV_MGMT_SCORE, dtype: float64




```python
hcd = df['HUMAN_CAPITAL_DEV_EXP_SCORE']
plt.hist(hcd, bins=10)
```




    (array([115.,  84., 164., 145., 292., 277., 452., 325., 222., 141.]),
     array([ 2.3 ,  3.07,  3.84,  4.61,  5.38,  6.15,  6.92,  7.69,  8.46,
             9.23, 10.  ]),
     <BarContainer object of 10 artists>)

![output_14_1](../images/2022-12-02-ESG-META/output_14_1.png)



```python
hcd.describe()
```




    count    2217.000000
    mean        6.673207
    std         1.833972
    min         2.300000
    25%         5.500000
    50%         7.000000
    75%         8.100000
    max        10.000000
    Name: HUMAN_CAPITAL_DEV_EXP_SCORE, dtype: float64



Meta’s Privacy Data Security exposure score is higher than industry’s average. However, Meta’s management score is lower than industry’s average. Therefore, we can infer that the gap between the exposure and management triggered Meta to get a bad score in this issue.


```python
pds_ind = inter_ind['PRIVACY_DATA_SEC_EXP_SCORE']
plt.hist(pds_ind, bins=7)
```




    (array([1., 0., 2., 2., 9., 5., 4.]),
     array([5.3       , 5.92857143, 6.55714286, 7.18571429, 7.81428571,
            8.44285714, 9.07142857, 9.7       ]),
     <BarContainer object of 7 artists>)




![output_17_1](../images/2022-12-02-ESG-META/output_17_1.png)



```python
pds_ind.describe()
```




    count    23.000000
    mean      8.234783
    std       0.982394
    min       5.300000
    25%       7.900000
    50%       8.200000
    75%       8.850000
    max       9.700000
    Name: PRIVACY_DATA_SEC_EXP_SCORE, dtype: float64




```python
pds_ind = inter_ind['PRIVACY_DATA_SEC_MGMT_SCORE']
plt.hist(pds_ind, bins=7)
```




    (array([2., 6., 9., 2., 2., 0., 2.]),
     array([3.3       , 4.11428571, 4.92857143, 5.74285714, 6.55714286,
            7.37142857, 8.18571429, 9.        ]),
     <BarContainer object of 7 artists>)



![output_19_1](../images/2022-12-02-ESG-META/output_19_1.png)

```python
pds_ind.describe()
```




    count    23.000000
    mean      5.421739
    std       1.416448
    min       3.300000
    25%       4.500000
    50%       5.200000
    75%       5.850000
    max       9.000000
    Name: PRIVACY_DATA_SEC_MGMT_SCORE, dtype: float64




```python
pds = df['PRIVACY_DATA_SEC_MGMT_SCORE']
plt.hist(pds, bins=10)
```




    (array([166.,  74., 101., 192., 339., 661., 333., 191.,  86.,  22.]),
     array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]),
     <BarContainer object of 10 artists>)




![output_21_1](../images/2022-12-02-ESG-META/output_21_1.png)



```python
pds.describe()
```




    count    2165.000000
    mean        4.885820
    std         2.079936
    min         0.000000
    25%         4.200000
    50%         5.000000
    75%         6.300000
    max        10.000000
    Name: PRIVACY_DATA_SEC_MGMT_SCORE, dtype: float64




```python
pds = df['PRIVACY_DATA_SEC_EXP_SCORE']
plt.hist(pds, bins=10)
```




    (array([ 12., 127., 746., 272., 144., 139., 222., 222., 205.,  75.]),
     array([0.5 , 1.42, 2.34, 3.26, 4.18, 5.1 , 6.02, 6.94, 7.86, 8.78, 9.7 ]),
     <BarContainer object of 10 artists>)

![output_23_1](../images/2022-12-02-ESG-META/output_23_1.png)



```python
pds.describe()
```




    count    2164.000000
    mean        4.696026
    std         2.220830
    min         0.500000
    25%         2.700000
    50%         3.900000
    75%         6.800000
    max         9.700000
    Name: PRIVACY_DATA_SEC_EXP_SCORE, dtype: float64




```python
ogp_ind = inter_ind['OWNERSHIP_GOV_PILLAR_SD']
plt.hist(ogp_ind, bins=7)
```




    (array([4., 3., 3., 3., 2., 2., 6.]),
     array([-2.3       , -1.98285714, -1.66571429, -1.34857143, -1.03142857,
            -0.71428571, -0.39714286, -0.08      ]),
     <BarContainer object of 7 artists>)



![output_25_1](../images/2022-12-02-ESG-META/output_25_1.png)

```python
ogp_ind.describe()
```




    count    23.000000
    mean     -1.113043
    std       0.736697
    min      -2.300000
    25%      -1.780000
    50%      -1.250000
    75%      -0.410000
    max      -0.080000
    Name: OWNERSHIP_GOV_PILLAR_SD, dtype: float64



As you can see this graph, in the Business Ethics issue, Meta is the one of the worst companies in the industry.
As a result, I figured out Two Issues, Privacy Data Security and Business Ethics, made Meta be an ESG laggard.


```python
ogp = df['OWNERSHIP_GOV_PILLAR_SD']
plt.hist(ogp, bins=10)
```




    (array([  1.,   1.,   2.,   5.,  34.,  75., 209., 524., 590., 774.]),
     array([-4.06 , -3.654, -3.248, -2.842, -2.436, -2.03 , -1.624, -1.218,
            -0.812, -0.406,  0.   ]),
     <BarContainer object of 10 artists>)

![output_28_1](../images/2022-12-02-ESG-META/output_28_1.png)



```python
ogp.describe()
```




    count    2215.000000
    mean       -0.709819
    std         0.501394
    min        -4.060000
    25%        -1.020000
    50%        -0.620000
    75%        -0.270000
    max         0.000000
    Name: OWNERSHIP_GOV_PILLAR_SD, dtype: float64



I think Privacy data security and Business ethics are not mutually exclusive. Basically Privacy data is users basic right, poorly managing pds can mean Meta’s business ethics are wrong.
Because Meta has 2.91 billion Monthly Active users, Meta should establish the most robust security system.
However, since the Scandal in 2018, Hackers have penetrated Meta’s database and steal users’ data even in 2021.
In the Business Ethics aspect. Meta confronts a variety of issues.
One of the crucial issues is Meta collects data without users’ consent.
Pointing out this issue, last year Apple started to provide a new service preventing sharing users’ data on social media. The service encouraged users to realize how badly Meta collects users’ data.
Especially yesterday, Mark Zuckerberg threatened to pull over Facebook and Instagram in Europe against data privacy regulation rather than comply with it.
Meta behaves like a tech dictator.


```python
be_ind = inter_ind['BUS_ETHICS_GOV_PILLAR_SD']
plt.hist(be_ind, bins=7)
```




    (array([ 3.,  2.,  3.,  2., 11.,  1.,  1.]),
     array([-3.3       , -2.94285714, -2.58571429, -2.22857143, -1.87142857,
            -1.51428571, -1.15714286, -0.8       ]),
     <BarContainer object of 7 artists>)




![output_31_1](../images/2022-12-02-ESG-META/output_31_1.png)



```python
be_ind.describe()
```




    count    23.00000
    mean     -2.06087
    std       0.62794
    min      -3.30000
    25%      -2.40000
    50%      -1.80000
    75%      -1.70000
    max      -0.80000
    Name: BUS_ETHICS_GOV_PILLAR_SD, dtype: float64




```python
be = df['BUS_ETHICS_GOV_PILLAR_SD']
plt.hist(be, bins=10)
```




    (array([  1.,   9.,  32., 127., 194., 410., 713., 530., 160.,  44.]),
     array([-4.5 , -4.07, -3.64, -3.21, -2.78, -2.35, -1.92, -1.49, -1.06,
            -0.63, -0.2 ]),
     <BarContainer object of 10 artists>)




![output_33_1](../images/2022-12-02-ESG-META/output_33_1.png)



```python
be.describe()
```




    count    2220.000000
    mean       -1.762027
    std         0.623763
    min        -4.500000
    25%        -2.100000
    50%        -1.700000
    75%        -1.300000
    max        -0.200000
    Name: BUS_ETHICS_GOV_PILLAR_SD, dtype: float64



Eventually, Meta’s destructive behaviors made users leave Facebook, and Meta’s earnings decreased. After Meta’s Q4 earnings announcement, the stock price plummeted.
Handling poorly at two issues caused not only Meta’s CF to decrease but also future risk to increase.
Investors concerned about the ESG rating would avoid investing in Meta if they checked that the rating plummeted last year.
