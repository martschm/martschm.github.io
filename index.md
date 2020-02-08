# [View My LinkedIn Profile](https://www.linkedin.com/in/martschm/)

<br>

# [Curriculum Vitae](/about_me)

<br>

# My Portfolio

---

## [Diploma Thesis](/diploma_thesis)

### Financial and Actuarial Mathematics

Aggregation of Integer-Valued Risks with Copula Induced Dependency Structure (including R-Code)

---

---
## [Route Optimization (brute force) in Python](/python_route_optimization)

The program gets as input a csv-file with locations (longitude and latitude) and determines the best route (based on time or distance) starting from a specified location using Google's Distance Matrix API

---

---
## [Web-Scraping and Clustering of News Articles in Python]()

Under construction ...

---







```python
import pandas as pd
import matplotlib.pyplot as plt

behebungen=pd.read_csv("datasets/FINAL_behebungen_training.csv",
                       dtype={"ATM_ID":str,"BETRAG":int,"DATE":str,"KUNDEN_ID":int},
                       parse_dates=["DATE"])

kunden=pd.read_csv("datasets/FINAL_kunden_training.csv", dtype={"ALTER":int,"FEMALE":bool,"PLZ":int,"KUNDEN_ID":int})

plz=pd.read_csv('datasets/geodata_semicolon.txt', sep=";", header=None)
plz.columns = ["PLZ", "ort", "lat", "long"]
plz_mean=pd.merge(plz.groupby('PLZ')['lat'].mean(), plz.groupby("PLZ")["long"].mean(), on="PLZ")

atm_id_plz=pd.read_csv("datasets/ATM_PLZ.csv", sep=";", dtype={"ATM_ID": str, "ATM_PLZ":int})

```


```python
kunden.head()
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
      <th>ALTER</th>
      <th>FEMALE</th>
      <th>PLZ</th>
      <th>KUNDEN_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>True</td>
      <td>8760</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>False</td>
      <td>1170</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52</td>
      <td>True</td>
      <td>2452</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29</td>
      <td>False</td>
      <td>1160</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>False</td>
      <td>1160</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
behebungen.head()
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
      <th>ATM_ID</th>
      <th>BETRAG</th>
      <th>DATE</th>
      <th>KUNDEN_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11154136</td>
      <td>500</td>
      <td>2017-01-30 14:06:14</td>
      <td>44321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11154126</td>
      <td>30</td>
      <td>2017-02-07 10:20:46</td>
      <td>44321</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11154216</td>
      <td>160</td>
      <td>2017-03-01 09:18:05</td>
      <td>44321</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11154146</td>
      <td>600</td>
      <td>2017-03-31 13:08:04</td>
      <td>44321</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11154146</td>
      <td>20</td>
      <td>2017-04-05 13:12:16</td>
      <td>44321</td>
    </tr>
  </tbody>
</table>
</div>




```python
data=pd.merge(behebungen, kunden, on='KUNDEN_ID')
data=pd.merge(data,atm_id_plz, on="ATM_ID")

data=pd.merge(data,plz_mean, on="PLZ")
data=pd.merge(data,plz_mean, how='left', left_on=['ATM_PLZ'], right_on=['PLZ'])

data.rename(index=str, columns={"lat_x": "PLZ_LAT", "long_x": "PLZ_LONG"})
data=data.rename(columns={"lat_x": "PLZ_LAT", "long_x": "PLZ_LONG","lat_y": "ATM_PLZ_LAT", "long_y": "ATM_PLZ_LONG"})

data["WEEKDAY"]=data["DATE"].apply(lambda x: x.weekday())
data["MONTH"]=data["DATE"].apply(lambda x: x.month)
data["DAY"]=data["DATE"].apply(lambda x: x.day)
data["HOUR"]=data["DATE"].apply(lambda x: x.hour)

```


```python

```


```python
import math
data["PLZ_2_DIGITS"]=data["PLZ"].apply(lambda x: math.floor(x/100))
data["PLZ_ATM_2_DIGITS"]=data["ATM_PLZ"].apply(lambda x: math.floor(x/100))
```


```python
import math
data["PLZ_1_DIGIT"]=data["PLZ"].apply(lambda x: math.floor(x/1000))
data["PLZ_ATM_1_DIGIT"]=data["ATM_PLZ"].apply(lambda x: math.floor(x/1000))
```


```python
most_freq_plz_2_digits=data.groupby(["KUNDEN_ID","PLZ_ATM_2_DIGITS"]).size().to_frame("COUNT")
idx=most_freq_plz_2_digits.groupby("KUNDEN_ID")["COUNT"].transform(max)==most_freq_plz_2_digits["COUNT"]
most_freq_plz_2_digits=most_freq_plz_2_digits[idx]
most_freq_plz_2_digits=most_freq_plz_2_digits.reset_index(level=['PLZ_ATM_2_DIGITS'])
most_freq_plz_2_digits=most_freq_plz_2_digits.reset_index(level=['KUNDEN_ID'])
most_freq_plz_2_digits=most_freq_plz_2_digits.groupby('KUNDEN_ID').tail(1)
```


```python
most_freq_plz=data.groupby(["KUNDEN_ID","ATM_PLZ"]).size().to_frame("COUNT")
idx=most_freq_plz.groupby("KUNDEN_ID")["COUNT"].transform(max)==most_freq_plz["COUNT"]
most_freq_plz=most_freq_plz[idx]
most_freq_plz=most_freq_plz.reset_index(level=['ATM_PLZ'])
most_freq_plz=most_freq_plz.reset_index(level=['KUNDEN_ID'])
most_freq_plz=most_freq_plz.groupby('KUNDEN_ID').tail(1)
```


```python
wochenende=data[(data["WEEKDAY"]==5) | (data["WEEKDAY"]==6) |
                ((data["HOUR"]>=18) & (data["HOUR"]<=7))]
most_freq_plz_wochenende=wochenende.groupby(["KUNDEN_ID","ATM_PLZ"]).size().to_frame("COUNT")
idx=most_freq_plz_wochenende.groupby("KUNDEN_ID")["COUNT"].transform(max)==most_freq_plz_wochenende["COUNT"]
most_freq_plz_wochenende=most_freq_plz_wochenende[idx]
most_freq_plz_wochenende=most_freq_plz_wochenende.reset_index(level=['ATM_PLZ'])
most_freq_plz_wochenende=most_freq_plz_wochenende.reset_index(level=['KUNDEN_ID'])
most_freq_plz_wochenende=most_freq_plz_wochenende.groupby("KUNDEN_ID").tail(1)
most_freq_plz_wochenende=most_freq_plz_wochenende.rename(index=str, 
                                                         columns={"ATM_PLZ": "ATM_PLZ_WE", "COUNT": "COUNT_WE"})

```


```python

```


```python

```


```python
1
```




    1




```python
test=pd.merge(kunden,most_freq_plz,on="KUNDEN_ID")
test=pd.merge(test, test.groupby("ATM_PLZ").size().to_frame('size'), on="ATM_PLZ")
test.size=test["size"]/max(test["size"])
```


```python
fig = plt.figure()
plt.scatter(test["PLZ"],test["ATM_PLZ"], alpha=0.2, 
                 s=test["size"]*100)
plt.yticks([1000,2000,3000,4000,5000,6000,7000,8000,9000])
plt.xticks([1000,2000,3000,4000,5000,6000,7000,8000,9000])
plt.xlabel("CUSTOMER POSTAL CODE")
plt.ylabel("ATM POSTAL CODE")
fig.savefig("dichtematrix.png", bbox_inches='tight', dpi=500)
```


![png](output_14_0.png)



```python

```


    <Figure size 576x360 with 0 Axes>



```python
wochenende=data[(data["WEEKDAY"]==5) | (data["WEEKDAY"]==6) |
                ((data["HOUR"]>=18) & (data["HOUR"]<=7))]
most_freq_plz_wochenende=wochenende.groupby(["KUNDEN_ID","ATM_PLZ"]).size().to_frame("COUNT")
idx=most_freq_plz_wochenende.groupby("KUNDEN_ID")["COUNT"].transform(max)==most_freq_plz_wochenende["COUNT"]
most_freq_plz_wochenende=most_freq_plz_wochenende[idx]
test=most_freq_plz_wochenende.reset_index(level=['ATM_PLZ'])
a=pd.merge(kunden,test,on="KUNDEN_ID")
```


```python
most_freq_plz=pd.merge(data.groupby(["KUNDEN_ID","ATM_PLZ"])["BETRAG"].agg("sum").to_frame("SUM"),
                       data.groupby(["KUNDEN_ID","ATM_PLZ"]).size().to_frame("COUNT"),
                       on=["KUNDEN_ID","ATM_PLZ"])
most_freq_plz=most_freq_plz.reset_index(level=['ATM_PLZ'])

most_freq_plz.groupby("KUNDEN_ID").max()
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
      <th>ATM_PLZ</th>
      <th>SUM</th>
      <th>COUNT</th>
    </tr>
    <tr>
      <th>KUNDEN_ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1170</td>
      <td>1780</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7350</td>
      <td>1240</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2334</td>
      <td>2790</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1230</td>
      <td>2000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1300</td>
      <td>13490</td>
      <td>56</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9026</td>
      <td>8430</td>
      <td>76</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7222</td>
      <td>6970</td>
      <td>69</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2331</td>
      <td>4630</td>
      <td>59</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9900</td>
      <td>1510</td>
      <td>40</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1230</td>
      <td>11040</td>
      <td>63</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8055</td>
      <td>1400</td>
      <td>16</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1050</td>
      <td>12350</td>
      <td>53</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1120</td>
      <td>25900</td>
      <td>42</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1220</td>
      <td>8700</td>
      <td>32</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1150</td>
      <td>6300</td>
      <td>78</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1180</td>
      <td>7150</td>
      <td>31</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6410</td>
      <td>3600</td>
      <td>9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7431</td>
      <td>14100</td>
      <td>45</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1120</td>
      <td>3590</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>9500</td>
      <td>3160</td>
      <td>40</td>
    </tr>
    <tr>
      <th>24</th>
      <td>4820</td>
      <td>3820</td>
      <td>18</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7210</td>
      <td>400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1100</td>
      <td>3700</td>
      <td>7</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8700</td>
      <td>8240</td>
      <td>99</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7142</td>
      <td>11620</td>
      <td>52</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1160</td>
      <td>5450</td>
      <td>34</td>
    </tr>
    <tr>
      <th>31</th>
      <td>4020</td>
      <td>1650</td>
      <td>34</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1110</td>
      <td>34800</td>
      <td>61</td>
    </tr>
    <tr>
      <th>33</th>
      <td>8020</td>
      <td>3310</td>
      <td>45</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1110</td>
      <td>4420</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>399966</th>
      <td>6021</td>
      <td>2180</td>
      <td>26</td>
    </tr>
    <tr>
      <th>399967</th>
      <td>1230</td>
      <td>4720</td>
      <td>14</td>
    </tr>
    <tr>
      <th>399968</th>
      <td>1150</td>
      <td>2140</td>
      <td>18</td>
    </tr>
    <tr>
      <th>399969</th>
      <td>1170</td>
      <td>280</td>
      <td>2</td>
    </tr>
    <tr>
      <th>399970</th>
      <td>1210</td>
      <td>3940</td>
      <td>103</td>
    </tr>
    <tr>
      <th>399971</th>
      <td>5300</td>
      <td>1710</td>
      <td>34</td>
    </tr>
    <tr>
      <th>399973</th>
      <td>1220</td>
      <td>7500</td>
      <td>35</td>
    </tr>
    <tr>
      <th>399974</th>
      <td>1200</td>
      <td>5900</td>
      <td>27</td>
    </tr>
    <tr>
      <th>399975</th>
      <td>8650</td>
      <td>6550</td>
      <td>27</td>
    </tr>
    <tr>
      <th>399976</th>
      <td>1160</td>
      <td>1560</td>
      <td>14</td>
    </tr>
    <tr>
      <th>399977</th>
      <td>9800</td>
      <td>3860</td>
      <td>23</td>
    </tr>
    <tr>
      <th>399978</th>
      <td>2700</td>
      <td>600</td>
      <td>5</td>
    </tr>
    <tr>
      <th>399979</th>
      <td>6020</td>
      <td>510</td>
      <td>3</td>
    </tr>
    <tr>
      <th>399980</th>
      <td>9020</td>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>399981</th>
      <td>4066</td>
      <td>13350</td>
      <td>37</td>
    </tr>
    <tr>
      <th>399982</th>
      <td>1220</td>
      <td>1460</td>
      <td>28</td>
    </tr>
    <tr>
      <th>399983</th>
      <td>2000</td>
      <td>6080</td>
      <td>33</td>
    </tr>
    <tr>
      <th>399984</th>
      <td>6262</td>
      <td>90</td>
      <td>3</td>
    </tr>
    <tr>
      <th>399985</th>
      <td>1040</td>
      <td>7180</td>
      <td>41</td>
    </tr>
    <tr>
      <th>399987</th>
      <td>3002</td>
      <td>19420</td>
      <td>163</td>
    </tr>
    <tr>
      <th>399988</th>
      <td>4040</td>
      <td>8050</td>
      <td>28</td>
    </tr>
    <tr>
      <th>399990</th>
      <td>3571</td>
      <td>400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>399991</th>
      <td>1150</td>
      <td>22480</td>
      <td>51</td>
    </tr>
    <tr>
      <th>399992</th>
      <td>2344</td>
      <td>2960</td>
      <td>25</td>
    </tr>
    <tr>
      <th>399993</th>
      <td>1150</td>
      <td>4560</td>
      <td>84</td>
    </tr>
    <tr>
      <th>399995</th>
      <td>6342</td>
      <td>6500</td>
      <td>32</td>
    </tr>
    <tr>
      <th>399996</th>
      <td>3131</td>
      <td>5350</td>
      <td>34</td>
    </tr>
    <tr>
      <th>399997</th>
      <td>6380</td>
      <td>950</td>
      <td>4</td>
    </tr>
    <tr>
      <th>399998</th>
      <td>4240</td>
      <td>380</td>
      <td>4</td>
    </tr>
    <tr>
      <th>399999</th>
      <td>1050</td>
      <td>3920</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
<p>335658 rows × 3 columns</p>
</div>




```python
plot_data=pd.merge(data, data.groupby("PLZ").size().to_frame('size'), on="PLZ")
plot_data["size"]=plot_data["size"]/max(plot_data["size"])
data_sample=plot_data.sample(n=100000, random_state=1)
```


```python
fig = plt.figure()
plt.rcParams["figure.figsize"]=(8,5)
data_sample.plot(kind="scatter",x="PLZ_LAT",y="PLZ_LONG", alpha=0.2, 
                 s=data_sample["size"]*100)

fig.savefig("leer.png", bbox_inches='tight', dpi=500)

```


    <Figure size 576x360 with 0 Axes>



![png](output_19_1.png)



```python
data_sample_pendler=data_sample[data_sample["PLZ_1_DIGIT"]!=data_sample["PLZ_ATM_1_DIGIT"]]
```


```python
data_sample_pendler;
```


```python
import matplotlib
cmap = matplotlib.cm.get_cmap('Spectral')
```


```python
fig = plt.figure()
plt.rcParams["figure.figsize"]=(8,5)
plt.scatter(data_sample["PLZ_LAT"],data_sample["PLZ_LONG"], alpha=0.2, 
                 s=data_sample["size"]*100)

fig.savefig("leer.png", bbox_inches='tight', dpi=500)
```


![png](output_23_0.png)



```python
fig = plt.figure()
plt.rcParams["figure.figsize"]=(8,5)
for x1,y1,x2,y2,c,d in zip(data_sample_pendler["PLZ_LAT"],data_sample_pendler["PLZ_LONG"],
                         data_sample_pendler["ATM_PLZ_LAT"],data_sample_pendler["ATM_PLZ_LONG"],
                         data_sample_pendler["PLZ_1_DIGIT"],data_sample_pendler["PLZ_ATM_1_DIGIT"]):
    #if c ==1:continue
    if (c ==2 or c ==3 or c == 7 or c==4) and d == 1:plt.plot([x1,x2],[y1,y2],alpha=0.1,color=cmap(c/10.))
plt.scatter(data_sample["PLZ_LAT"],data_sample["PLZ_LONG"], alpha=0.2, 
                 s=data_sample["size"]*100)

fig.savefig("PendlerNachWien.png", bbox_inches='tight', dpi=500)
```


![png](output_24_0.png)



```python
fig = plt.figure()
plt.rcParams["figure.figsize"]=(8,5)
for x1,y1,x2,y2,c,d in zip(data_sample_pendler["PLZ_LAT"],data_sample_pendler["PLZ_LONG"],
                         data_sample_pendler["ATM_PLZ_LAT"],data_sample_pendler["ATM_PLZ_LONG"],
                         data_sample_pendler["PLZ_1_DIGIT"],data_sample_pendler["PLZ_ATM_1_DIGIT"]):
    #if c ==1:continue
    if (c ==9) and (d == 8):plt.plot([x1,x2],[y1,y2],alpha=0.1,color=cmap(c/40.))
plt.scatter(data_sample["PLZ_LAT"],data_sample["PLZ_LONG"], alpha=0.2, 
                 s=data_sample["size"]*100)

fig.savefig("PendlerKaerntenSteiermark.png", bbox_inches='tight', dpi=500)
```


![png](output_25_0.png)



```python
fig = plt.figure()
plt.rcParams["figure.figsize"]=(8,5)
for x1,y1,x2,y2,c,d in zip(data_sample_pendler["PLZ_LAT"],data_sample_pendler["PLZ_LONG"],
                         data_sample_pendler["ATM_PLZ_LAT"],data_sample_pendler["ATM_PLZ_LONG"],
                         data_sample_pendler["PLZ_1_DIGIT"],data_sample_pendler["PLZ_ATM_1_DIGIT"]):

    if ((c == 7) & (d == 2)):plt.plot([x1,x2],[y1,y2],alpha=0.1,color=cmap(d/10.))
plt.scatter(data_sample["PLZ_LAT"],data_sample["PLZ_LONG"], alpha=0.2, 
                 s=data_sample["size"]*100)

fig.savefig("Pendler_Burgenland_Noe.png", bbox_inches='tight', dpi=500)

```


![png](output_26_0.png)



```python
fig = plt.figure()
plt.rcParams["figure.figsize"]=(8,5)
for x1,y1,x2,y2,c,d in zip(data_sample_pendler["PLZ_LAT"],data_sample_pendler["PLZ_LONG"],
                         data_sample_pendler["ATM_PLZ_LAT"],data_sample_pendler["ATM_PLZ_LONG"],
                         data_sample_pendler["PLZ_1_DIGIT"],data_sample_pendler["PLZ_ATM_1_DIGIT"]):
    #if c ==1:continue
    if (c == 1):plt.plot([x1,x2],[y1,y2],alpha=0.1,color=cmap(d/10.))
plt.scatter(data_sample["PLZ_LAT"],data_sample["PLZ_LONG"], alpha=0.2, 
                 s=data_sample["size"]*100)

fig.savefig("Wiener.png", bbox_inches='tight', dpi=500)
```


![png](output_27_0.png)



```python
plot_data_2=pd.merge(data, data.groupby("ATM_PLZ").size().to_frame('size'), on="ATM_PLZ")
plot_data_2["size"]=plot_data_2["size"]/max(plot_data_2["size"])
data_sample_2=plot_data_2.sample(n=100000, random_state=1)

```


```python
plt.rcParams["figure.figsize"]=(8,5)
data_sample_2.plot(kind="scatter",x="ATM_PLZ_LAT",y="ATM_PLZ_LONG", alpha=0.2, 
                 s=data_sample_2["size"]*100)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1298992b0>




![png](output_29_1.png)



```python

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
      <th>ATM_ID</th>
      <th>BETRAG</th>
      <th>DATE</th>
      <th>KUNDEN_ID</th>
      <th>ALTER</th>
      <th>FEMALE</th>
      <th>PLZ</th>
      <th>ATM_PLZ</th>
      <th>PLZ_LAT</th>
      <th>PLZ_LONG</th>
      <th>...</th>
      <th>ATM_PLZ_LONG</th>
      <th>WEEKDAY</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>HOUR</th>
      <th>PLZ_2_DIGITS</th>
      <th>PLZ_ATM_2_DIGITS</th>
      <th>PLZ_1_DIGIT</th>
      <th>PLZ_ATM_1_DIGIT</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12313703</th>
      <td>25683</td>
      <td>30</td>
      <td>2017-11-08 10:20:01</td>
      <td>204728</td>
      <td>21</td>
      <td>False</td>
      <td>2734</td>
      <td>2700</td>
      <td>15.963200</td>
      <td>47.778867</td>
      <td>...</td>
      <td>47.810260</td>
      <td>2</td>
      <td>11</td>
      <td>8</td>
      <td>10</td>
      <td>27</td>
      <td>27</td>
      <td>2</td>
      <td>2</td>
      <td>0.002278</td>
    </tr>
    <tr>
      <th>7702851</th>
      <td>S6EE0870</td>
      <td>200</td>
      <td>2017-11-07 09:07:47</td>
      <td>12361</td>
      <td>77</td>
      <td>False</td>
      <td>8010</td>
      <td>8334</td>
      <td>15.541700</td>
      <td>47.083300</td>
      <td>...</td>
      <td>46.995000</td>
      <td>1</td>
      <td>11</td>
      <td>7</td>
      <td>9</td>
      <td>80</td>
      <td>83</td>
      <td>8</td>
      <td>8</td>
      <td>0.203274</td>
    </tr>
    <tr>
      <th>3275377</th>
      <td>S6EE6125</td>
      <td>250</td>
      <td>2017-12-05 16:55:33</td>
      <td>137033</td>
      <td>36</td>
      <td>True</td>
      <td>1040</td>
      <td>1040</td>
      <td>16.367100</td>
      <td>48.192000</td>
      <td>...</td>
      <td>48.192000</td>
      <td>1</td>
      <td>12</td>
      <td>5</td>
      <td>16</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0.134075</td>
    </tr>
    <tr>
      <th>8898072</th>
      <td>11024136</td>
      <td>50</td>
      <td>2017-04-03 14:44:31</td>
      <td>199527</td>
      <td>27</td>
      <td>False</td>
      <td>7100</td>
      <td>1020</td>
      <td>16.855350</td>
      <td>47.937150</td>
      <td>...</td>
      <td>48.216700</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>14</td>
      <td>71</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>0.016230</td>
    </tr>
    <tr>
      <th>8520491</th>
      <td>46911</td>
      <td>50</td>
      <td>2017-08-24 07:47:59</td>
      <td>331976</td>
      <td>33</td>
      <td>False</td>
      <td>6020</td>
      <td>9020</td>
      <td>11.401758</td>
      <td>47.245950</td>
      <td>...</td>
      <td>46.650609</td>
      <td>3</td>
      <td>8</td>
      <td>24</td>
      <td>7</td>
      <td>60</td>
      <td>90</td>
      <td>6</td>
      <td>9</td>
      <td>0.304742</td>
    </tr>
    <tr>
      <th>4067358</th>
      <td>11104116</td>
      <td>680</td>
      <td>2017-07-10 17:01:50</td>
      <td>230329</td>
      <td>38</td>
      <td>False</td>
      <td>1100</td>
      <td>1100</td>
      <td>16.387800</td>
      <td>48.152100</td>
      <td>...</td>
      <td>48.152100</td>
      <td>0</td>
      <td>7</td>
      <td>10</td>
      <td>17</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8839223</th>
      <td>S6EE0035</td>
      <td>90</td>
      <td>2017-01-09 07:18:45</td>
      <td>180426</td>
      <td>51</td>
      <td>True</td>
      <td>9100</td>
      <td>9020</td>
      <td>14.625718</td>
      <td>46.666004</td>
      <td>...</td>
      <td>46.650609</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>7</td>
      <td>91</td>
      <td>90</td>
      <td>9</td>
      <td>9</td>
      <td>0.014070</td>
    </tr>
    <tr>
      <th>9284966</th>
      <td>4613</td>
      <td>200</td>
      <td>2017-05-05 08:32:22</td>
      <td>384522</td>
      <td>60</td>
      <td>False</td>
      <td>4040</td>
      <td>4053</td>
      <td>14.270133</td>
      <td>48.349193</td>
      <td>...</td>
      <td>48.188378</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>40</td>
      <td>40</td>
      <td>4</td>
      <td>4</td>
      <td>0.082254</td>
    </tr>
    <tr>
      <th>863293</th>
      <td>S6EE0277</td>
      <td>10</td>
      <td>2017-10-06 07:31:08</td>
      <td>66211</td>
      <td>53</td>
      <td>False</td>
      <td>1220</td>
      <td>1220</td>
      <td>16.495000</td>
      <td>48.219000</td>
      <td>...</td>
      <td>48.219000</td>
      <td>4</td>
      <td>10</td>
      <td>6</td>
      <td>7</td>
      <td>12</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0.555237</td>
    </tr>
    <tr>
      <th>10339667</th>
      <td>13352</td>
      <td>10</td>
      <td>2017-07-28 15:00:52</td>
      <td>225524</td>
      <td>33</td>
      <td>False</td>
      <td>2134</td>
      <td>2824</td>
      <td>16.493340</td>
      <td>48.663360</td>
      <td>...</td>
      <td>47.709600</td>
      <td>4</td>
      <td>7</td>
      <td>28</td>
      <td>15</td>
      <td>21</td>
      <td>28</td>
      <td>2</td>
      <td>2</td>
      <td>0.001580</td>
    </tr>
    <tr>
      <th>6445038</th>
      <td>11213126</td>
      <td>100</td>
      <td>2017-08-31 13:30:50</td>
      <td>345207</td>
      <td>57</td>
      <td>True</td>
      <td>1210</td>
      <td>1210</td>
      <td>16.380650</td>
      <td>48.290550</td>
      <td>...</td>
      <td>48.290550</td>
      <td>3</td>
      <td>8</td>
      <td>31</td>
      <td>13</td>
      <td>12</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0.518508</td>
    </tr>
    <tr>
      <th>4958339</th>
      <td>26302</td>
      <td>20</td>
      <td>2017-03-17 23:12:48</td>
      <td>29318</td>
      <td>24</td>
      <td>True</td>
      <td>1100</td>
      <td>1110</td>
      <td>16.387800</td>
      <td>48.152100</td>
      <td>...</td>
      <td>48.164000</td>
      <td>4</td>
      <td>3</td>
      <td>17</td>
      <td>23</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3824207</th>
      <td>4491224</td>
      <td>300</td>
      <td>2017-08-13 08:34:42</td>
      <td>95764</td>
      <td>17</td>
      <td>True</td>
      <td>1120</td>
      <td>1120</td>
      <td>16.322300</td>
      <td>48.170500</td>
      <td>...</td>
      <td>48.170500</td>
      <td>6</td>
      <td>8</td>
      <td>13</td>
      <td>8</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.357642</td>
    </tr>
    <tr>
      <th>7421215</th>
      <td>12551</td>
      <td>50</td>
      <td>2017-04-05 12:47:54</td>
      <td>125717</td>
      <td>45</td>
      <td>False</td>
      <td>4030</td>
      <td>4020</td>
      <td>14.286100</td>
      <td>48.306400</td>
      <td>...</td>
      <td>48.304100</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>12</td>
      <td>40</td>
      <td>40</td>
      <td>4</td>
      <td>4</td>
      <td>0.067580</td>
    </tr>
    <tr>
      <th>7420480</th>
      <td>40523</td>
      <td>20</td>
      <td>2017-05-22 17:30:32</td>
      <td>395777</td>
      <td>33</td>
      <td>True</td>
      <td>4030</td>
      <td>4151</td>
      <td>14.286100</td>
      <td>48.306400</td>
      <td>...</td>
      <td>48.588591</td>
      <td>0</td>
      <td>5</td>
      <td>22</td>
      <td>17</td>
      <td>40</td>
      <td>41</td>
      <td>4</td>
      <td>4</td>
      <td>0.067580</td>
    </tr>
    <tr>
      <th>2770727</th>
      <td>17452</td>
      <td>30</td>
      <td>2017-06-17 08:49:08</td>
      <td>216961</td>
      <td>43</td>
      <td>False</td>
      <td>1190</td>
      <td>1190</td>
      <td>16.333900</td>
      <td>48.259100</td>
      <td>...</td>
      <td>48.259100</td>
      <td>5</td>
      <td>6</td>
      <td>17</td>
      <td>8</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.197179</td>
    </tr>
    <tr>
      <th>4024431</th>
      <td>27763</td>
      <td>70</td>
      <td>2017-02-15 09:05:26</td>
      <td>250680</td>
      <td>64</td>
      <td>False</td>
      <td>1120</td>
      <td>1120</td>
      <td>16.322300</td>
      <td>48.170500</td>
      <td>...</td>
      <td>48.170500</td>
      <td>2</td>
      <td>2</td>
      <td>15</td>
      <td>9</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.357642</td>
    </tr>
    <tr>
      <th>11535543</th>
      <td>S6EE0500</td>
      <td>20</td>
      <td>2017-09-18 08:30:10</td>
      <td>270077</td>
      <td>25</td>
      <td>True</td>
      <td>8750</td>
      <td>8720</td>
      <td>14.627008</td>
      <td>47.189085</td>
      <td>...</td>
      <td>47.235712</td>
      <td>0</td>
      <td>9</td>
      <td>18</td>
      <td>8</td>
      <td>87</td>
      <td>87</td>
      <td>8</td>
      <td>8</td>
      <td>0.010477</td>
    </tr>
    <tr>
      <th>8315056</th>
      <td>16300126</td>
      <td>1000</td>
      <td>2017-07-31 10:23:14</td>
      <td>244953</td>
      <td>89</td>
      <td>True</td>
      <td>6300</td>
      <td>6300</td>
      <td>12.061700</td>
      <td>47.489100</td>
      <td>...</td>
      <td>47.489100</td>
      <td>0</td>
      <td>7</td>
      <td>31</td>
      <td>10</td>
      <td>63</td>
      <td>63</td>
      <td>6</td>
      <td>6</td>
      <td>0.030563</td>
    </tr>
    <tr>
      <th>5442655</th>
      <td>11110216</td>
      <td>500</td>
      <td>2017-04-11 20:16:49</td>
      <td>307945</td>
      <td>76</td>
      <td>True</td>
      <td>1110</td>
      <td>1110</td>
      <td>16.446300</td>
      <td>48.164000</td>
      <td>...</td>
      <td>48.164000</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>20</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.447086</td>
    </tr>
    <tr>
      <th>4952741</th>
      <td>11192</td>
      <td>70</td>
      <td>2017-06-07 19:10:35</td>
      <td>399255</td>
      <td>19</td>
      <td>True</td>
      <td>1100</td>
      <td>1030</td>
      <td>16.387800</td>
      <td>48.152100</td>
      <td>...</td>
      <td>48.198100</td>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>19</td>
      <td>11</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1331148</th>
      <td>11160146</td>
      <td>150</td>
      <td>2017-07-13 11:27:18</td>
      <td>150441</td>
      <td>47</td>
      <td>False</td>
      <td>1160</td>
      <td>1160</td>
      <td>16.300000</td>
      <td>48.216700</td>
      <td>...</td>
      <td>48.216700</td>
      <td>3</td>
      <td>7</td>
      <td>13</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.524514</td>
    </tr>
    <tr>
      <th>2869953</th>
      <td>27233</td>
      <td>200</td>
      <td>2017-12-02 17:39:02</td>
      <td>289885</td>
      <td>39</td>
      <td>False</td>
      <td>1030</td>
      <td>1030</td>
      <td>16.394800</td>
      <td>48.198100</td>
      <td>...</td>
      <td>48.198100</td>
      <td>5</td>
      <td>12</td>
      <td>2</td>
      <td>17</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0.305588</td>
    </tr>
    <tr>
      <th>1669507</th>
      <td>26212</td>
      <td>150</td>
      <td>2017-06-06 14:56:12</td>
      <td>207052</td>
      <td>19</td>
      <td>False</td>
      <td>1160</td>
      <td>1220</td>
      <td>16.300000</td>
      <td>48.216700</td>
      <td>...</td>
      <td>48.219000</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>14</td>
      <td>11</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0.524514</td>
    </tr>
    <tr>
      <th>8040874</th>
      <td>S6EE0863</td>
      <td>400</td>
      <td>2017-04-07 11:28:30</td>
      <td>355774</td>
      <td>70</td>
      <td>True</td>
      <td>2202</td>
      <td>3180</td>
      <td>16.415067</td>
      <td>48.345933</td>
      <td>...</td>
      <td>48.000900</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>11</td>
      <td>22</td>
      <td>31</td>
      <td>2</td>
      <td>3</td>
      <td>0.004455</td>
    </tr>
    <tr>
      <th>11033073</th>
      <td>19771</td>
      <td>100</td>
      <td>2017-08-12 14:13:26</td>
      <td>159941</td>
      <td>66</td>
      <td>True</td>
      <td>5500</td>
      <td>5201</td>
      <td>13.216667</td>
      <td>47.417778</td>
      <td>...</td>
      <td>47.918295</td>
      <td>5</td>
      <td>8</td>
      <td>12</td>
      <td>14</td>
      <td>55</td>
      <td>52</td>
      <td>5</td>
      <td>5</td>
      <td>0.040045</td>
    </tr>
    <tr>
      <th>9242306</th>
      <td>S6EE6060</td>
      <td>200</td>
      <td>2017-07-19 11:05:56</td>
      <td>106925</td>
      <td>65</td>
      <td>True</td>
      <td>4040</td>
      <td>4040</td>
      <td>14.270133</td>
      <td>48.349193</td>
      <td>...</td>
      <td>48.349193</td>
      <td>2</td>
      <td>7</td>
      <td>19</td>
      <td>11</td>
      <td>40</td>
      <td>40</td>
      <td>4</td>
      <td>4</td>
      <td>0.082254</td>
    </tr>
    <tr>
      <th>2852148</th>
      <td>11154126</td>
      <td>10</td>
      <td>2017-12-22 08:27:13</td>
      <td>200956</td>
      <td>23</td>
      <td>False</td>
      <td>1030</td>
      <td>1150</td>
      <td>16.394800</td>
      <td>48.198100</td>
      <td>...</td>
      <td>48.196000</td>
      <td>4</td>
      <td>12</td>
      <td>22</td>
      <td>8</td>
      <td>10</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.305588</td>
    </tr>
    <tr>
      <th>2257920</th>
      <td>27333</td>
      <td>90</td>
      <td>2017-05-22 08:36:58</td>
      <td>374043</td>
      <td>55</td>
      <td>False</td>
      <td>1020</td>
      <td>1020</td>
      <td>16.400000</td>
      <td>48.216700</td>
      <td>...</td>
      <td>48.216700</td>
      <td>0</td>
      <td>5</td>
      <td>22</td>
      <td>8</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0.491521</td>
    </tr>
    <tr>
      <th>6319002</th>
      <td>20263</td>
      <td>50</td>
      <td>2017-02-23 12:10:42</td>
      <td>286609</td>
      <td>21</td>
      <td>True</td>
      <td>1200</td>
      <td>1190</td>
      <td>16.377300</td>
      <td>48.240200</td>
      <td>...</td>
      <td>48.259100</td>
      <td>3</td>
      <td>2</td>
      <td>23</td>
      <td>12</td>
      <td>12</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.298284</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>10597511</th>
      <td>S6EE6047</td>
      <td>500</td>
      <td>2017-05-31 09:40:01</td>
      <td>275223</td>
      <td>49</td>
      <td>False</td>
      <td>2192</td>
      <td>2130</td>
      <td>16.650000</td>
      <td>48.550000</td>
      <td>...</td>
      <td>48.572589</td>
      <td>2</td>
      <td>5</td>
      <td>31</td>
      <td>9</td>
      <td>21</td>
      <td>21</td>
      <td>2</td>
      <td>2</td>
      <td>0.001975</td>
    </tr>
    <tr>
      <th>11454709</th>
      <td>63643</td>
      <td>50</td>
      <td>2017-11-10 18:54:14</td>
      <td>251843</td>
      <td>22</td>
      <td>False</td>
      <td>6491</td>
      <td>6450</td>
      <td>10.657400</td>
      <td>47.196700</td>
      <td>...</td>
      <td>46.923025</td>
      <td>4</td>
      <td>11</td>
      <td>10</td>
      <td>18</td>
      <td>64</td>
      <td>64</td>
      <td>6</td>
      <td>6</td>
      <td>0.000890</td>
    </tr>
    <tr>
      <th>13203974</th>
      <td>S6EE0032</td>
      <td>290</td>
      <td>2017-03-27 11:30:26</td>
      <td>15377</td>
      <td>56</td>
      <td>False</td>
      <td>9170</td>
      <td>9170</td>
      <td>14.348940</td>
      <td>46.512680</td>
      <td>...</td>
      <td>46.512680</td>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>11</td>
      <td>91</td>
      <td>91</td>
      <td>9</td>
      <td>9</td>
      <td>0.003562</td>
    </tr>
    <tr>
      <th>11888845</th>
      <td>38501</td>
      <td>100</td>
      <td>2017-05-12 08:24:50</td>
      <td>93148</td>
      <td>54</td>
      <td>False</td>
      <td>3542</td>
      <td>3552</td>
      <td>15.475905</td>
      <td>48.521500</td>
      <td>...</td>
      <td>48.471175</td>
      <td>4</td>
      <td>5</td>
      <td>12</td>
      <td>8</td>
      <td>35</td>
      <td>35</td>
      <td>3</td>
      <td>3</td>
      <td>0.002418</td>
    </tr>
    <tr>
      <th>11265662</th>
      <td>19300126</td>
      <td>200</td>
      <td>2017-12-02 10:22:21</td>
      <td>299781</td>
      <td>63</td>
      <td>True</td>
      <td>6421</td>
      <td>9300</td>
      <td>11.050000</td>
      <td>47.283300</td>
      <td>...</td>
      <td>46.778014</td>
      <td>5</td>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>64</td>
      <td>93</td>
      <td>6</td>
      <td>9</td>
      <td>0.002878</td>
    </tr>
    <tr>
      <th>5463519</th>
      <td>11152136</td>
      <td>100</td>
      <td>2017-10-31 17:30:16</td>
      <td>270913</td>
      <td>47</td>
      <td>False</td>
      <td>1170</td>
      <td>1150</td>
      <td>16.290100</td>
      <td>48.233800</td>
      <td>...</td>
      <td>48.196000</td>
      <td>1</td>
      <td>10</td>
      <td>31</td>
      <td>17</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.193003</td>
    </tr>
    <tr>
      <th>10137765</th>
      <td>48391</td>
      <td>400</td>
      <td>2017-10-23 15:54:26</td>
      <td>217004</td>
      <td>53</td>
      <td>True</td>
      <td>9020</td>
      <td>9020</td>
      <td>14.353896</td>
      <td>46.650609</td>
      <td>...</td>
      <td>46.650609</td>
      <td>0</td>
      <td>10</td>
      <td>23</td>
      <td>15</td>
      <td>90</td>
      <td>90</td>
      <td>9</td>
      <td>9</td>
      <td>0.231106</td>
    </tr>
    <tr>
      <th>2428582</th>
      <td>S6EE6044</td>
      <td>400</td>
      <td>2017-05-24 20:31:04</td>
      <td>347933</td>
      <td>58</td>
      <td>False</td>
      <td>1230</td>
      <td>1230</td>
      <td>16.293400</td>
      <td>48.143300</td>
      <td>...</td>
      <td>48.143300</td>
      <td>2</td>
      <td>5</td>
      <td>24</td>
      <td>20</td>
      <td>12</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0.235984</td>
    </tr>
    <tr>
      <th>10027082</th>
      <td>57663</td>
      <td>40</td>
      <td>2017-11-25 14:07:52</td>
      <td>343890</td>
      <td>44</td>
      <td>True</td>
      <td>5020</td>
      <td>5110</td>
      <td>13.044000</td>
      <td>47.799400</td>
      <td>...</td>
      <td>47.956833</td>
      <td>5</td>
      <td>11</td>
      <td>25</td>
      <td>14</td>
      <td>50</td>
      <td>51</td>
      <td>5</td>
      <td>5</td>
      <td>0.240393</td>
    </tr>
    <tr>
      <th>3810559</th>
      <td>48791</td>
      <td>200</td>
      <td>2017-05-05 19:42:01</td>
      <td>395312</td>
      <td>48</td>
      <td>False</td>
      <td>1120</td>
      <td>1120</td>
      <td>16.322300</td>
      <td>48.170500</td>
      <td>...</td>
      <td>48.170500</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>19</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.357642</td>
    </tr>
    <tr>
      <th>1615677</th>
      <td>11160136</td>
      <td>20</td>
      <td>2017-08-17 14:55:31</td>
      <td>227747</td>
      <td>22</td>
      <td>False</td>
      <td>1160</td>
      <td>1160</td>
      <td>16.300000</td>
      <td>48.216700</td>
      <td>...</td>
      <td>48.216700</td>
      <td>3</td>
      <td>8</td>
      <td>17</td>
      <td>14</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.524514</td>
    </tr>
    <tr>
      <th>215070</th>
      <td>4490604</td>
      <td>20</td>
      <td>2017-02-13 14:35:43</td>
      <td>59898</td>
      <td>54</td>
      <td>False</td>
      <td>1150</td>
      <td>1060</td>
      <td>16.318300</td>
      <td>48.196000</td>
      <td>...</td>
      <td>48.195200</td>
      <td>0</td>
      <td>2</td>
      <td>13</td>
      <td>14</td>
      <td>11</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0.386007</td>
    </tr>
    <tr>
      <th>5266410</th>
      <td>11115216</td>
      <td>20</td>
      <td>2017-04-05 13:40:02</td>
      <td>55749</td>
      <td>20</td>
      <td>True</td>
      <td>1110</td>
      <td>1110</td>
      <td>16.446300</td>
      <td>48.164000</td>
      <td>...</td>
      <td>48.164000</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.447086</td>
    </tr>
    <tr>
      <th>10672314</th>
      <td>S6EE0221</td>
      <td>50</td>
      <td>2017-04-22 14:21:26</td>
      <td>77405</td>
      <td>56</td>
      <td>False</td>
      <td>4050</td>
      <td>4050</td>
      <td>14.241740</td>
      <td>48.218180</td>
      <td>...</td>
      <td>48.218180</td>
      <td>5</td>
      <td>4</td>
      <td>22</td>
      <td>14</td>
      <td>40</td>
      <td>40</td>
      <td>4</td>
      <td>4</td>
      <td>0.048904</td>
    </tr>
    <tr>
      <th>13632772</th>
      <td>82013</td>
      <td>400</td>
      <td>2017-06-07 16:44:30</td>
      <td>22209</td>
      <td>33</td>
      <td>True</td>
      <td>8144</td>
      <td>8192</td>
      <td>15.377985</td>
      <td>46.967123</td>
      <td>...</td>
      <td>47.412475</td>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>16</td>
      <td>81</td>
      <td>81</td>
      <td>8</td>
      <td>8</td>
      <td>0.002066</td>
    </tr>
    <tr>
      <th>13114778</th>
      <td>1683314</td>
      <td>300</td>
      <td>2017-07-01 20:48:26</td>
      <td>88463</td>
      <td>36</td>
      <td>True</td>
      <td>8580</td>
      <td>8570</td>
      <td>15.072854</td>
      <td>47.038992</td>
      <td>...</td>
      <td>47.045913</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>20</td>
      <td>85</td>
      <td>85</td>
      <td>8</td>
      <td>8</td>
      <td>0.017297</td>
    </tr>
    <tr>
      <th>13573109</th>
      <td>82533</td>
      <td>20</td>
      <td>2017-03-24 13:52:27</td>
      <td>46902</td>
      <td>51</td>
      <td>False</td>
      <td>8630</td>
      <td>8911</td>
      <td>15.330160</td>
      <td>47.767300</td>
      <td>...</td>
      <td>47.577320</td>
      <td>4</td>
      <td>3</td>
      <td>24</td>
      <td>13</td>
      <td>86</td>
      <td>89</td>
      <td>8</td>
      <td>8</td>
      <td>0.007049</td>
    </tr>
    <tr>
      <th>4059149</th>
      <td>11104116</td>
      <td>90</td>
      <td>2017-07-13 15:13:59</td>
      <td>365186</td>
      <td>42</td>
      <td>True</td>
      <td>1100</td>
      <td>1100</td>
      <td>16.387800</td>
      <td>48.152100</td>
      <td>...</td>
      <td>48.152100</td>
      <td>3</td>
      <td>7</td>
      <td>13</td>
      <td>15</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3077666</th>
      <td>11006126</td>
      <td>1000</td>
      <td>2017-11-02 15:14:04</td>
      <td>37058</td>
      <td>33</td>
      <td>False</td>
      <td>1030</td>
      <td>1030</td>
      <td>16.394800</td>
      <td>48.198100</td>
      <td>...</td>
      <td>48.198100</td>
      <td>3</td>
      <td>11</td>
      <td>2</td>
      <td>15</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0.305588</td>
    </tr>
    <tr>
      <th>4304487</th>
      <td>11810216</td>
      <td>100</td>
      <td>2017-06-21 11:17:54</td>
      <td>338543</td>
      <td>58</td>
      <td>False</td>
      <td>1100</td>
      <td>1100</td>
      <td>16.387800</td>
      <td>48.152100</td>
      <td>...</td>
      <td>48.152100</td>
      <td>2</td>
      <td>6</td>
      <td>21</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9851558</th>
      <td>57793</td>
      <td>400</td>
      <td>2017-11-27 10:49:19</td>
      <td>114897</td>
      <td>41</td>
      <td>True</td>
      <td>5020</td>
      <td>5020</td>
      <td>13.044000</td>
      <td>47.799400</td>
      <td>...</td>
      <td>47.799400</td>
      <td>0</td>
      <td>11</td>
      <td>27</td>
      <td>10</td>
      <td>50</td>
      <td>50</td>
      <td>5</td>
      <td>5</td>
      <td>0.240393</td>
    </tr>
    <tr>
      <th>9218390</th>
      <td>14013216</td>
      <td>400</td>
      <td>2017-04-05 10:10:16</td>
      <td>155385</td>
      <td>52</td>
      <td>False</td>
      <td>4040</td>
      <td>4020</td>
      <td>14.270133</td>
      <td>48.349193</td>
      <td>...</td>
      <td>48.304100</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>10</td>
      <td>40</td>
      <td>40</td>
      <td>4</td>
      <td>4</td>
      <td>0.082254</td>
    </tr>
    <tr>
      <th>1758663</th>
      <td>4472014</td>
      <td>50</td>
      <td>2017-10-26 18:00:09</td>
      <td>154520</td>
      <td>34</td>
      <td>False</td>
      <td>1160</td>
      <td>1020</td>
      <td>16.300000</td>
      <td>48.216700</td>
      <td>...</td>
      <td>48.216700</td>
      <td>3</td>
      <td>10</td>
      <td>26</td>
      <td>18</td>
      <td>11</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0.524514</td>
    </tr>
    <tr>
      <th>12349301</th>
      <td>84383</td>
      <td>160</td>
      <td>2017-12-06 10:45:19</td>
      <td>368828</td>
      <td>47</td>
      <td>False</td>
      <td>8101</td>
      <td>8042</td>
      <td>15.360240</td>
      <td>47.158720</td>
      <td>...</td>
      <td>47.135200</td>
      <td>2</td>
      <td>12</td>
      <td>6</td>
      <td>10</td>
      <td>81</td>
      <td>80</td>
      <td>8</td>
      <td>8</td>
      <td>0.013128</td>
    </tr>
    <tr>
      <th>12257189</th>
      <td>S6EE0500</td>
      <td>10</td>
      <td>2017-02-07 05:13:12</td>
      <td>252673</td>
      <td>30</td>
      <td>False</td>
      <td>8761</td>
      <td>8720</td>
      <td>14.586707</td>
      <td>47.231293</td>
      <td>...</td>
      <td>47.235712</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>5</td>
      <td>87</td>
      <td>87</td>
      <td>8</td>
      <td>8</td>
      <td>0.000975</td>
    </tr>
    <tr>
      <th>9962207</th>
      <td>S6EE0799</td>
      <td>10</td>
      <td>2017-06-19 10:19:23</td>
      <td>232133</td>
      <td>16</td>
      <td>False</td>
      <td>5020</td>
      <td>5020</td>
      <td>13.044000</td>
      <td>47.799400</td>
      <td>...</td>
      <td>47.799400</td>
      <td>0</td>
      <td>6</td>
      <td>19</td>
      <td>10</td>
      <td>50</td>
      <td>50</td>
      <td>5</td>
      <td>5</td>
      <td>0.240393</td>
    </tr>
    <tr>
      <th>12897455</th>
      <td>S6EE6060</td>
      <td>100</td>
      <td>2017-06-18 10:42:58</td>
      <td>387387</td>
      <td>46</td>
      <td>False</td>
      <td>4090</td>
      <td>4040</td>
      <td>13.646200</td>
      <td>48.518988</td>
      <td>...</td>
      <td>48.349193</td>
      <td>6</td>
      <td>6</td>
      <td>18</td>
      <td>10</td>
      <td>40</td>
      <td>40</td>
      <td>4</td>
      <td>4</td>
      <td>0.003206</td>
    </tr>
    <tr>
      <th>11697003</th>
      <td>4473634</td>
      <td>10</td>
      <td>2017-12-12 13:18:27</td>
      <td>349335</td>
      <td>33</td>
      <td>False</td>
      <td>7533</td>
      <td>2700</td>
      <td>16.166700</td>
      <td>47.183300</td>
      <td>...</td>
      <td>47.810260</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>13</td>
      <td>75</td>
      <td>27</td>
      <td>7</td>
      <td>2</td>
      <td>0.001352</td>
    </tr>
    <tr>
      <th>5045442</th>
      <td>11113126</td>
      <td>300</td>
      <td>2017-02-02 12:58:55</td>
      <td>72428</td>
      <td>75</td>
      <td>False</td>
      <td>1110</td>
      <td>1110</td>
      <td>16.446300</td>
      <td>48.164000</td>
      <td>...</td>
      <td>48.164000</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.447086</td>
    </tr>
    <tr>
      <th>13456631</th>
      <td>18572216</td>
      <td>10</td>
      <td>2017-07-01 21:46:07</td>
      <td>275903</td>
      <td>70</td>
      <td>True</td>
      <td>8561</td>
      <td>8572</td>
      <td>15.269800</td>
      <td>47.005612</td>
      <td>...</td>
      <td>47.084750</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>21</td>
      <td>85</td>
      <td>85</td>
      <td>8</td>
      <td>8</td>
      <td>0.002542</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 21 columns</p>
</div>




```python
#data=pd.merge(data, data.groupby("PLZ").size().to_frame('size'), on="PLZ")
#data.size=data["size"]/max(data["size"])
#data=pd.merge(data,data.groupby('PLZ')['ALTER'].mean().to_frame('alter_mean'), on="PLZ")

```


```python
import pandas as pd

source = pd.DataFrame({'Country' : ['USA', 'USA', 'Russia','USA'], 
                  'City' : ['New-York', 'New-York', 'Sankt-Petersburg', 'New-York'],
                  'Short name' : ['NY','New','Spb','NY']})

source.groupby(['Country','City']).agg(lambda x:x.value_counts().index[0])
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
      <th></th>
      <th>Short name</th>
    </tr>
    <tr>
      <th>Country</th>
      <th>City</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Russia</th>
      <th>Sankt-Petersburg</th>
      <td>Spb</td>
    </tr>
    <tr>
      <th>USA</th>
      <th>New-York</th>
      <td>NY</td>
    </tr>
  </tbody>
</table>
</div>




```python
from gcmap import GCMapper, Gradient

```




    1




```python

```
