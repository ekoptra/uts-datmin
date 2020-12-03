# UTS Data Mining - Neural Network With Python

- Nama : Eko Putra Wahyuddin
- NIM : 221810259
- No Absen : 9
- Kelas : 3SD1
- Dosen : Ibnu Santoso SST, MT
- Mata Kuliah : Data Mining and Knowledge Management
- Hari/Tanggal : Selasa, 27 Oktober 2020

> ### "Saya menyatakan bahwa ujian ini saya kerjakan dengan juju sesuai kemampuan sendiri dan tidak mengutip sebagian atau seluruh pekerjaan orang lain. Jika suatu saat ditemukan saya melanggar ketentuan ujian, saya siap menerima konsekuensi yang berlaku”

TTD

Eko Putra Wahyuddin


### Import Library 


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
```

### Baca Data


```python
df = pd.read_csv("parkinsons.data")
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>...</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>status</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phon_R01_S01_1</td>
      <td>119.992</td>
      <td>157.302</td>
      <td>74.997</td>
      <td>0.00784</td>
      <td>0.00007</td>
      <td>0.00370</td>
      <td>0.00554</td>
      <td>0.01109</td>
      <td>0.04374</td>
      <td>...</td>
      <td>0.06545</td>
      <td>0.02211</td>
      <td>21.033</td>
      <td>1</td>
      <td>0.414783</td>
      <td>0.815285</td>
      <td>-4.813031</td>
      <td>0.266482</td>
      <td>2.301442</td>
      <td>0.284654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phon_R01_S01_2</td>
      <td>122.400</td>
      <td>148.650</td>
      <td>113.819</td>
      <td>0.00968</td>
      <td>0.00008</td>
      <td>0.00465</td>
      <td>0.00696</td>
      <td>0.01394</td>
      <td>0.06134</td>
      <td>...</td>
      <td>0.09403</td>
      <td>0.01929</td>
      <td>19.085</td>
      <td>1</td>
      <td>0.458359</td>
      <td>0.819521</td>
      <td>-4.075192</td>
      <td>0.335590</td>
      <td>2.486855</td>
      <td>0.368674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phon_R01_S01_3</td>
      <td>116.682</td>
      <td>131.111</td>
      <td>111.555</td>
      <td>0.01050</td>
      <td>0.00009</td>
      <td>0.00544</td>
      <td>0.00781</td>
      <td>0.01633</td>
      <td>0.05233</td>
      <td>...</td>
      <td>0.08270</td>
      <td>0.01309</td>
      <td>20.651</td>
      <td>1</td>
      <td>0.429895</td>
      <td>0.825288</td>
      <td>-4.443179</td>
      <td>0.311173</td>
      <td>2.342259</td>
      <td>0.332634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phon_R01_S01_4</td>
      <td>116.676</td>
      <td>137.871</td>
      <td>111.366</td>
      <td>0.00997</td>
      <td>0.00009</td>
      <td>0.00502</td>
      <td>0.00698</td>
      <td>0.01505</td>
      <td>0.05492</td>
      <td>...</td>
      <td>0.08771</td>
      <td>0.01353</td>
      <td>20.644</td>
      <td>1</td>
      <td>0.434969</td>
      <td>0.819235</td>
      <td>-4.117501</td>
      <td>0.334147</td>
      <td>2.405554</td>
      <td>0.368975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phon_R01_S01_5</td>
      <td>116.014</td>
      <td>141.781</td>
      <td>110.655</td>
      <td>0.01284</td>
      <td>0.00011</td>
      <td>0.00655</td>
      <td>0.00908</td>
      <td>0.01966</td>
      <td>0.06425</td>
      <td>...</td>
      <td>0.10470</td>
      <td>0.01767</td>
      <td>19.649</td>
      <td>1</td>
      <td>0.417356</td>
      <td>0.823484</td>
      <td>-3.747787</td>
      <td>0.234513</td>
      <td>2.332180</td>
      <td>0.410335</td>
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
      <th>190</th>
      <td>phon_R01_S50_2</td>
      <td>174.188</td>
      <td>230.978</td>
      <td>94.261</td>
      <td>0.00459</td>
      <td>0.00003</td>
      <td>0.00263</td>
      <td>0.00259</td>
      <td>0.00790</td>
      <td>0.04087</td>
      <td>...</td>
      <td>0.07008</td>
      <td>0.02764</td>
      <td>19.517</td>
      <td>0</td>
      <td>0.448439</td>
      <td>0.657899</td>
      <td>-6.538586</td>
      <td>0.121952</td>
      <td>2.657476</td>
      <td>0.133050</td>
    </tr>
    <tr>
      <th>191</th>
      <td>phon_R01_S50_3</td>
      <td>209.516</td>
      <td>253.017</td>
      <td>89.488</td>
      <td>0.00564</td>
      <td>0.00003</td>
      <td>0.00331</td>
      <td>0.00292</td>
      <td>0.00994</td>
      <td>0.02751</td>
      <td>...</td>
      <td>0.04812</td>
      <td>0.01810</td>
      <td>19.147</td>
      <td>0</td>
      <td>0.431674</td>
      <td>0.683244</td>
      <td>-6.195325</td>
      <td>0.129303</td>
      <td>2.784312</td>
      <td>0.168895</td>
    </tr>
    <tr>
      <th>192</th>
      <td>phon_R01_S50_4</td>
      <td>174.688</td>
      <td>240.005</td>
      <td>74.287</td>
      <td>0.01360</td>
      <td>0.00008</td>
      <td>0.00624</td>
      <td>0.00564</td>
      <td>0.01873</td>
      <td>0.02308</td>
      <td>...</td>
      <td>0.03804</td>
      <td>0.10715</td>
      <td>17.883</td>
      <td>0</td>
      <td>0.407567</td>
      <td>0.655683</td>
      <td>-6.787197</td>
      <td>0.158453</td>
      <td>2.679772</td>
      <td>0.131728</td>
    </tr>
    <tr>
      <th>193</th>
      <td>phon_R01_S50_5</td>
      <td>198.764</td>
      <td>396.961</td>
      <td>74.904</td>
      <td>0.00740</td>
      <td>0.00004</td>
      <td>0.00370</td>
      <td>0.00390</td>
      <td>0.01109</td>
      <td>0.02296</td>
      <td>...</td>
      <td>0.03794</td>
      <td>0.07223</td>
      <td>19.020</td>
      <td>0</td>
      <td>0.451221</td>
      <td>0.643956</td>
      <td>-6.744577</td>
      <td>0.207454</td>
      <td>2.138608</td>
      <td>0.123306</td>
    </tr>
    <tr>
      <th>194</th>
      <td>phon_R01_S50_6</td>
      <td>214.289</td>
      <td>260.277</td>
      <td>77.973</td>
      <td>0.00567</td>
      <td>0.00003</td>
      <td>0.00295</td>
      <td>0.00317</td>
      <td>0.00885</td>
      <td>0.01884</td>
      <td>...</td>
      <td>0.03078</td>
      <td>0.04398</td>
      <td>21.209</td>
      <td>0</td>
      <td>0.462803</td>
      <td>0.664357</td>
      <td>-5.724056</td>
      <td>0.190667</td>
      <td>2.555477</td>
      <td>0.148569</td>
    </tr>
  </tbody>
</table>
<p>195 rows × 24 columns</p>
</div>



Dari tampilan diatas diketahui bahwa data terdiri dari 195 baris dan 24 kolom. Target kelasnya adalah kolom **status** yang bernilai 1 untuk orang yang menderita Parkinson's dan 0 untuk orang yang sehat. Kolom name tidak akan saya gunakan karena itu hanya berisi informasi nama/kode untuk subjek

## Preprocessing dan Explorasi Data


```python
df = df.iloc[:, 1:]
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>MDVP:Shimmer(dB)</th>
      <th>Shimmer:APQ3</th>
      <th>...</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>status</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>157.302</td>
      <td>74.997</td>
      <td>0.00784</td>
      <td>0.00007</td>
      <td>0.00370</td>
      <td>0.00554</td>
      <td>0.01109</td>
      <td>0.04374</td>
      <td>0.426</td>
      <td>0.02182</td>
      <td>...</td>
      <td>0.06545</td>
      <td>0.02211</td>
      <td>21.033</td>
      <td>1</td>
      <td>0.414783</td>
      <td>0.815285</td>
      <td>-4.813031</td>
      <td>0.266482</td>
      <td>2.301442</td>
      <td>0.284654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>148.650</td>
      <td>113.819</td>
      <td>0.00968</td>
      <td>0.00008</td>
      <td>0.00465</td>
      <td>0.00696</td>
      <td>0.01394</td>
      <td>0.06134</td>
      <td>0.626</td>
      <td>0.03134</td>
      <td>...</td>
      <td>0.09403</td>
      <td>0.01929</td>
      <td>19.085</td>
      <td>1</td>
      <td>0.458359</td>
      <td>0.819521</td>
      <td>-4.075192</td>
      <td>0.335590</td>
      <td>2.486855</td>
      <td>0.368674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131.111</td>
      <td>111.555</td>
      <td>0.01050</td>
      <td>0.00009</td>
      <td>0.00544</td>
      <td>0.00781</td>
      <td>0.01633</td>
      <td>0.05233</td>
      <td>0.482</td>
      <td>0.02757</td>
      <td>...</td>
      <td>0.08270</td>
      <td>0.01309</td>
      <td>20.651</td>
      <td>1</td>
      <td>0.429895</td>
      <td>0.825288</td>
      <td>-4.443179</td>
      <td>0.311173</td>
      <td>2.342259</td>
      <td>0.332634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>137.871</td>
      <td>111.366</td>
      <td>0.00997</td>
      <td>0.00009</td>
      <td>0.00502</td>
      <td>0.00698</td>
      <td>0.01505</td>
      <td>0.05492</td>
      <td>0.517</td>
      <td>0.02924</td>
      <td>...</td>
      <td>0.08771</td>
      <td>0.01353</td>
      <td>20.644</td>
      <td>1</td>
      <td>0.434969</td>
      <td>0.819235</td>
      <td>-4.117501</td>
      <td>0.334147</td>
      <td>2.405554</td>
      <td>0.368975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>141.781</td>
      <td>110.655</td>
      <td>0.01284</td>
      <td>0.00011</td>
      <td>0.00655</td>
      <td>0.00908</td>
      <td>0.01966</td>
      <td>0.06425</td>
      <td>0.584</td>
      <td>0.03490</td>
      <td>...</td>
      <td>0.10470</td>
      <td>0.01767</td>
      <td>19.649</td>
      <td>1</td>
      <td>0.417356</td>
      <td>0.823484</td>
      <td>-3.747787</td>
      <td>0.234513</td>
      <td>2.332180</td>
      <td>0.410335</td>
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
      <th>190</th>
      <td>230.978</td>
      <td>94.261</td>
      <td>0.00459</td>
      <td>0.00003</td>
      <td>0.00263</td>
      <td>0.00259</td>
      <td>0.00790</td>
      <td>0.04087</td>
      <td>0.405</td>
      <td>0.02336</td>
      <td>...</td>
      <td>0.07008</td>
      <td>0.02764</td>
      <td>19.517</td>
      <td>0</td>
      <td>0.448439</td>
      <td>0.657899</td>
      <td>-6.538586</td>
      <td>0.121952</td>
      <td>2.657476</td>
      <td>0.133050</td>
    </tr>
    <tr>
      <th>191</th>
      <td>253.017</td>
      <td>89.488</td>
      <td>0.00564</td>
      <td>0.00003</td>
      <td>0.00331</td>
      <td>0.00292</td>
      <td>0.00994</td>
      <td>0.02751</td>
      <td>0.263</td>
      <td>0.01604</td>
      <td>...</td>
      <td>0.04812</td>
      <td>0.01810</td>
      <td>19.147</td>
      <td>0</td>
      <td>0.431674</td>
      <td>0.683244</td>
      <td>-6.195325</td>
      <td>0.129303</td>
      <td>2.784312</td>
      <td>0.168895</td>
    </tr>
    <tr>
      <th>192</th>
      <td>240.005</td>
      <td>74.287</td>
      <td>0.01360</td>
      <td>0.00008</td>
      <td>0.00624</td>
      <td>0.00564</td>
      <td>0.01873</td>
      <td>0.02308</td>
      <td>0.256</td>
      <td>0.01268</td>
      <td>...</td>
      <td>0.03804</td>
      <td>0.10715</td>
      <td>17.883</td>
      <td>0</td>
      <td>0.407567</td>
      <td>0.655683</td>
      <td>-6.787197</td>
      <td>0.158453</td>
      <td>2.679772</td>
      <td>0.131728</td>
    </tr>
    <tr>
      <th>193</th>
      <td>396.961</td>
      <td>74.904</td>
      <td>0.00740</td>
      <td>0.00004</td>
      <td>0.00370</td>
      <td>0.00390</td>
      <td>0.01109</td>
      <td>0.02296</td>
      <td>0.241</td>
      <td>0.01265</td>
      <td>...</td>
      <td>0.03794</td>
      <td>0.07223</td>
      <td>19.020</td>
      <td>0</td>
      <td>0.451221</td>
      <td>0.643956</td>
      <td>-6.744577</td>
      <td>0.207454</td>
      <td>2.138608</td>
      <td>0.123306</td>
    </tr>
    <tr>
      <th>194</th>
      <td>260.277</td>
      <td>77.973</td>
      <td>0.00567</td>
      <td>0.00003</td>
      <td>0.00295</td>
      <td>0.00317</td>
      <td>0.00885</td>
      <td>0.01884</td>
      <td>0.190</td>
      <td>0.01026</td>
      <td>...</td>
      <td>0.03078</td>
      <td>0.04398</td>
      <td>21.209</td>
      <td>0</td>
      <td>0.462803</td>
      <td>0.664357</td>
      <td>-5.724056</td>
      <td>0.190667</td>
      <td>2.555477</td>
      <td>0.148569</td>
    </tr>
  </tbody>
</table>
<p>195 rows × 22 columns</p>
</div>



### Cek Missing Value


```python
df.isna().sum()
```




    MDVP:Fhi(Hz)        0
    MDVP:Flo(Hz)        0
    MDVP:Jitter(%)      0
    MDVP:Jitter(Abs)    0
    MDVP:RAP            0
    MDVP:PPQ            0
    Jitter:DDP          0
    MDVP:Shimmer        0
    MDVP:Shimmer(dB)    0
    Shimmer:APQ3        0
    Shimmer:APQ5        0
    MDVP:APQ            0
    Shimmer:DDA         0
    NHR                 0
    HNR                 0
    status              0
    RPDE                0
    DFA                 0
    spread1             0
    spread2             0
    D2                  0
    PPE                 0
    dtype: int64



Terlihat bahwa tidak terdapat missing value pada data.

Pisah data menjadi feature class dan targat class. Feature class pada variabel `X` dan target class pada variabel `y`


```python
X = df.drop(["status"], axis =1)
y = df['status']
```

### Visualisasi Target Class


```python
count = y.value_counts()
plt.figure(figsize=(5,5))
plt.title("Target Class Distribution")
ax = count.plot.bar()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

for i, v in enumerate(count):
    ax.text(i - 0.08, v + 1 , str(v))

for tick in ax.get_xticklines():
    tick.set_visible(False)
for tick in ax.get_yticklines():
    tick.set_visible(False)
    
ax.set_yticklabels([])
ax.set_xticklabels(["Parkinson's", "Healty"], rotation='horizontal', size = 13);
```


![png](assets/output_13_0.png)


Dari Visualisasi tersebut terlihat bahwa data terdiri dari 147 orang yang terkena Parkinson's dan 48 yang sehat

### Summary Feature Class


```python
X.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>MDVP:Shimmer(dB)</th>
      <th>...</th>
      <th>MDVP:APQ</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>...</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>154.228641</td>
      <td>197.104918</td>
      <td>116.324631</td>
      <td>0.006220</td>
      <td>0.000044</td>
      <td>0.003306</td>
      <td>0.003446</td>
      <td>0.009920</td>
      <td>0.029709</td>
      <td>0.282251</td>
      <td>...</td>
      <td>0.024081</td>
      <td>0.046993</td>
      <td>0.024847</td>
      <td>21.885974</td>
      <td>0.498536</td>
      <td>0.718099</td>
      <td>-5.684397</td>
      <td>0.226510</td>
      <td>2.381826</td>
      <td>0.206552</td>
    </tr>
    <tr>
      <th>std</th>
      <td>41.390065</td>
      <td>91.491548</td>
      <td>43.521413</td>
      <td>0.004848</td>
      <td>0.000035</td>
      <td>0.002968</td>
      <td>0.002759</td>
      <td>0.008903</td>
      <td>0.018857</td>
      <td>0.194877</td>
      <td>...</td>
      <td>0.016947</td>
      <td>0.030459</td>
      <td>0.040418</td>
      <td>4.425764</td>
      <td>0.103942</td>
      <td>0.055336</td>
      <td>1.090208</td>
      <td>0.083406</td>
      <td>0.382799</td>
      <td>0.090119</td>
    </tr>
    <tr>
      <th>min</th>
      <td>88.333000</td>
      <td>102.145000</td>
      <td>65.476000</td>
      <td>0.001680</td>
      <td>0.000007</td>
      <td>0.000680</td>
      <td>0.000920</td>
      <td>0.002040</td>
      <td>0.009540</td>
      <td>0.085000</td>
      <td>...</td>
      <td>0.007190</td>
      <td>0.013640</td>
      <td>0.000650</td>
      <td>8.441000</td>
      <td>0.256570</td>
      <td>0.574282</td>
      <td>-7.964984</td>
      <td>0.006274</td>
      <td>1.423287</td>
      <td>0.044539</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>117.572000</td>
      <td>134.862500</td>
      <td>84.291000</td>
      <td>0.003460</td>
      <td>0.000020</td>
      <td>0.001660</td>
      <td>0.001860</td>
      <td>0.004985</td>
      <td>0.016505</td>
      <td>0.148500</td>
      <td>...</td>
      <td>0.013080</td>
      <td>0.024735</td>
      <td>0.005925</td>
      <td>19.198000</td>
      <td>0.421306</td>
      <td>0.674758</td>
      <td>-6.450096</td>
      <td>0.174351</td>
      <td>2.099125</td>
      <td>0.137451</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>148.790000</td>
      <td>175.829000</td>
      <td>104.315000</td>
      <td>0.004940</td>
      <td>0.000030</td>
      <td>0.002500</td>
      <td>0.002690</td>
      <td>0.007490</td>
      <td>0.022970</td>
      <td>0.221000</td>
      <td>...</td>
      <td>0.018260</td>
      <td>0.038360</td>
      <td>0.011660</td>
      <td>22.085000</td>
      <td>0.495954</td>
      <td>0.722254</td>
      <td>-5.720868</td>
      <td>0.218885</td>
      <td>2.361532</td>
      <td>0.194052</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>182.769000</td>
      <td>224.205500</td>
      <td>140.018500</td>
      <td>0.007365</td>
      <td>0.000060</td>
      <td>0.003835</td>
      <td>0.003955</td>
      <td>0.011505</td>
      <td>0.037885</td>
      <td>0.350000</td>
      <td>...</td>
      <td>0.029400</td>
      <td>0.060795</td>
      <td>0.025640</td>
      <td>25.075500</td>
      <td>0.587562</td>
      <td>0.761881</td>
      <td>-5.046192</td>
      <td>0.279234</td>
      <td>2.636456</td>
      <td>0.252980</td>
    </tr>
    <tr>
      <th>max</th>
      <td>260.105000</td>
      <td>592.030000</td>
      <td>239.170000</td>
      <td>0.033160</td>
      <td>0.000260</td>
      <td>0.021440</td>
      <td>0.019580</td>
      <td>0.064330</td>
      <td>0.119080</td>
      <td>1.302000</td>
      <td>...</td>
      <td>0.137780</td>
      <td>0.169420</td>
      <td>0.314820</td>
      <td>33.047000</td>
      <td>0.685151</td>
      <td>0.825288</td>
      <td>-2.434031</td>
      <td>0.450493</td>
      <td>3.671155</td>
      <td>0.527367</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>



Dari tabel diatas terlihat 5 Number Summary. Terlihat bahwa rata-rata dari 3 kolom pertama sangat berbeda jauh dengan kolom lainnya, sehingga perlu dilakukan scalling data agar 3 kolom pertama tidak membuat model menjadi bias.

### MinMax Scaler


```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
X_scaled.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>MDVP:Shimmer(dB)</th>
      <th>Shimmer:APQ3</th>
      <th>...</th>
      <th>MDVP:APQ</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>...</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.193841</td>
      <td>0.292748</td>
      <td>0.144233</td>
      <td>0.146083</td>
      <td>0.126513</td>
      <td>0.135389</td>
      <td>0.126504</td>
      <td>0.184126</td>
      <td>0.162080</td>
      <td>0.214063</td>
      <td>...</td>
      <td>0.129347</td>
      <td>0.214101</td>
      <td>0.077019</td>
      <td>0.546410</td>
      <td>0.564574</td>
      <td>0.572963</td>
      <td>0.412332</td>
      <td>0.495783</td>
      <td>0.426421</td>
      <td>0.335549</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.186761</td>
      <td>0.250564</td>
      <td>0.154007</td>
      <td>0.137636</td>
      <td>0.142956</td>
      <td>0.147855</td>
      <td>0.142934</td>
      <td>0.172147</td>
      <td>0.160129</td>
      <td>0.195554</td>
      <td>...</td>
      <td>0.129771</td>
      <td>0.195527</td>
      <td>0.128652</td>
      <td>0.179865</td>
      <td>0.242525</td>
      <td>0.220456</td>
      <td>0.197110</td>
      <td>0.187758</td>
      <td>0.170294</td>
      <td>0.186649</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.066786</td>
      <td>0.108323</td>
      <td>0.056544</td>
      <td>0.051383</td>
      <td>0.047206</td>
      <td>0.050375</td>
      <td>0.047279</td>
      <td>0.063584</td>
      <td>0.052177</td>
      <td>0.071167</td>
      <td>...</td>
      <td>0.045103</td>
      <td>0.071222</td>
      <td>0.016790</td>
      <td>0.437170</td>
      <td>0.384375</td>
      <td>0.400291</td>
      <td>0.273893</td>
      <td>0.378364</td>
      <td>0.300658</td>
      <td>0.192433</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.150411</td>
      <td>0.223606</td>
      <td>0.103558</td>
      <td>0.090909</td>
      <td>0.087669</td>
      <td>0.094855</td>
      <td>0.087494</td>
      <td>0.122604</td>
      <td>0.111750</td>
      <td>0.158706</td>
      <td>...</td>
      <td>0.084769</td>
      <td>0.158685</td>
      <td>0.035045</td>
      <td>0.554499</td>
      <td>0.558550</td>
      <td>0.589516</td>
      <td>0.405738</td>
      <td>0.478618</td>
      <td>0.417393</td>
      <td>0.309661</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.249162</td>
      <td>0.429160</td>
      <td>0.180591</td>
      <td>0.209486</td>
      <td>0.151975</td>
      <td>0.162647</td>
      <td>0.151951</td>
      <td>0.258764</td>
      <td>0.217749</td>
      <td>0.302677</td>
      <td>...</td>
      <td>0.170074</td>
      <td>0.302703</td>
      <td>0.079543</td>
      <td>0.676034</td>
      <td>0.772299</td>
      <td>0.747391</td>
      <td>0.527720</td>
      <td>0.614472</td>
      <td>0.539698</td>
      <td>0.431709</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>



### Cek Korelasi


```python
cor = X_scaled.corr()
plt.figure(figsize = (24,20))
ax = sns.heatmap(cor, annot = True, cbar=False)
plt.xticks(rotation = 90);
```


![png](assets/output_19_0.png)


Terlihat bahwa kolom HNR memiliki nilai korelasi yang tinggi dengan hampir semua kolom lainnya. Oleh karena itu saya memutuskan untuk tidak menggunakan kolom HNR dalam pembuatan model


```python
X_scaled = X_scaled.drop("HNR", axis = 1)
X_scaled.columns
```



    Index(['MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
           'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
           'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',
           'Shimmer:DDA', 'NHR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
          dtype='object')



## Buat Model

### Split Data


```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 0, train_size = 0.8)
print("Jumlah Training Data : ", len(X_train), " | Jumlah Test Data : ", len(X_test))
```

    Jumlah Training Data :  156  | Jumlah Test Data :  39
    

Saya membagi data menjadi 80% data training (156 baris) dan 20% data testing (39 baris)

Karena jumlah data yang lumayan sedikit maka saya memutuskan hanya menggunakan 5 hidden layer dimana setiap hidden layer memiliki 15 black box. Saya akan menggunakan 4 activation function, yaitu **relu, logistic, tanh, dan identity** dan membandingkan hasilnya

### Relu Activation Function


```python
layer_size = [5, 10]
activation_function = "relu"

clf_relu = MLPClassifier(hidden_layer_sizes = layer_size, random_state = 0, activation = activation_function)
clf_relu.fit(X_train, y_train)
```

    


    MLPClassifier(hidden_layer_sizes=[5, 10], random_state=0)




```python
y_pred = clf_relu.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

target_names = ["Healty", "Parkinson's"]

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, square=True, cbar=False, xticklabels = target_names, yticklabels = target_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Accuracy Score: {:.3}'.format(accuracy));
```


![png](assets/output_28_0.png)


Dari confusion matrix diatas diperoleh accuracy model sebesar 0.872. Terlihat juga tidak terdapat kesalahan (FN) dalam memprediksi seseorang yang mengalami Parkinson, yang berarti recall dari model ini sebesar 1. Akan tetapi model ini masih salah  dalam memprediksi 5 orang yang sehat, sehingga precision sebesar 0.85. Sementara itu F1 score dari model ini sebesar 87%


```python
print(classification_report(y_test, y_pred, target_names = target_names))
```

                  precision    recall  f1-score   support
    
          Healty       1.00      0.50      0.67        10
     Parkinson's       0.85      1.00      0.92        29
    
        accuracy                           0.87        39
       macro avg       0.93      0.75      0.79        39
    weighted avg       0.89      0.87      0.86        39
    
    

### Logistic Activation Function


```python
layer_size = [5, 10]
activation_function = "logistic"

clf_log = MLPClassifier(hidden_layer_sizes = layer_size, random_state = 0, activation = activation_function)
clf_log.fit(X_train, y_train)
```




    MLPClassifier(activation='logistic', hidden_layer_sizes=[5, 10], random_state=0)




```python
y_pred = clf_log.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

target_names = ["Healty", "Parkinson's"]

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, square=True, cbar=False, xticklabels = target_names, yticklabels = target_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Accuracy Score: {:.3}'.format(accuracy));
```


![png](assets/output_33_0.png)



```python
print(classification_report(y_test, y_pred, target_names = target_names))
```

                  precision    recall  f1-score   support
    
          Healty       0.00      0.00      0.00        10
     Parkinson's       0.74      1.00      0.85        29
    
        accuracy                           0.74        39
       macro avg       0.37      0.50      0.43        39
    weighted avg       0.55      0.74      0.63        39
    
    
    

Dari confusion matrix diatas diperoleh accuracy model sebesar 0.744. Terlihat juga tidak terdapat kesalahan (FN) dalam memprediksi seseorang yang mengalami Parkinson, yang berarti recall dari model ini sebesar 1. Akan tetapi model ini masih sangat buruk dalam memprediksi sesorang yang sehat, semua testing data diprediksi Parkinson, sehingga precision sebesar 0.74. Sementara itu F1 score dari model ini sebesar 85%, 2% lebih dibawah dari model yang menggunakan activation function relu.

Sehingga dapat disimpulkan model dengan activation function relu lebih baik daripada model dengan activation function logistic. Untuk lebih meyakinkan saya akan membuat ROC Curve dari dua model tersebut.

### ROC Curve


```python
plt.figure(figsize = (6,6))
for act_function, model in zip(('relu', 'logistic'), (clf_relu, clf_log)):
    y_probs = model.predict_proba(X_test)[:, 1] 
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (act_function, auc))
    

plt.plot([0, 1], [0, 1],'r--', label='No Skill (area = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right");
```


![png](assets/output_37_0.png)


Terlihat bahwa area under curve ddari relu lebih tinggi dari logistic

### Melihat Perbedaan Jika Jumlah Hidden Layer di Perbanyak


```python
lstLayer = []
lstAccuracy = []
for hidden_layer in [1, 5, 10, 15, 20]:
    for black_box in [1, 10, 20]:
        layer_size = [hidden_layer, black_box]
        activation_function = "relu"
        
        clf_relu = MLPClassifier(hidden_layer_sizes = layer_size, random_state = 0, activation = activation_function)
        clf_relu.fit(X_train, y_train)
        
        y_pred = clf_relu.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        lstLayer.append("{} {}".format(hidden_layer, black_box))
        lstAccuracy.append(accuracy)
```
    

```python
pd.DataFrame({'layer' : lstLayer, 'accuracy' : lstAccuracy})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>layer</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 1</td>
      <td>0.692308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1 10</td>
      <td>0.743590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1 20</td>
      <td>0.743590</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5 1</td>
      <td>0.743590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5 10</td>
      <td>0.871795</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5 20</td>
      <td>0.871795</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10 1</td>
      <td>0.794872</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10 10</td>
      <td>0.871795</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10 20</td>
      <td>0.871795</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15 1</td>
      <td>0.256410</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15 10</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15 20</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <th>12</th>
      <td>20 1</td>
      <td>0.743590</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20 10</td>
      <td>0.871795</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20 20</td>
      <td>0.871795</td>
    </tr>
  </tbody>
</table>
</div>



Terlihat ternyata sakin banyak hidden layer dan black box maka semakin baik modelnya
