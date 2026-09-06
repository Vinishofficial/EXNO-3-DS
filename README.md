## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```python
import pandas as pd
df = pd.read_csv("Encoding Data.csv")
df
```
<img width="325" height="337" alt="image" src="https://github.com/user-attachments/assets/857cc67a-ef22-4888-a6c3-f5b991065166" />

```python
# ORDINAL ENCODING 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```
<img width="178" height="193" alt="image" src="https://github.com/user-attachments/assets/d66b0781-9bb2-4993-9f13-29a97ba444a4" />

```python
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
```
<img width="355" height="358" alt="image" src="https://github.com/user-attachments/assets/64529202-f3f7-4ce2-83c9-865e9df7d392" />

```python
# Label Encoder ( orders in alphabetical order)
le=LabelEncoder() 
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
```
<img width="347" height="338" alt="image" src="https://github.com/user-attachments/assets/a0613b88-f5c8-4734-bb8f-dd3487883eab" />

```python
# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
df2=pd.concat([df2,enc],axis=1) 
df2
```
<img width="460" height="362" alt="image" src="https://github.com/user-attachments/assets/b3e1b6d8-7bbc-4d4f-b989-dd6ab0627d5b" />

```python
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="647" height="363" alt="image" src="https://github.com/user-attachments/assets/54706efb-6edb-40ba-8026-7f7a15d8ad4c" />

```python
# BINARY ENCODER 
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
```
<img width="503" height="357" alt="image" src="https://github.com/user-attachments/assets/fcbce3e8-7c7b-4012-a813-1ecb2dd11680" />

```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```
<img width="702" height="360" alt="image" src="https://github.com/user-attachments/assets/4329068d-f8b4-44ee-870e-8a6f6b92c69b" />

```python
# MEAN ENCODING 
from category_encoders import TargetEncoder 
te=TargetEncoder()
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```
<img width="510" height="345" alt="image" src="https://github.com/user-attachments/assets/570a53f1-1ac1-4330-838f-3843d8eaa927" />

```python
# FEATURE TRANFORMATION
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="706" height="397" alt="image" src="https://github.com/user-attachments/assets/cd550b46-d490-44c9-baaf-258d6fabcef6" />

```python
df.skew()
```
<img width="315" height="103" alt="image" src="https://github.com/user-attachments/assets/a096ef0f-f540-4749-9435-1e3b1a9bea5b" />

```python
# 1. LOG TRANSFORMATION
np.log(df["Highly Positive Skew"])
```
<img width="490" height="221" alt="image" src="https://github.com/user-attachments/assets/27fcf6cc-828c-4754-9bc2-b89f628c5e02" />

```python
# 2. RECIPROCAL TRANSFORMATION
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="502" height="227" alt="image" src="https://github.com/user-attachments/assets/c2f84a84-a9bd-4cda-9ae8-82d6e8fe64f3" />

```python
# 3. SQUARE ROOT TRANSFORMATION
np.sqrt(df["Highly Positive Skew"])
```
<img width="492" height="220" alt="image" src="https://github.com/user-attachments/assets/f9cd1b59-16cf-45cc-80e3-b8b97936ac81" />

```python
# POWER TRANSFORMATIONS
# BOX COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="997" height="427" alt="image" src="https://github.com/user-attachments/assets/03d9d81a-2da2-44c5-8824-cde1f25f432b" />

```python
df.skew()
```
<img width="392" height="127" alt="image" src="https://github.com/user-attachments/assets/ad4a5c38-6181-48bb-b704-64fc4f742960" />

```python
# YEO_JOHNSON
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="420" height="148" alt="image" src="https://github.com/user-attachments/assets/103cb107-b2d8-40ad-8f52-96162d883171" />

```python
# QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1125" height="427" alt="image" src="https://github.com/user-attachments/assets/2af00d0b-c754-4272-a924-db986507d3df" />

```python
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="652" height="438" alt="image" src="https://github.com/user-attachments/assets/77fe28b4-288a-4a04-80e3-0fd9e71def3c" />

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="652" height="432" alt="image" src="https://github.com/user-attachments/assets/5e2554b6-4847-4185-8224-06bcb4af0aa9" />

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="693" height="445" alt="image" src="https://github.com/user-attachments/assets/ca210eb9-f3b7-4e3c-822b-f33e4366af6a" />

# RESULT:
 Thus, the given data successfully performed Feature Encoding and Feature Tranformation

# SUMMARY
The given dataset was successfully loaded, cleaned, and processed using various Feature Encoding and Feature Transformation techniques.
Categorical features were converted into numerical form using Ordinal, Label, One-Hot, Binary, and Target Encoding methods.
Numerical features were transformed using Log, Reciprocal, Square Root, Box-Cox, Yeo-Johnson, and Quantile transformations to reduce skewness and improve data distribution.
Skewness analysis and Q-Q plots were used to evaluate the effectiveness of the transformations.
Thus, the experiment successfully demonstrates how encoding and transformation techniques prepare raw data for effective statistical analysis and machine learning models.
