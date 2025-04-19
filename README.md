## EXNO-3-DS
```
Reg no: 212224040020
Name: AMAN SINGH
```
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
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Encoding Data.csv")
```
![image](https://github.com/user-attachments/assets/df9d0b40-c715-4fb6-ab74-185c4beb43d5)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder,QuantileTransformer
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/acd9000f-5fe3-4f4e-9f54-477f29e70571)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/fe783c8d-67ba-4a96-a4da-564c5c864d46)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/6b209f85-b375-4657-9f4b-62b6316eb9e3)
```
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/e86f81ab-fd86-4710-b7da-489c0dbbf1e5)
```
pd.get_dummies(df2,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/b9c19ed7-ca12-4002-91d9-4d63236a42ef)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/9dbe2037-e654-4349-b810-0eb44dad8e29)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/3ec753ba-2e5a-4ed9-a616-011fcef8353b)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/9deccee8-5baa-49e2-a95e-a8b7de9fc6fc)
```
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/d4e14f36-a612-4d93-8a2a-80e46e44d016)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/19ac1487-d792-4f71-8084-ab120c7ae274)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/136bdb46-6ada-442d-b45f-7c10bf167bfa)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d8c6eb9c-a532-4a35-920c-6daed2cf4805)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0277e146-daa9-4244-acd3-c33aa7377554)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/02b591c0-c329-40ec-aa81-a0362ec53c4c)
```
df["Higly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/3eaaa4ad-4d73-449c-a3c7-0a6daab5c385)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/e39fee6e-f961-4e62-aeb9-5cc157af0b4d)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/787018f9-42d1-4fd2-9c44-c15f742631e5)
```
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/668985d4-12ea-49b9-8b6b-d4a263670fca)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/bc1d814d-9237-4f4e-9880-374c93d1a7c3)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/d53bd274-892c-41bd-9a10-608d2658f90a)
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/0389490e-eba2-4926-ad64-dccdb1105288)
```
df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()  
```
![image](https://github.com/user-attachments/assets/dd661e98-2d9c-4c0e-9881-ef499c968839)
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt['Age_1']=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8b9d160f-d4e7-426e-a37e-926c7f4b2b41)
# RESULT:
Feature Encoding and Transformation process has been successfully performed using the data set.

```














```
       
