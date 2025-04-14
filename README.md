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
```
Developed by : Aman Singh
Reg No : 212224040020
```

```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/218eaa1c-cdf7-46db-9125-bd6035c92249)


```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/11572a9c-b125-46c2-a68c-23846f478b12)




```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/d0f268e5-6ace-4d58-8b21-3d7cb83dee4b)



```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/8fd3a44c-e1e8-4389-9fc2-eb7f6b0198c9)




```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/231b943e-4b89-4989-a475-a5549d7edf03)



```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/bffe5356-ba88-4a3e-86c7-0501f9a1ae8f)


```python
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/c0232205-6898-4ea5-a7af-f789c88276d4)




```python
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/6fa40515-4a65-44c0-801e-abb92a301ab8)




```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/ea7553e2-7c54-4558-8a72-a8a0d7b18be4)




```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/e06c275a-b6da-4b4c-888c-5d6a9dfe6075)


```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/dff82f73-df44-4bd6-9db2-d8bbeec38b31)





```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/442de61f-aa24-49b5-829d-826d3c15970c)





```python
df.skew()
```
![image](https://github.com/user-attachments/assets/d6e84347-3fd2-4c49-9f67-8a898fc6b81d)





```python
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/31b09c87-d4d2-419f-afa8-fe16d41847a1)




```python
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6eba53db-e098-41cb-92e9-5b9cec22df2d)



```python
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a84f0d68-215e-40cc-aa86-7cda9620575d)



```python
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/94afe8a0-a62f-4cd9-b6a9-ac775a3bfd2c)



```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/ef0dac47-61fb-42ba-ab18-fa26f5a5ad23)


```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/03fa507b-01c3-415d-909a-e70e060bb59d)



```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="473" alt="Screenshot 2025-04-14 at 3 54 09 PM" src="https://github.com/user-attachments/assets/b8374c15-bee2-4de4-85e4-9086d218aeff" />




```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="481" alt="Screenshot 2025-04-14 at 3 54 33 PM" src="https://github.com/user-attachments/assets/fdc5a501-347a-4615-9411-0559f79199e7" />




```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="492" alt="Screenshot 2025-04-14 at 3 54 59 PM" src="https://github.com/user-attachments/assets/a98b36ef-5738-4fbe-8e31-6bb5a29b8754" />


```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```


<img width="476" alt="Screenshot 2025-04-14 at 3 55 21 PM" src="https://github.com/user-attachments/assets/e430b24e-c871-4d37-a07c-69d7aea56c6d" />



```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="486" alt="Screenshot 2025-04-14 at 3 55 41 PM" src="https://github.com/user-attachments/assets/200bb387-ecf2-49dc-a3ae-a4743d648f32" />




```python
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line='45')
plt.show()
```



<img width="469" alt="Screenshot 2025-04-14 at 3 56 43 PM" src="https://github.com/user-attachments/assets/8523d6f6-ecd9-4f42-b3d1-ab7f1f383682" />

```python
sm.qqplot(dt["Age_1"],line='45')
plt.show()
```
<img width="494" alt="Screenshot 2025-04-14 at 3 57 25 PM" src="https://github.com/user-attachments/assets/4f662021-7885-4f2f-8caf-ecfc7fcf7e9e" />



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
