
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#url = "./Concrete_Data.xls"
data = pd.read_excel("Concrete_Data.xls")
data.head()

len(data)

req_col_names = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
                 "CoarseAggregate", "FineAggregate", "Age", "CC_Strength"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)

data.head()

data.isna().sum()

data.describe()

x = data.iloc[:,:-1]   
y = data.iloc[:,-1]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


regression=LinearRegression()


fit = regression.fit(x_train,y_train)


regression.predict(x_test)

print(regression.score(x_test,y_test))

pickle.dump(fit,open('model.pkl','wb'))
