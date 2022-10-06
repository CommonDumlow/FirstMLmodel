# FirstMLmodel
This is rundown of the code.

## Decribe the dataset

```
import pandas as pd
ds = pd.read_csv('smoke_detection_iot.csv')
ds.describe()
```
## To predict
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ds = pd.read_csv('smoke_detection_iot.csv')
X = ds.drop(columns = ['Fire Alarm', 'Unnamed: 0', 'UTC'])
# here we are splitting the dataset into an output set and input set according to our needs.
y = ds['Fire Alarm']

model = DecisionTreeClassifier()
model.fit(X, y)

predictions = model.predict([ [5153, 26.029, 54.6, 4, 403 ,32532, 14523, 956.668, 0.550 ,0.30, 0.20, 0.2, 2.000] ])
predictions
```
## Show the input set
```
import pandas as pd

ds = pd.read_csv('smoke_detection_iot.csv')
X = ds.drop(columns = ['Fire Alarm', 'Unnamed: 0', 'UTC'])
y = ds['Fire Alarm']
X
```
## Show the output set
```
import pandas as pd

ds = pd.read_csv('smoke_detection_iot.csv')
X = ds.drop(columns = ['Fire Alarm', 'Unnamed: 0', 'UTC'])
y = ds['Fire Alarm']
y
```
## To check accuracy and split
```
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ds = pd.read_csv('smoke_detection_iot.csv')
X = ds.drop(columns = ['Fire Alarm', 'Unnamed: 0', 'UTC'])
y = ds['Fire Alarm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
```
## Visualizing the model
```
from sklearn import tree

ds = pd.read_csv('smoke_detection_iot.csv')
X = ds.drop(columns = ['Fire Alarm', 'Unnamed: 0', 'UTC'])
y = ds['Fire Alarm']

model = DecisionTreeClassifier()
model.fit(X, y)
tree.export_graphviz(model, out_file="model.dot", feature_names=['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT'], label = 'all', rounded = True, filled=True)
```
