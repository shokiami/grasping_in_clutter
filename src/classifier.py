from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

OBJECTS = ['duck', 'marker']

X = []
y = []
for i in range(len(OBJECTS)):
  object = OBJECTS[i]
  



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)