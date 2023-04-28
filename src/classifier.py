from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

OBJECTS = ['duck', 'marker', 'ball', 'cup', 'horse']

if __name__ == '__main__':
  X = []
  y = []
  for i in range(len(OBJECTS)):
    object = OBJECTS[i]
    with open(f'../data/{object}_pos.csv', 'r') as pos_csv, open(f'../data/{object}_touch.csv', 'r') as touch_csv:
      for pos_row, touch_row in zip(pos_csv, touch_csv):
        X.append(eval(pos_row) + eval(touch_row))
        y.append(i)

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  knn = KNeighborsClassifier()
  knn.fit(X_train, y_train)
  print(f'train accuracy: {knn.score(X_train, y_train)}')
  print(f'test accuracy: {knn.score(X_test, y_test)}')
