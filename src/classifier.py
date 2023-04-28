from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

OBJECTS = ['duck', 'marker']

if __name__ == '__main__':
  X = []
  y = []
  for i in range(len(OBJECTS)):
    object = OBJECTS[i]
    with open(f'../data/{object}_pos.csv', 'r') as pos_csv, open(f'../data/{object}_touch.csv', 'r') as touch_csv:
      for pos_data, touch_data in zip(pos_csv, touch_csv):
        X.append(pos_data + touch_data)
        y.append(i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print(f'train accuracy: {knn.score(X_train, y_train)}')
    print(f'test accuracy: {knn.score(X_test, y_test)}')
