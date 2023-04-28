from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

OBJECTS = ['marker', 'duck', 'ball', 'cup', 'horse']

if __name__ == '__main__':
  X = []
  y = []
  for object in OBJECTS:
    with open(f'../data/{object}_pos.csv', 'r') as pos_csv, open(f'../data/{object}_touch.csv', 'r') as touch_csv:
      for pos_row, touch_row in zip(pos_csv, touch_csv):
        X.append(eval(touch_row))
        y.append(object)

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)
  print(f'train accuracy: {model.score(X_train, y_train)}')
  print(f'test accuracy: {model.score(X_test, y_test)}')
  print('confusion matrix:')
  print(confusion_matrix(y_test, model.predict(X_test)))
