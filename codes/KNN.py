from codes.Preprocess import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = data_reader()
X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print('Score Of Test :'+str(knn.score(X_test, y_test)))
