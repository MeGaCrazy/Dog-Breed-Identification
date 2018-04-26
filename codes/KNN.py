from codes.Preprocess import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X, y, our_classes = data_reader()
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

print('Accuracy :' + str(knn.score(X_test, y_test)))
predictions = knn.predict(X_test)
# Plot the last batch results:

X_test /= 255
Nrows = 2
Ncols = 3
for i in range(0, 6):
    plt.subplot(Nrows, Ncols, i + 1)
    plt.imshow(np.reshape(X_test[i], [80, 80, 3]))
    plt.title('Actual: ' + str(our_classes[y_test[i]]) + ' Pred: ' + str(our_classes[predictions[i]]),
              fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

plt.show()
