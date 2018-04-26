from codes.Preprocess import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
X, y, our_classes = data_reader(0)
X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
#
# # Loop over different values of k
# for i, k in enumerate(neighbors):
#     # Setup a k-NN Classifier with k neighbors: knn
#     knn = KNeighborsClassifier(n_neighbors=k)
#
#     # Fit the classifier to the training data
#     knn.fit(X_train, y_train)
#
#     # Compute accuracy on the training set
#     train_accuracy[i] = knn.score(X_train, y_train)
#
#     # Compute accuracy on the testing set
#     test_accuracy[i] = knn.score(X_test, y_test)
#
# # Generate plot
# plt.title('k-NN: Varying Number of Neighbors')
# plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label='Training Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()
knn.fit(X_train, y_train)
print('Accuracy = ' + str(knn.score(X_test, y_test)))
predictions = knn.predict(X_test[:7])
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
