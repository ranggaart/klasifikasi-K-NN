import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social-Network.csv')

dataset.head()

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

print('\nRangga Aristianto')
print('\nA11.2022.14568')
print('\n\nData Kasus:')
print('\nJumlah Data X:')
len(x)
print('\nData X:')
print(x)
print('\n\nJumlah Data Y:')
len(y)
print('\nData Y:')
print(y)

# Membagi Dataset ke dalam Training Set dan Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

print ('\n\nData Training Set dan Test Set:')
print('\nJumlah Data X Train:')
len(x_train)
print('\nData X :')
print(x_train)
print('\n\nJumlah Data Y Train:')
len(y_train)
print('\nData Y Train:')
print(y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print ('\n\nData Feature Scaling:')
print('\nData X Train:')
print(x_train)
print('\nData X Test:')
print(x_test)

# K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('\nData Confusion Matrix:')
print(cm)

# Klasifikasi Data dengan K-NN (Data Training)
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Klasifikasi Data dengan K-NN (Data Training)')
plt.xlabel('Umur')
plt.ylabel('Estimasi Gaji')
plt.legend()
plt.show()

# Klasifikasi Data dengan K-NN (Data Testing)
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 0].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Klasifikasi Data dengan K-NN (Data Testing)')
plt.xlabel('Umur')
plt.ylabel('Estimasi Gaji')
plt.legend()
plt.show()