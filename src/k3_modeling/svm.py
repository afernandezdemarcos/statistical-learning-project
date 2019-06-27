import sklearn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC

# Concatenate training and validation sets
def concatenate_svm(train_features, train_labels, validation_features, validation_labels):
    svm_features = np.concatenate((train_features, validation_features))
    svm_labels = np.concatenate((train_labels, validation_labels))
    return svm_features, svm_labels

train_dir = 'data/processed/'
train_feat = np.load(train_dir+'train_features.npy')
train_lab = np.load(train_dir+'train_labels.npy')
val_feat = np.load(train_dir+'validation_features.npy')
val_lab = np.load(train_dir+'validation_labels.npy')

svm_features, svm_labels = concatenate_svm(train_feat, train_lab, val_feat, val_lab)

# Build model
X_train, y_train = svm_features, [np.where(r == 1)[0][0] for r in svm_labels]

param = [{
          "C": [0.01, 0.1, 1, 10, 100]
         }] 
svm = LinearSVC(penalty='l2', loss='squared_hinge')

param = [{
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": [0.01, 0.1, 1, 10, 100]
}]
svm = SVC(kernel='rbf')

clf = GridSearchCV(svm, param, cv=10)
clf.fit(X_train, y_train)

print(clf.best_score_)