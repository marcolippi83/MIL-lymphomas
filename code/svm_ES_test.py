import pandas as pd
import numpy as np
import scipy
import sys
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, f1_score, confusion_matrix, make_scorer

################################################################
def norm_data(X_train,X_test):
    # Normalize data (0-1)
    scaler = MinMaxScaler().fit(X_train)
    s_X_train = scaler.transform(X_train)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1,-1)
    s_X_test = scaler.transform(X_test)
    return s_X_train, s_X_test
################################################################
def stand_data(X_train,X_test):
    # Standardize data (0 mean, 1 stdev)
    scaler = StandardScaler().fit(X_train)
    s_X_train = scaler.transform(X_train)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1,-1)
    s_X_test = scaler.transform(X_test)
    return s_X_train, s_X_test
################################################################
def joint_shuffle(a, b):
    # Shuffle two numpy arrays together
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b
################################################################
def cross_validation(tuned_parameters, X_train, y_train, patients_train):
    sample_weights = []
    prev = -1
    count = 0.0
    for i in range(len(patients_train)):
        if patients_train[i] != prev and prev != -1:
            for j in range(int(count)):
                sample_weights.append(1.0/count)
            count = 0.0
        prev = patients_train[i]
        count = count + 1.0
    for j in range(int(count)):
        sample_weights.append(1.0/count)
    print("Running cross-validation:")
    group_kfold = GroupKFold(n_splits=36)
    folds = group_kfold.split(X_train, y_train, patients_train)
    clf = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=folds, scoring=make_scorer(fbeta_score, beta=1))
    clf.fit(X_train,y_train,sample_weight=sample_weights)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print("---")
    return clf.best_estimator_
################################################################
def print_results(y_test, y_pred):
    c_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test,y_pred)
    pre = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print("---")
    print("Confusion Matrix")
    print(c_matrix)
    print("---")
    print("Accuracy = " + str(acc) + "\nPrecision = " + str(pre) + "\nRecall = " + str(rec) + "\nF1 = " + str(f1))
################################################################
def compute_bags(data_file,list_features):
    df = pd.read_csv(data_file,sep=",",header=0)

    num_rows = len(df)
    bags = {}
    patients = {}
    prev_patient = -1

    for feature in list_features:
        if (feature != 'region' and min(df[feature]) > 0 and pd.DataFrame.skew(df[feature] > 2)):
            df[feature] = np.log(df[feature])
        if (feature != 'region' and max(df[feature]) < 0 and pd.DataFrame.skew(df[feature] < -2)):
            df[feature] = np.exp(df[feature])

    for i in range(num_rows):
        voi = []
        patient = df['patient'][i]
        for feature in list_features:
            if feature == 'region':
                tmp = np.zeros(8)
                tmp[int(df[feature][i])-1] = 1
                for r in range(len(tmp)):
                    voi.append(tmp[r])
            val = float(df[feature][i])
            voi.append(val)
        if (patient != prev_patient and prev_patient != -1):
            stats_bag = []
            stats = scipy.stats.describe(bag,axis=0)
            quantiles = scipy.stats.mstats.mquantiles(bag,axis=0)
            for k in range(len(stats.mean)):
                stats_bag.append(stats.mean[k])
                stats_bag.append(stats.minmax[0][k])
                stats_bag.append(stats.minmax[1][k])
                #stats_bag.append(quantiles[0][k])
                #stats_bag.append(quantiles[1][k])
                #stats_bag.append(quantiles[2][k])
                #stats_bag.append(stats.variance[k])
                #stats_bag.append(stats.skewness[k])
                #stats_bag.append(stats.kurtosis[k])
            stats_bag.append(bag.shape[0])
            if label in bags:
                bags[label].append(stats_bag)
                patients[label].append(prev_patient)
            else:
                bags[label] = [stats_bag]
                patients[label] = [prev_patient]
            bag = np.asmatrix(voi)
        elif (prev_patient == -1):
            bag = np.asmatrix(voi)
        else:
            bag = np.vstack((bag,np.asarray(voi)))
        prev_patient = patient
        label = df['type'][i]

    bags[label].append(stats_bag)
    patients[label].append(patient)

    return bags, patients
################################################################

train_data_file = sys.argv[1]
test_data_file = sys.argv[2]
positive_class = sys.argv[3]

np.random.seed(7)

classes = ['FL','HL','MCL','DLBCL']
negative_classes = [neg for neg in classes if neg != positive_class]

#list_features = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A27','A28','A29','A30','A31','A32','A33','A34','A35','A36','A37','A38','A39','A40','A41','A42','A43','A44','A47','A48','A49','A50','A51','A52','A53','A54','A55','A56','A57','A58','A59','A60','A61','A63','A65','A67','A68','A69','A70','A71','A73','A74','A75','A76','A77','A79','A80','A81','A82','A83','A84','A85','A86','A87','A88','A89','A90','A91','A92','A94','A95','A96','A97','A98','A99','A100','A101','A103','A104','A105','A106','A108','region']
list_features = ['A18','A19','A20','A21','A22','A104','A105','A106','A108','region']

bags,patients = compute_bags(train_data_file,list_features)

bags_pos = list(bags[positive_class])
patients_pos = list(patients[positive_class])
bags_neg = list(bags[negative_classes[0]])
patients_neg = list(patients[negative_classes[0]])
bags_neg.extend(bags[negative_classes[1]])
patients_neg.extend(patients[negative_classes[1]])
bags_neg.extend(bags[negative_classes[2]])
patients_neg.extend(patients[negative_classes[2]])

bags_train = list(bags_pos)
bags_train.extend(bags_neg)
patients_train = list(patients_pos)
patients_train.extend(patients_neg)

y_train = []
for k in range(len(bags_train)):
   if (k < len(bags_pos)):
       y_train.append(1)
   else:
       y_train.append(-1)

bags,patients = compute_bags(test_data_file,list_features)

bags_pos = list(bags[positive_class])
patients_pos = list(patients[positive_class])

bags_test = list(bags_pos)
patients_test = list(patients_pos)

predictions_all = []
patients_all = []
labels_all = []

X_train = bags_train
X_test = bags_test

y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
X_test = np.asmatrix(X_test)

print(y_train)

X_train, X_test = norm_data(X_train, X_test)

tuned_parameters = [{'kernel': ['linear'], 'C': [2**i for i in np.arange(-8,15.0,0.5)]}]
clf = cross_validation(tuned_parameters,X_train,y_train,patients_train)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(y_pred)

np.savetxt('pred_svm_skew_' + positive_class + '.txt', y_pred, fmt="%d")
np.savetxt('patients_svm_skew_' + positive_class + '.txt', patients_test, fmt="%d")




