from array import array
from re import sub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import itertools
import sys
from tqdm import tqdm #tqmd progress bar

###########################
#### UTILITY FUNCTIONS ####
###########################
# utility function for printing progress bar
def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = float(index) / float(total)
    prog_text = "{a:5.2f}%".format(a = 100*progress)
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {prog_text}  {label}")
    sys.stdout.flush()

# Function for confusion matrix.
def plot_confusion_matrix(cm, CADes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CADes))
    plt.xticks(tick_marks, CADes, rotation=45)
    plt.yticks(tick_marks, CADes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# function for getting all possible combinations of a list
from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

######################
#### ML ALGORITHM ####
######################
# create a dataframe from the data and read it
df = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/src/cad_dset.csv')
df.head()

# look at the distribution of the CADes based on Clump thickness and Uniformity of cell size:
# ax = df[df['CAD'] == 1][0:50].plot(kind='scatter', x='known CAD', y='previous AMI', color='DarkBlue', label='previous CABG');
# df[df['CAD'] == 0][0:50].plot(kind='scatter', x='known CAD', y='previous AMI', color='Yellow', label='benign', ax=ax);
# plt.show()
# plt.savefig("plot.jpg")

## Data pre-processing and selection
df.dtypes
# # turn non numeric 'BareNuc' column values to numeric
# df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
# df['BareNuc'] = df['BareNuc'].astype('int')
# df.dtypes

# test = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
# 'Chronic Kindey Disease','Family History of CAD','ASYMPTOMATIC','ATYPICAL SYMPTOMS','ANGINA LIKE','DYSPNOEA ON EXERTION','INCIDENT OF PRECORDIAL PAIN','RST ECG','male','Overweight',
# 'Obese','normal_weight','<40','40-50','50-60','>60','CNN_Healthy','Doctor: Healthy']

test = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
'Chronic Kindey Disease','Family History of CAD','ASYMPTOMATIC','ATYPICAL SYMPTOMS','ANGINA LIKE','DYSPNOEA ON EXERTION','INCIDENT OF PRECORDIAL PAIN','RST ECG','male','Overweight',
'Obese','normal_weight','<40','40-50','50-60','>60','CNN_Healthy']

## FOR TESTING PURPOSES
test2 = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'male', 'Overweight', '40-50', '>60', 'Doctor: Healthy']
feature_df = df[list(test2)]
X = np.asarray(feature_df)
df['CAD'] = df['CAD'].astype('int')
y = np.asarray(df['CAD'])
k = 13
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4, shuffle=True)
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# print("neigh :", neigh)
## Prediction
n_yhat = neigh.predict(X_test)
print("!!!!new best accuracy: ", metrics.accuracy_score(y_test, n_yhat))



# doctor's decision's accuracy
doctor = np.asarray(df['Doctor: Healthy'])
true = np.asarray(df['HEALTHY'])
# initialize best accuracy as the doctor's accuracy
best_acc=metrics.accuracy_score(true, doctor)
# best_acc=0
print("doctor's accuracy: ", metrics.accuracy_score(true, doctor))

i=0
# total = len(test)**len(test)
total = 134217728
# for subset in tqdm(all_subsets(test)): #tqmd progress bar
with tqdm(total=total) as prog_bar:
    for subset in all_subsets(test):
        if len(subset) < 4 :
            i += 1 # commend if using #tqmd progress bar
            # pass #tqmd progress bar
        else:
            i += 1 # commend if using #tqmd progress bar
            # select features
            feature_df = df[list(subset)]
            X = np.asarray(feature_df)
            # select target variable
            df['CAD'] = df['CAD'].astype('int')
            y = np.asarray(df['CAD'])

            ## Train/Test split
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4, shuffle=True)
            # # print the shape of X_trainset and y_trainset
            # print('Shape of X training set {}'.format(X_train.shape),'&','Size of Y training set {}'.format(y_train.shape))
            # # print the shape of X_testset and y_testset
            # print('Shape of X test set {}'.format(X_test.shape),'&','Size of Y test set {}'.format(y_test.shape))


            #################################
            #### Support Vector Machines ####
            #################################
            # print("\nSupport Vector Machines\n")
            alg="svm"
            ## Modelling
            # Each of Linear, Polynomial, Radial basis function (RBF) and Sigmoid functions has its characteristics, its pros and cons, and its equation,
            # but as there's no easy way of knowing which function performs best with any given dataset. Usually choose different functions in turn and 
            # compare the results. Now use the default, RBF (Radial Basis Function)
            rbf = svm.SVC(kernel='rbf')
            rbf.fit(X_train, y_train)

            ## Prediction
            yhat = rbf.predict(X_test) 

            ## Evaluation
            # Compute confusion matrix
            # cnf_matrix = metrics.confusion_matrix(y_test, yhat, labels=[0,1])
            # print(metrics.confusion_matrix(y_test, yhat, labels=[0,1]))
            # np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            # plt.figure()
            # plt.savefig("plot.jpg")
            # plot_confusion_matrix(cnf_matrix, CADes=['Healthy(0)','CAD(1)'],normalize= False,  title='Confusion matrix')

            if metrics.accuracy_score(y_test, yhat) > best_acc:
                best_acc=metrics.accuracy_score(y_test, yhat)
                print("new best accuracy: ", best_acc)
                print("algorithm: ", alg)
                print("features: ", subset)

            # doctor's decision's accuracy
            # # doctor = np.asarray(df['Doctor: Healthy'])
            # # true = np.asarray(df['HEALTHY'])
            # # print("doctor's accuracy: ", metrics.accuracy_score(true, doctor))
            # # alternative approach: use accuracy
            # print("accuracy: ", metrics.accuracy_score(y_test, yhat))
            # # alternative approach: use f1 score
            # f1_sc = metrics.f1_score(y_test, yhat, average='weighted')
            # print("f1_score: ", f1_sc)
            # # alternative approach: use jaccard score
            # j_sc = metrics.jaccard_score(y_test, yhat,pos_label=1)
            # print("jaccard_score: ", j_sc)

            ###########################
            #### Linear Regression ####
            ###########################
            # print("\nLinear Regression\n")
            alg="lr"
            # train a Logistic Regression Model using the train_x you created and the train_y created previously
            regr = linear_model.LinearRegression()
            X = np.asarray(feature_df)
            y = np.asarray(df['CAD'])
            regr.fit(X, y)

            test_x = np.asarray(feature_df)
            test_y = np.asarray(df['CAD'])
            # find the predictions using the model's predict function and the test_x data
            test_y_ = regr.predict(test_x)
            # convert test_y_ to boolean
            bool_test_y_ = []
            for y in test_y_:
                if y < 0.5:
                    bool_test_y_.append(0)
                else:
                    bool_test_y_.append(1)

            if metrics.accuracy_score(test_y, bool_test_y_) > best_acc:
                best_acc=metrics.accuracy_score(test_y, bool_test_y_)
                print("new best accuracy: ", best_acc)
                print("algorithm: ", alg)
                print("features: ", subset)

            # The coefficients
            # print ('Coefficients: ', regr.coef_)
            # print ('Intercept: ',regr.intercept_)

            # ## Evaluation
            # # Compute confusion matrix
            # cnf_matrix = metrics.confusion_matrix(test_y, bool_test_y_, labels=[0,1])
            # np.set_printoptions(precision=2)
            # # doctor's decision's accuracy
            # doctor = np.asarray(df['Doctor: Healthy'])
            # true = np.asarray(df['HEALTHY'])
            # print("doctor's accuracy: ", metrics.accuracy_score(true, doctor))
            # # alternative approach: use accuracy
            # print("accuracy: ", metrics.accuracy_score(test_y, bool_test_y_))
            # # alternative approach: use f1 score
            # f1_sc = metrics.f1_score(test_y, bool_test_y_, average='weighted')
            # print("f1_score: ", f1_sc)
            # # alternative approach: use jaccard score
            # j_sc = metrics.jaccard_score(test_y, bool_test_y_,pos_label=1)
            # print("jaccard_score: ", j_sc)

            #############################
            #### K Nearest Neighbors ####
            #############################
            # print("\nK Nearest Neighbors\n")
            alg="knn"
            # train a Logistic Regression Model using the train_x you created and the train_y created previously
            ## Training
            # with k=4
            k = 7
            neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
            # print("neigh :", neigh)

            ## Prediction
            n_yhat = neigh.predict(X_test)

            # # check for best K (1-10)
            # num_K = 10
            # mean_acc = np.zeros((num_K-1))
            # std_acc = np.zeros((num_K-1))

            # for n in range(1,num_K):  
            #     #Train Model and Predict  
            #     neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
            #     n_yhat2=neigh.predict(X_test)
            #     mean_acc[n-1] = metrics.accuracy_score(y_test, n_yhat2)
            #     std_acc[n-1]=np.std(n_yhat2==y_test)/np.sqrt(n_yhat2.shape[0])
            # print("mean acc: ", mean_acc)

            ## Evaluation
            # # Compute confusion matrix
            # cnf_matrix = metrics.confusion_matrix(y_test, n_yhat, labels=[0,1])
            # np.set_printoptions(precision=2)
            # # doctor's decision's accuracy
            # doctor = np.asarray(df['Doctor: Healthy'])
            # true = np.asarray(df['HEALTHY'])
            # print("doctor's accuracy: ", metrics.accuracy_score(true, doctor))
            # # alternative approach: use accuracy
            # print("accuracy: ", metrics.accuracy_score(y_test, n_yhat))
            # # alternative approach: use f1 score
            # f1_sc = metrics.f1_score(y_test, n_yhat, average='weighted')
            # print("f1_score: ", f1_sc)
            # # alternative approach: use jaccard score
            # j_sc = metrics.jaccard_score(y_test, n_yhat,pos_label=1)
            # print("jaccard_score: ", j_sc)

            if metrics.accuracy_score(y_test, n_yhat) > best_acc:
                best_acc=metrics.accuracy_score(y_test, n_yhat)
                print("new best accuracy: ", best_acc)
                print("algorithm: ", alg)
                print("features: ", subset)



            ########################
            #### Decision Trees ####
            ########################
            # print("\nDecision Trees\n")
            alg="dt"
            ## Modelling
            # We will first create an instance of the DecisionTreeClassifier called drugTree
            # Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
            drugTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth = 4)
            drugTree # it shows the default parameters
            # fit the data with the training feature matrix X_train and training response vector y_train
            drugTree.fit(X_train,y_train)

            ## Prediction
            predTree = drugTree.predict(X_test)

            # ## Evaluation
            # # Compute confusion matrix
            # cnf_matrix = metrics.confusion_matrix(y_test, predTree, labels=[0,1])
            # np.set_printoptions(precision=2)
            # # doctor's decision's accuracy
            # doctor = np.asarray(df['Doctor: Healthy'])
            # true = np.asarray(df['HEALTHY'])
            # print("doctor's accuracy: ", metrics.accuracy_score(true, doctor))
            # # alternative approach: use accuracy
            # print("accuracy: ", metrics.accuracy_score(y_test, predTree))
            # # alternative approach: use f1 score
            # f1_sc = metrics.f1_score(y_test, predTree, average='weighted')
            # print("f1_score: ", f1_sc)
            # # alternative approach: use jaccard score
            # j_sc = metrics.jaccard_score(y_test, predTree,pos_label=1)
            # print("jaccard_score: ", j_sc)

            if metrics.accuracy_score(y_test, predTree) > best_acc:
                best_acc=metrics.accuracy_score(y_test, predTree)
                print("new best accuracy: ", best_acc)
                print("algorithm: ", alg)
                print("features: ", subset)

            # commend if using #tqmd progress bar
            if i % 13400 == 0:
                prog_bar.update(13400)
                # print_progress_bar(i, total=total, label="total progress\n")
                # print("PROGRESS: {a:5.2f}%".format(a =float(i)/float(total)))
