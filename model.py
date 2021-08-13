import numpy as np 
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
#from kmodes.kprototypes import KPrototypes
#from imblearn.over_sampling import SMOTENC 
import time
import os
from sklearn import metrics



def Model_Export(model, name): 
    pickle_out = open("saved_models/" + name + ".pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()
    
def Model_Import(file): 
    pickle_in = open("saved_models/" + file, "rb")
    return pickle.load(pickle_in)

def read_in_data(name):
    Import = open("Data/clean_data/" + name + ".pickle", "rb")
    X_train, X_test, y_train, y_test, feature_index = pickle.load(Import)
    return X_train, X_test, y_train, y_test, feature_index 

def metrics(true, predict):
    ### Need to add the reference to Bobra 
    TN = 0.
    TP = 0.
    FP = 0.
    FN = 0.
    for i in range(len(predict)):
        if (predict[i] == 0 and true[i] == 0):
            TN += 1
        elif (predict[i] == 1 and true[i] == 0):
            FP += 1
        elif (predict[i] == 1 and true[i] == 1):
            TP += 1
        elif (predict[i] == 0 and true[i] == 1):
            FN += 1

    return TN, FP, TP, FN

def TSS(true, predict): 
    TN, FP, TP, FN = metrics(true, predict)
    return (TP/(TP+FN)) - ( FP/(FP+TN))



class Model: 
    
    
    
    def __init__(self,X_train, X_test, y_train, y_test, feature_index, name):
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_index, self.name = X_train, X_test, y_train, y_test, feature_index, name
        self.id =  time.strftime("%d%m%Y%H%M")
        
        
        
    def feature_reduction(self,method = "F_score", threshold = False):
        """
        Reduce the amount of features in the dataset. Three methods can be used: F-score, L1-svm, and Logistic regression. If 
        using F-scoring, an adittional parameter of threshold must be added, where threshold is the what is the minumum F-score
        for features selected 
        """
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        if method == "F_score":
            scorer = SelectKBest(k = "all").fit(X_train, y_train)
            F_score_data = pd.DataFrame( scorer.scores_)
            F_score_data.index = feature_index
            F_score_data.columns = ["F-score"]
            self.F_score_data = F_score_data
            if threshold == False: 
                return "Look up F-score data to decide a threshold "
            else: 
                new_feature_index = F_score_data[F_score_data["F-score"] > threshold].index
                best_no = F_score_data[F_score_data["F-score"] > threshold].shape[0]
                print(X_train.shape)
                X_train_new = SelectKBest(k = best_no).fit_transform(X_train, y_train)
                print(X_train_new.shape)
                self.X_train = X_train_new 
                X_test = pd.DataFrame(X_test)
                X_test.columns = self.feature_index
                self.X_test = np.array(X_test[new_feature_index])
                self.feature_index = new_feature_index 
                print("new feature index:")
                print(new_feature_index)
        
        if method == "L1_svm": 
            C_values = [2**i for i in range(-5,17,2)]
            parameters = {'penalty': ['l1'], 
                              'C': C_values, 
                              "dual":[False],
                         "class_weight": ["balanced"] }
            svc = LinearSVC()
            clf = GridSearchCV(svc, parameters,scoring = 'roc_auc' , cv = 5)
            clf.fit(X_train, y_train )
            model = SelectFromModel(clf.best_estimator_, prefit=True)
            print(X_train.shape)
            X_train_new = model.transform(X_train)
            print(X_train_new.shape)
            self.X_train = X_train_new 
            self.X_test = model.transform(X_test)
            self.feature_index = self.feature_index[model.get_support() == True]
            print("new feature index:")
            print(self.feature_index)
            
        if method == "Logistic": 
            print(X_train.shape)
            selector = SelectFromModel(estimator=LogisticRegression()).fit(X_train, y_train)
            X_train_new = selector.transform(X_train)
            print(X_train_new.shape)
            self.X_train = X_train_new 
            self.X_test = selector.transform(X_test)
            self.feature_index = self.feature_index[selector.get_support() == True]
            print("new feature index:")
            print(self.feature_index)
            
        

    def balance_data(self, N_clusters, l = 4):
        """
        Balance data by undersampling the negative class via K-prototype clustering and oversampling the positive class
        via SMOTE. N_clusters determines amount of datapoints for the undersampling, and categorical index is a list for 
        all categorical features in the dataset 
        
        """
        categorical_index = [i for i in range(l, len(self.feature_index))]
        
        # Seperate positive and negative classes 
        X_train = self.X_train 
        train = np.column_stack((X_train_new,y_train))
        train_pos = train[train[:,-1] == 1]
        X_train_pos = train[train[:,-1] == 1][:,:-1].copy()
        X_train_neg = train[train[:,-1] == 0][:,:-1].copy()
        
        ## undersample the negative class 
        km = KPrototypes(n_clusters = N_clusters).fit(X_train_neg, categorical = categorical_index)
        K_train_neg = km.cluster_centers_
        
        # Merge positve and negative classes 
        train_neg = np.column_stack((K_train_neg, np.zeros(K_train_neg.shape[0])))
        train_new = np.vstack((train_pos, train_neg))
        np.random.shuffle(train_new)
        print("training data shape")
        print(X_train.shape)
        X_new = train_new[:,:-1]
        y_new = train_new[:,-1]
        ## oversample the positive class 
        sm =  SMOTENC(categorical_features = categorical_index, random_state=42)
        X_res, y_res = sm.fit_resample(X_new, y_new)
        print()
        print("new training data shape")
        print(X_res.shape)
        self.X_train = X_res 
        self.y_train = y_res 
            
            
            
    def gridsearch(self):
        C_values = [10**i for i in range(-2,5)]
        Gamma_values = [10**i for i in range(-4,3)]
        parameters = {'kernel': ['rbf'], 
                  'C': C_values, 
                  "gamma":Gamma_values ,
                  "class_weight": ["balanced"] }

        svc = SVC()
        clf = GridSearchCV(svc, parameters,scoring = 'roc_auc' , cv = 10)
        clf.fit(self.X_train, self.y_train )
        ranked_data = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
        self.grid_data = ranked_data
        return ranked_data 

    
    
    def finesearch(self,C_values, Gamma_values):
        parameters = {'kernel': ['rbf'], 
                  'C': C_values, 
                  "gamma":Gamma_values ,
                  "class_weight": ["balanced"] }

        svc = SVC()
        clf = GridSearchCV(svc, parameters,scoring = 'roc_auc' , cv = 10)
        clf.fit(self.X_train, self.y_train )
        ranked_data = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
        self.fine_data = ranked_data
        clf_best = clf.best_estimator_
        cw, C, gamma  = clf_best.class_weight , clf_best.C, clf_best.gamma
        final = SVC(kernel = "rbf", gamma = gamma, C = C, class_weight = cw, probability = True)
        final.fit(self.X_train, self.y_train )
        self.final = final
        return ranked_data
    
    
def Run_Model(model,
          C_values = np.linspace(1, 10, 10),
          Gamma_values= np.linspace(0.0001, 0.001, 10),
         balance_data = False, 
         feature_reduction = False, 
         search = "grid"): 
    if balance_data == False: 
        pass 
    else: 
        pass 

    if feature_reduction == False: 
        pass 
    else: 
        model.feature_reduction(method = feature_reduction )

    if search == "grid": 
        print("start model grid search")
        model.gridsearch()
        data = model.grid_data
        print("done model grid search")

    if search == "fine": 
        print("start model fine search")
        model.finesearch(C_values, Gamma_values)
        data = model.fine_data
        print("end model fine search")

    Model_Export(model, model.name)
    data.to_csv(str(model.id) + "_"+ search + "_" + model.name +".csv", index_label=False)
    return data 
    
     