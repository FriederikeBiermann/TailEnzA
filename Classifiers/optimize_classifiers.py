import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)

def create_training_test_set(path_feature_matrix, test_size):
    feature_matrix=pd.read_csv(path_feature_matrix)
    feature_matrix = feature_matrix.sample(frac = 1) 
    # define target and features
    x_data = feature_matrix.loc[:, feature_matrix.columns != 'target' ]
    y_data = feature_matrix['target']
    # split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data ,test_size = test_size, shuffle=True)
    #resample to balance 
    x_train, y_train = ros.fit_resample(x_train, y_train)
    return x_train, x_test, y_train, y_test, x_data, y_data
def train_classifier_and_get_accuracies(classifier,name_classifier, enzyme, x_data,x_train,y_train,y_test,y_train ):
    classifier = classifier.fit(x_train,y_train)
    #predict for test set
    test_predict_classifier=classifier.predict(x_test)
    #predict for complete set
    y_pred = classifier.predict(x_data)
    feature_matrix['prediction']=y_pred
    #calculate accuracy
    cross_validation_classifier = cross_val_score(classifier, x_data, y_data, cv=5, scoring='f1_macro')
    balanced_accuracy_classifier=balanced_accuracy_score(y_test, test_predict_classifier)
    # creating a confusion matrix 
    cm = multilabel_confusion_matrix(y_test, test_predict_classifier)
    #print all the scores
    print ("Max depth:", maxd)
    print(name_classifier," score:", classifier.score(x_test, y_test))
    print (name_classifier," Balanced Accuracy Score:",balanced_accuracy_classifier)
    print (name_classifier," Crossvalidation scores:",cross_validation_classifier)
    print(classification_report(y_test,test_predict_classifier))
    #save trained classifier
    filename_classifier = foldernameoutput+enzyme+"_"+name_classifier+'_classifier.sav'
    pickle.dump(classifier, open(filename_classifier, 'wb'))

    
