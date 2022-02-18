import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pickle
import pyplot as plt
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
def train_classifier_and_get_accuracies(classifier,name_classifier, enzyme, x_data,y_data,x_train,y_train,x_test,y_test, foldernameoutput ):
    classifier = classifier.fit(x_train,y_train)
    #predict for test set
    test_predict_classifier=classifier.predict(x_test)
    #calculate accuracy
    cross_validation_classifier = cross_val_score(classifier, x_data, y_data, cv=5, scoring='f1_macro')
    balanced_accuracy_classifier=balanced_accuracy_score(y_test, test_predict_classifier)
    #print all the scores
    print(enzyme," ",name_classifier," score:", classifier.score(x_test, y_test))
    print (enzyme," ",name_classifier," Balanced Accuracy Score:",balanced_accuracy_classifier)
    print (enzyme," ",name_classifier," Crossvalidation scores:",cross_validation_classifier)
    print(enzyme," ",classification_report(y_test,test_predict_classifier))
    #save trained classifier
    filename_classifier = foldernameoutput+enzyme+"_"+name_classifier+'_classifier.sav'
    pickle.dump(classifier, open(filename_classifier, 'wb'))
def optimize_leaf_number(classifier,name_classifier, enzyme, x_data,x_train,y_train,x_test,y_test, foldernameoutput ):

        balanced_accuracy=0.50
        #determine best mnimum number of leafes
        leafdiagr=pd.DataFrame(columns=['Minimum samples per leaf', 'Balanced accuracy'])
        for minleaf in range (1,5):

            classifier = classifier.fit(x_train,y_train)
            test_predict_classifier=classifier.predict(x_test)
            balanced_accuracy_new=balanced_accuracy_score(y_test, test_predict_classifier)
            new_line={'Minimum samples per leaf':minleaf, 'Balanced accuracy':balanced_accuracy_new}
            leafdiagr=leafdiagr.append(new_line, ignore_index=True)
            if balanced_accuracy_new>balanced_accuracy:
                bestminleaf=minleaf
                balanced_accuracy=balanced_accuracy_new
            print (leafdiagr)
        print ("Best minimum samples per leaf:", bestminleaf) 
    
        #plot diagram of best minleaf
        plt.plot('Minimum samples per leaf', 'Balanced accuracy', data=leafdiagr,color='black')
        plt.xlabel('Minimum samples per leaf')
        plt.ylabel('Balanced accuracy')
        plt.savefig(foldernameoutput+"_"+enzyme+"_"+name_classifier+"leafdiagr.png",format="png")
        plt.show()
        return bestminleaf
def optimize_depth_classifier(classifier, name_classifier, enzyme, foldernameoutput, x_train,y_train, x_test, y_test):
        balanced_accuracy=0.50
        #determine best mnimum number of leafes
        depthdiagr=pd.DataFrame(columns=['Maximal depth of random forest', 'Balanced accuracy'])
        for maximum_depth in range (10,20):

            classifier = classifier.fit(x_train,y_train)
            test_predict_classifier=classifier.predict(x_test)
            balanced_accuracy_new=balanced_accuracy_score(y_test, test_predict_classifier)
            new_line={'Maximal depth':maximum_depth, 'Balanced accuracy':balanced_accuracy}
            depthdiagr=depthdiagr.append(new_line, ignore_index=True)
            if balanced_accuracy_new>balanced_accuracy:
                bestmaximum_depth=maximum_depth
                balanced_accuracy=balanced_accuracy_new

        print ("Best Max depth:", bestmaximum_depth) 
    
        #plot diagram of best minleaf
        plt.plot('Maximal depth', 'Balanced accuracy', data=depthdiagr,color='black')
        plt.xlabel('Maximal depth')
        plt.ylabel('Balanced accuracy')
        plt.savefig(foldernameoutput+"_"+enzyme+"_"+name_classifier+"depthdiagr.png",format="png")
        plt.show()
        return bestmaximum_depth
       
    
