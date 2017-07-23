import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

#0. Calculate how many features are in the dataset
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    return len(cancer['feature_names'])

#1. Organize the dataset in a dataframe based on the features as columns. Then add a new column with the category label.
#This function should return a dataframe (269 x 31) 
def answer_one():
	#Convert the sklearn.dataset cancer to a DataFrame.
	cancerdf = pd.DataFrame(cancer.data , columns=cancer.feature_names)
	#Assign new column to a DataFrame. Add Target Column (0,1)
	cancerdf = cancerdf.assign(target=pd.Series(cancer.target))
	return cancerdf

#2. Calculate how many data samples are in the dataframe for each label
#This function should return a Series named `target` of length 2 with integer values and index =* `['malignant', 'benign']`
def answer_two():
	cancerdf = answer_one()
	#Based on a previous answer. Count data samples for each target (1,0).
	distribution = cancerdf.target.value_counts()
	#Define index as requested. Print order (1 then 0, therefore 'benign' then 'malignant')
	distribution.index = [ 'benign','malignant']
	#Series named target. 
	target = pd.Series(distribution, name="target")
	#print (distribution)  #Debug
	return target

#3. Split the DataFrame into X (the data) and y (the labels).
def answer_three():
	cancerdf = answer_one()
	# Use iloc (selectio based on positions) - http://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
	# data.iloc[<row selection>,<colum selection>]

	# In the DataFrame the data start at Column 0 - Column 30
	X = cancerdf.iloc[:,0:30]

	# In the DataFrame the labels are already in a column named 'target'
	y = cancerdf['target']

	#return a tuple of length 2: (X, y)
	output = (X,y)
	return output

#4. Using train_test_split, split X (the data) and y (tha labels) into training and test sets (X_train, X_test, y_train, and y_test).
from sklearn.model_selection import train_test_split

def answer_four():
	#Use previous answer 2-tuple to set local variables
	X, y = answer_three()

	#Use train_test_split  75% train  25% test
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	#return a tuple of length 4: (X_train, X_test, y_train, y_test)
	output = (X_train, X_test, y_train, y_test)
	return output

#5. Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
	#Use previous answer 4-tuple to set local variables
	X_train, X_test, y_train, y_test = answer_four()

	# Create classifier with k = 1
	knn = KNeighborsClassifier(n_neighbors = 1)

	# Train classifier
	return knn.fit(X_train, y_train)

#6. Using the knn classifier to predict the class label using the mean value for each feature.
# OJO - La verdad NO entendi este paso para que es
def answer_six():
	cancerdf = answer_one()
	knn = answer_five()

	#gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).
	means = cancerdf.mean()[:-1].values.reshape(1, -1)

	#predict the class label for a test sample "means"
	prediction = knn.predict(means)

	return prediction

#7. Using the knn classifier to predict the class labels for the test set X_test.
def answer_seven():
	X_train, X_test, y_train, y_test = answer_four()
	knn = answer_five()

	#predict the class labels for the test set X_test
	prediction = knn.predict(X_test)

	return prediction

#8. Find the score (mean accuracy) of your knn classifier using X_test and y_test.
def answer_eight():
	X_train, X_test, y_train, y_test = answer_four()
	knn = answer_five()

	#Estimate the accuracy of the classifier on future data, using the test data
	score = knn.score(X_test, y_test)

	return score


#9. plotting function below to visualize the differet predicition scores between training and test sets, as well as malignant and benign cells.
def accuracy_plot():
    import matplotlib.pyplot as plt

    #%matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)




cancer = load_breast_cancer()

### Information about cancer
#type(cancer)
#<class 'sklearn.datasets.base.Bunch'>

#dir(cancer)
#['DESCR', 'data', 'feature_names', 'target', 'target_names']

#print(cancer.DESCR) # Print the data set description

#print(cancer.data) # Print the data set
#array([[  1.79900000e+01,   1.03800000e+01,   1.22800000e+02, ...,
#          2.65400000e-01,   4.60100000e-01,   1.18900000e-01],

#print(cancer.feature_names) # Print the feature names
#array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#       'mean smoothness', 'mean compactness', 'mean concavity',

#print(cancer.target) # Print the target
#array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
#       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,

#print(cancer.target_names) # Print the target names
#array(['malignant', 'benign']

print ("cancer data")

#print (cancer.keys())  #Debug
#print (cancer['feature_names'])   #Debug

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
print ("Answer Zero")
print (answer_zero())
#print (cancer['feature_names'])   #Debug

print ("Answer One")
print (answer_one())

print ("Answer Two")
print (answer_two())

print ("Answer Three")
print (answer_three())

print ("Answer Four")
print (answer_four())

print ("Answer Five")
print (answer_five())

print ("Answer Six")
print (answer_six())

print ("Answer Seven")
print (answer_seven())

print ("Answer Eight")
print (answer_eight())

print ("Accuracy Plot")
accuracy_plot() 

