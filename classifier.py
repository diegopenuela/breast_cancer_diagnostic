import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

#Convert the sklearn.dataset cancer to a DataFrame.
def answer_one():

	#Convert the sklearn.dataset cancer to a DataFrame.
	cancerdf = pd.DataFrame(cancer.data , columns=cancer.feature_names)

	#Assign new column to a DataFrame. Add Target Column (0,1)
	cancerdf = cancerdf.assign(target=pd.Series(cancer.target))
	return cancerdf

def answer_two():
	cancerdf = answer_one()
	
	#Based on previous answer. Count instances for each target (1,0).
	distribution = cancerdf.target.value_counts()

	#Define index as requested. Print order (1 then 0, therefore 'benign' then 'malignant')
	distribution.index = [ 'benign','malignant']

	#Series named target. 
	target = pd.Series(distribution, name="target")

	print (distribution)
	return # Return your answer


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
print (cancer.data)
#print (type(cancer))
#print (cancer)



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