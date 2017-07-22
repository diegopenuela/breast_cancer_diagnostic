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
	cancerdf = pd.DataFrame(cancer.data , columns=cancer.feature_names)
	cancerdf = cancerdf.assign(target=pd.Series(cancer.target))
	return cancerdf

def answer_two():
	cancerdf = answer_one()
	# Your code here
	return # Return your answer


cancer = load_breast_cancer()

#print ("cancer")
#print (type(cancer))
#print (cancer)

#print(cancer.DESCR) # Print the data set description

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