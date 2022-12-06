import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

#Investigate the data and features
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

#Investigate the labels to be classified
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

#Import train_test_split
from sklearn.model_selection import train_test_split

#Split the data into training and validation sets
training_set, validation_set, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

#Verify data in training and validation sets
print(len(training_set))
print(len(validation_set))
print(len(training_labels))
print(len(validation_labels))

#Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Create K Neighbor classifier and test its accuracy
classifier = KNeighborsClassifier(n_neighbors = 3)

#Train the classifiers using the fit function. Use training set and labels
classifier.fit(training_set, training_labels)

#Test the classifiers accuracy
print(classifier.score(validation_set, validation_labels))

#Create an accuracy list
accuracies = []

#Create a for loop to find the best k score from 1 to 100.
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_set, training_labels)
  score = classifier.score(validation_set, validation_labels)
  accuracies.append(score)

#Visualize the classifier score
#Start by importing matplotlibdi
import matplotlib.pyplot as plt

#Create a range from 1 to 100
k_list = range(1, 101)

#Create plot
plt.plot(k_list, accuracies)
plt.xlabel('K')
plt.ylabel('Validation Accuracy')
plt.show()