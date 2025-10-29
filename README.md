# Recruitment Task – Classification of a Heart with Hypertrophic Cardiomyopathy (Cardiomegaly)

The task was to classify wether patient is sick or not based on the medical exams.
I creates 3 models: 
  - *K-nearest neighbours classifier*
  - *Decision Tree Classifier*
  - *Random Forest Classifier*
    
whose mission is to predict sickness of the given patient.

# Models description
### K-nearest neighbours (KNN)
KNN works by treating each input vector as a point in n-dimensional space, where n is a number of features in feature vector.
The training set is memorized and prediction is calculated as follows:
  - Place input vector (point) in space
  - Calculate distance from new point to every memorized point
  - Check k (k is some hyperparameter) closest points
  - Return most commonly occuring label among the k neigbours

### Decision Tree
Decision tree works by repedetly splitting given dataset into smaller parts based on features it is given. Given new input vector we can follow the splits the tree has made
and after we get to the end of the branch we can choose most frequewntly occuring label and return it as probable label for the given vector.


### Random Forest
Random Forest is bassicly a few Decision Trees fitted on randomly chosen subsets of given dataset. Returned label is chosen from majority vote over all trees.
That way random Forest should be more precise than a single Decision Tree

# Showcase
Example of models behaviour is shown in ``showcase_of_models.ipynb`` file 

# Benchmark 
I conducted a benchamrk of created models in ``benchmark.ipynb`` file which compares performance of each model on the same train and test datasets.
