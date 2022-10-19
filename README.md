# All_In_One_Classification
There is no longer need to execute any classification algorithms and calculate the accuracy of each model to select the best one.
It’s simply prepared for you. You just nead to call "All_In_One_Classification" and give "x" and "y" to it as a Pandas DataFrame
and while you enjoy drink a cup of coffee, this amazing model prepare the result for you.
I put 9 machine learning classification algorithm in it including:
LDA, QDA, Naive Bayes, Logestic Regression, KNN, Decision Tree, Bagging, Random Forrest, Boosting
Additionally, there is no need to estimate parameters for some algorithms that required. 
I used Cross Validation method for it. For instance K in KNN, tree's depth in Decision tree and so on.
The outpout would be somthing like it:

|   |Algorithm|MeanAccuracy|STDAccuracy|
|---|---------|------------|-----------|
|4   |               KNN |     0.916116 |    0.010227|
|7   |    Random Forrest  |    0.909091  |   0.011975|
|6    |          Bagging  |    0.902466   |  0.010740|
|5   |     Decision Tree  |    0.831598  |   0.014698|
|8   |          Boosting  |    0.792452 |    0.013393|
|2    |      Naive Bayes  |    0.725909  |   0.016227|
|3 | Logestic Regression  |    0.704711  |   0.015353|
|1   |               QDA  |    0.704614  |   0.015394|
|0    |              LDA  |    0.702300  |   0.015633|

Th best method for this data set is KNN with 91.6% mean accuracy and 0.01 standard deviation.
There is a sample model to examine All_In_One_Classification.
Use ths codes to perform the model:

df=pd.read_csv('churn-dataset-c.csv')

x=df.iloc[:,:-1]

y=df.iloc[:,-1]

All_In_One_Classification(x, y)
