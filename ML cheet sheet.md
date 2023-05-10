# Machine Learning Cheet Sheet / Notes 

## Model Fitting 
 - Overfitting: making a model that tries to include everything to where it belongs no matter what 
    - Solve this by making training and testing data by splitting the original data provided. 
    - Report the percent accuracy. 
## Errors 
- The more you train the model, the more error you will introduce over time into the validation data 
## K nearest neighbor (KNN)
- Looks at the data that is closest to it - what data points are close to the new point? 
    - This uses those jank distance formulas that were real whack like. 
- If 2/3 points that were near to the new one were red for example, then we can predict that the new point would be red. 
    - The larger the K, the more smooth the prediction process is - don't make too big tho - introduces more error. 
## Confusion Matrix 

True Positive  | False Negative 
---------------|----------------  
False Positive | True Negative 

Precision = TP / TP + FP

Recall = TP / TP + FN

F1 Score = combo of Precision and Recall

## Lesson Two: 4/28

Machine learning models and AI can be over confident when over trained thus producing over-fitted models. 

### Classification Model
- Accuracy 
- Precision 
- Recall
- F1 Score (Precision and Recall)

### Regression Model 
- Absolute error - How much you are off by 
- Mean Squared Error 
- RMSE - puts MSE back into the original units 
- r^2 - how good the model is at predicting results on a scale of 0 - 1 

### Decision Trees
- data is split by decisions made creating different outcomes by using one demention or feature at a time. 
- looks like a flow chart with the splits created by data falling into one bucket or another. 
- A leaf node (last boxes of trees) with a small sample size within it means the model is being overfitted - deeper a tree goes, the more likely it is to overfit the data. 
- Changing the perspective or feature you start with will change how the splits are made. 
- Avoid overfitting by making multiple trees or a forest hahaha random forest classifier
- If you one-hot encode the holdout data may not include values for some of the new one-hot encodes. you can make a fake column to solve this issue 
- Make a vanilla (or baseline) model to have something to compare against and see improvements. 

### Entropy 
- Presence of level of randomness
- log2 (1/P(x)) gives us the number of bits --> -log2 P(x)
- Entropy = SUM P(xi) log2 P(xi)
- Model is bring trained correctly if the info is being split and there is less info being produced -> meaning which model has the lowest entropy.

### New Stuff
