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

### Gradient Boosted Trees

- Gridsearch cv brute force searches all options 
- A better way is to narrow search by what hyper parameter combo is best (guided hyperparameter grid search)
- Both those use number of trees vs depth of the tree
- Subspace sampling uses same data across all trees but different sets of features for each tree
- Bootstap and Aggregating aka Bagging means letting every tree have different samples and then put all the findings together 
- Metrics: how wrong was I? called absolute error 
- uses mean absolute error (MAE) this is the median
- uses squared error this is the mean 
- Mean squared error (MSE) good statistical value by its not in the original value as the data (but taking the sqrt does)
- R^2 takes MSE of the model divided by the MSE of a guess of the mean of each sample (do 1 - R^2 close to 1 good, close to 0, guessing)
- Make sure to create testing, validation, and training sets and do not let your model see the testing set. This will allow you to be more confident in the results. 
- Something else big brain, create for loops that parse through all different options and throw out the columns, hyperparameters, etc that do not provide the best results. An average of those should be used for the final model. 
- Remember that the best model for the testing set may not be the best for the holdout set for various reasons. 
- Squared error reduces the impact of outliers - smooths things out. In this case two wrongs make a right (sorta).
- Normalization: rescales everything between 0-1. The issue is you end up refitting the holdout set compared to the training set. The max and mins can be different. 

### (Artificial) Neural Networks

- Mimicks the human brain (neurons and synapses)
- neurons connect to each other and pass info, the next neuron decides whether or not the previous one's decision was important or nah. 
- Learning Rate - a hyper parameter which tells the model how far to jump ahead on the line that represents the model's outputs 
- Perceptron - an individual neuron that makes a decision. 
- Need multiple layers of neurons because each neuron learns one new feature that is unique among the other neurons. Having many neurons allows you to create the features using the network that gives you the best result.
- There's a lot of calculus going on in the background of this moduel. 
- Neural networks takes linear inputs and a bias (how much to adjust things) and shoves it into a nonlinear activation function, then it moves onto the next neuron. (You can also influence how much that output has on the next neuron.)
- You can create logistic regression models to replot the data making it easier to pick out what you want. 
- Each neuron layer transforms the previous one in order to make an easier decision. 
- Backpropogate our errors to figure out how to fix our mistakes 
- Forwardpropogate - how our data is entered and flows through the network. 
 