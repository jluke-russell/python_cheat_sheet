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