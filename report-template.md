# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

--------------------------------------------------------------------------
Purpose of the Analysis:

The purpose of this analysis was to develop a predictive model to identify whether a loan is healthy or has a high risk of defaulting. This is a critical task in the financial industry to assess the creditworthiness of borrowers and manage lending risks effectively. By creating a predictive model, the goal was to enhance the accuracy of loan risk assessment and minimize the potential losses associated with high-risk loans.

Financial Information and Prediction:

The dataset used in the analysis contained financial information related to loans. This could include features like loan size, interest rate, borrower's income, and potentially other features related. The target variable was the "loan_status" column targeted  as labels, where a value of 0 indicated a healthy loan and a value of 1 indicated a high-risk loan.


Certainly, I can provide you with an overview of the analysis based on the context you've provided:

Purpose of the Analysis:
The purpose of this analysis was to develop a predictive model to identify whether a loan is healthy or has a high risk of defaulting. This is a critical task in the financial industry to assess the creditworthiness of borrowers and manage lending risks effectively. By creating a predictive model, the goal was to enhance the accuracy of loan risk assessment and minimize the potential losses associated with high-risk loans.

Financial Information and Prediction:
The dataset used in the analysis contained financial information related to loans. This could include features like loan size, interest rate, borrower's income, and potentially other credit-related attributes. The target variable was the "loan_status" column, where a value of 0 indicated a healthy loan and a value of 1 indicated a high-risk loan.

Basic Information about the Variables:

To provide an understanding of the distribution of loan statuses, a value_counts analysis was likely performed on the "loan_status" column. This analysis would show how many loans fall into each category: healthy (0) and high-risk (1) additionally other important variables that we can count where predictions based on test data from x and shape that determine the rows from the model trained

Stages of the Machine Learning Process:

  -Data Preprocessing: This includes loading the dataset, encoding categorical variables, and splitting the data into training and testing sets.

  -Model Selection: In this analysis, the LogisticRegression model was used, given its suitability for binary classification problems like predicting loan default risk.

  -Model Training: The model was trained using the training data. In order to address class imbalance, a resampling method like RandomOverSampler from the imbalanced-learn library was employed to balance the distribution of the target classes.

  -Model Evaluation: The trained model was then evaluated on the testing data using various metrics such as accuracy, precision, recall, F1-score, and the confusion matrix. These metrics provide insights into how well the model performs in different aspects of classification.

Methods Used:

In this analysis, the primary method used was the LogisticRegression algorithm from the sklearn library. This algorithm is commonly used for binary classification tasks, additionally, to address class imbalance, the RandomOverSampler from the imbalanced-learn library was used to balance the class distribution in the training data.

Overall, this analysis aimed to build a predictive model that could effectively classify loans into healthy and high-risk categories, thus assisting financial institutions in making informed lending decisions and managing risks.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

• Balanced Accuracy Score: The balanced accuracy score was 0.95, indicating that the model achieved a high level of overall accuracy.

• Precision and Recall (Class 0): The precision for class 0 was 1.00, showing that the model correctly identified all instances predicted as healthy loans. The recall for class 0 was 0.99, indicating that the model captured a large portion of actual healthy loans in its predictions.

Precision and Recall (Class 1): For class 1 (high-risk loans), the precision was 0.85, indicating that 85% of instances predicted as high-risk loans were truly high-risk loans. The recall for class 1 was 0.91, implying that the model identified 91% of the actual high-risk loans.

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

• Balanced Accuracy Score: The balanced accuracy score was 0.99, reflecting a strong overall accuracy.

• Precision and Recall (Class 0): The precision for class 0 was 1, indicating full prediction for all healthy loans, the recall for class 0 was 0.99, meaning the model captured a substantial portion of actual healthy loans.

Precision and Recall (Class 1): For class 1, the precision was 0.84, indicating that 84% of instances predicted as high-risk loans were truly high-risk loans. The recall for class 1 was 0.99, implying that the model identified 99% of the actual high-risk loans.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.


Trained Data:

Balanced Accuracy Score: 0.95
Precision for Class 0: 1.00
Recall for Class 0: 0.99
Precision for Class 1: 0.85
Recall for Class 1: 0.91


Oversampled Trained Data:

Balanced Accuracy Score: 0.99
Precision for Class 0: 1.00
Recall for Class 0: 0.99
Precision for Class 1: 0.84
Recall for Class 1: 0.99

Comparing the results of the two models:

Both models achieve very high accuracy, which indicates that they are both effective in making correct predictions.
The oversampled model achieves a higher balanced accuracy score (0.99) compared to the model trained on the original data (0.95). This suggests that the oversampled model performs better in terms of overall accuracy.
For the oversampled model, the recall for both classes (0 and 1) is consistently higher than in the model trained on the original data. This indicates that the oversampled model is better at capturing instances of both healthy loans (0) and high-risk loans (1).

Problem Context Recommendation:

If the goal is to predict both healthy loans and high-risk loans accurately, the oversampled model seems to be the better choice due to its higher balanced accuracy score and improved recall for both classes.
If the problem context places equal importance on identifying healthy loans (0) and high-risk loans (1), the oversampled model is recommended.
Conclusion:

Based on the provided results, the oversampled model performs better in terms of balanced accuracy and recall for both classes. Therefore, the oversampled model is recommended for making predictions on this dataset, as it demonstrates better overall performance in identifying both healthy loans and high-risk loans.