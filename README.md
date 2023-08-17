# credit-risk-classification

The code performs the following tasks using Python and various libraries like NumPy, pandas, scikit-learn, and imbalanced-learn:

Data Import and Preprocessing:
Reads a CSV file called lending_data.csv into a pandas DataFrame.
Separates the dataset into features (X) and labels (y), where X contains all columns except the "loan_status" column, and y contains the "loan_status" column.
Data Splitting:
Splits the data into training and testing sets using the train_test_split function from scikit-learn.
Uses a 80-20 train-test split ratio.
Logistic Regression Model (Original Data):
Fits a logistic regression model to the original training data.
Makes predictions on the testing data.
Evaluates the model's performance by calculating accuracy, generating a confusion matrix, and printing a classification report.
Logistic Regression Model (Resampled Data):
Uses RandomOverSampler from imbalanced-learn to oversample the training data, ensuring equal representation of both classes.
Fits a logistic regression model to the resampled training data.
Makes predictions on the testing data.
Evaluates the model's performance using the same metrics as before.

Answering Questions:
Provides answers to questions about the performance of both models in predicting healthy loans (0) and high-risk loans (1).
Overall, the code demonstrates the process of importing data, splitting it into training and testing sets, building and evaluating logistic regression models both with the original data and resampled (oversampled) data to address class imbalance. The resampled model tends to perform better in predicting both healthy loans (0) and high-risk loans (1), indicating improved performance due to handling class imbalance.
