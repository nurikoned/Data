Project 1: **Predictive Modeling with Linear Regression**

1. We selected 1000 sample data points from a dataset containing real estate information for testing purposes.
2. Any missing values (NaN) in the dataset were handled by either removing them or replacing them with the mean/mode of their respective columns.
3. We analyzed the correlation between the 'acre_lot', 'house_size', and 'price' columns.
4. Using the Seaborn library, regression plots were generated for the relationships between 'acre_lot' and 'house_size' with 'price'.
5. The dataset was divided into training and testing sets using the train_test_split function, with an 80/20 split ratio.
6. A Linear Regression model was instantiated and trained using the training data.
7. Predicted values of 'price' (Yhat) were generated using the trained multiple linear regression model.
8. A distribution plot (Distplot) comparing the actual and predicted values of 'price' was created to visualize their distributions.1. 

