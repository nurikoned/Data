Project 1: **Predictive Modeling with Linear Regression**

1. We selected 1000 sample data points from a dataset containing real estate information for testing purposes.
2. Any missing values (NaN) in the dataset were handled by either removing them or replacing them with the mean/mode of their respective columns.
3. We analyzed the correlation between the 'acre_lot', 'house_size', and 'price' columns.
4. Using the Seaborn library, regression plots were generated for the relationships between 'acre_lot' and 'house_size' with 'price'.
5. The dataset was divided into training and testing sets using the train_test_split function, with an 80/20 split ratio.
6. A Linear Regression model was instantiated and trained using the training data.
7. Predicted values of 'price' (Yhat) were generated using the trained multiple linear regression model.
8. A distribution plot (Distplot) comparing the actual and predicted values of 'price' was created to visualize their distributions.1. 

Project 3: **Predicting Correct Drug using Decision Tree**

1. We gathered a dataset comprising 200 rows and 6 columns, including Age, Sex, Blood Pressure (BP), Cholesterol level, Sodium-to-Potassium ratio (Na_to_K), and the corresponding Drug prescribed.
2. The dataset was divided into predictor variables (x) and the target variable (y).
3. Categorical variables in the predictor set were encoded into numerical values using LabelEncoder to facilitate compatibility with Decision Trees.
4. The dataset was further divided into training and testing subsets using the train_test_split method.
5. A DecisionTreeClassifier was instantiated and trained on the training data.
6. The trained model was then used to predict the target variable (y) for the test dataset (x_test).
7. The accuracy of the model was assessed and found to be 100%.

Project 4: **Diabetes Predictor using Logistic Regression**

   The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases, aimed at predicting diabetes diagnosis based on specific diagnostic measurements. Here's how we processed the data:
1. **Data Inspection**: We started by importing the dataset and conducting initial checks including examining its shape, identifying null values, and inspecting data types.
2. **Data Preparation**: Next, we split the dataset into two parts: independent variables (X) and the dependent variable (y), representing diagnostic measurements and diabetes diagnosis respectively. We also normalized the data to ensure uniformity across features.
3. **Train-Test Split**: To evaluate the model's performance, we divided the dataset into training and testing sets, with the testing set comprising 20% of the total data.
4. **Model Fitting**: We applied logistic regression to train the model on the training data, enabling it to learn patterns and relationships between the diagnostic measurements and diabetes diagnosis.
5. **Model Evaluation**: Using the trained model, we made predictions on the test data. To quantify the model's accuracy, we calculated the Jaccard score, providing insight into its predictive performance.
6. **Confusion Matrix Visualization**: Additionally, we plotted a confusion matrix to gain a visual understanding of the model's accuracy, allowing us to assess its ability to correctly classify instances of diabetes and non-diabetes.
