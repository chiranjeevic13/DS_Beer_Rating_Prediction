# Beer Rating Prediction

## Overview
This project aims to build a Machine Learning model to predict the overall rating of beers based on various features available in the dataset. The dataset contains reviews of different beers, along with several attributes like style, alcohol by volume (ABV), user demographics, and textual reviews.

## Dataset
The dataset used in this project is `train.csv`, which contains the following columns:

- `index`: An identifier for the review.
- `beer/ABV`: Alcohol by volume of the beer.
- `beer/beerId`: A unique ID indicating the beer reviewed.
- `beer/brewerId`: A unique ID indicating the brewery.
- `beer/name`: Name of the beer.
- `beer/style`: Style of the beer.
- `review/appearance`: Rating of the beer's appearance (1.0 to 5.0).
- `review/aroma`: Rating of the beer's aroma (1.0 to 5.0).
- `review/overall`: Overall rating of the beer (1.0 to 5.0).
- `review/palate`: Rating of the beer's palate (1.0 to 5.0).
- `review/taste`: Rating of the beer's taste (1.0 to 5.0).
- `review/text`: The text of the review.
- `review/timeStruct`: A dictionary specifying when the review was submitted.
- `review/timeUnix`: Unix timestamp of the review submission time.
- `user/ageInSeconds`: Age of the user in seconds.
- `user/birthdayRaw`: User's birthday in raw format.
- `user/birthdayUnix`: Unix timestamp of the user's birthday.
- `user/gender`: Gender of the user (if specified).
- `user/profileName`: Profile name of the user.

## Goals
1. **Data Cleaning and Preprocessing**: Handle missing values and convert categorical variables for analysis.
2. **Feature Engineering**: Create relevant features from the dataset to enhance model performance.
3. **Modeling**: Implement multiple machine learning models to predict overall beer ratings, treating the problem as a binary classification task.
4. **Model Validation**: Evaluate model performance using multiple metrics, including accuracy, F1 score, precision, and recall.

## Approach
1. **Data Loading and Exploration**: Load the dataset using Pandas and perform initial exploration to understand the data.
2. **Data Visualization**: Visualize the distribution of ratings and the impact of various features using Matplotlib and Seaborn.
3. **Data Preprocessing**:
   - Fill missing values with empty strings.
   - Convert categorical features to appropriate types.
   - One-hot encode categorical variables and vectorize text reviews using TF-IDF.
4. **Model Training**:
   - Split the dataset into training and testing sets.
   - Train multiple models, including Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, and AdaBoost Classifier.
5. **Model Evaluation**: Evaluate and compare the performance of each model using accuracy, F1 score, precision, and recall.

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

## Installation
To run this project, ensure you have the required libraries installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Results
### Model Evaluation Metrics
- **Logistic Regression**
  - Accuracy: 0.8843
  - F1 Score: 0.9344
  - Precision: 0.8965
  - Recall: 0.9755

- **Random Forest Classifier**
  - Accuracy: 0.8555
  - F1 Score: 0.9210
  - Precision: 0.8552
  - Recall: 0.9978

- **Gradient Boosting Classifier**
  - Accuracy: 0.8620
  - F1 Score: 0.9237
  - Precision: 0.8661
  - Recall: 0.9896

- **AdaBoost Classifier**
  - Accuracy: 0.8679
  - F1 Score: 0.9254
  - Precision: 0.8840
  - Recall: 0.9709

## Conclusion
This project provides insights into predicting beer ratings using various features, demonstrating how machine learning techniques can be applied to real-world datasets. The Logistic Regression model achieved the highest accuracy, indicating its effectiveness for this prediction task.
