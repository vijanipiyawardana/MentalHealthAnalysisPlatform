## Mental Health Analysis Platform

### $${\color{lightgreen}Project:}$$
Deep Learning Model Development for Mental Health Condition Classification

### $${\color{lightgreen}Objective:}$$
Train a deep learning model to classify mental health conditions based on the features in the dataset.

### $${\color{lightgreen}Approach:}$$
- Target Variable: Diagnosis
- Features: Use all or a selected subset of features to classify the diagnosis into various mental health conditions.
- Steps:
    - Perform feature importance analysis to identify the most relevant features.
    - Train a fastai tabular learner.
    - Evaluate the model using metrics like accuracy, precision, recall, and F1 score.

### $${\color{lightgreen}Dataset:}$$
- Dataset is taken from kaggle: Local Mental health Dataset (pakistan).
- This dataset contains sample data from questionnaire responses, obtained from The Fountain House Mental Health Institute Pakistan.
https://www.kaggle.com/datasets/mariatamoor/local-mental-health-dataset-pakistan

- The dataset contains variable Diagnosis which is used as the output variable for the mental health classification problem and all other variables (110) are features of mental health.

### $${\color{lightgreen}Project \space phases \space and \space methodology:}$$

**Data Preprocessing and Exploration**

- Used wolta.data_tools library for data preprocessing.
- Initially the dataset contained 764 rows and 111 columns.

- Handling null values, single values, unique values:
    - Deleted some columns from the dataset with null values by considering the maximum tolerated null value amount as 152.
    - Deleted one column because it has a single value.
    - Dropped some columns from the dataset with unique values by considering the maximum tolerated unique value amount is 76 in string data.
    - Handled missing data in remaining columns by filling the gaps with a placeholder value 'unknown', ensuring there are no NaN values in columns.
