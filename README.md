# Predicting Credit Card Approvals (Classification - Supervised Learning)
### Overview
![Approval_image](https://github.com/user-attachments/assets/ea03e11e-cd58-4582-9168-5a2ef257189f)

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this workbook, you will build an automatic credit card approval predictor using machine learning techniques, just like real banks do.</p>
In this project, I explore a machine learning classification problem focused on predicting credit card approval status based on applicant information. The dataset includes 690 samples with features such as demographic information, financial status, and employment history.</p>

### Dataset Description

| Old Column Name | Column Name    | Description                     | Data Type |
|-----------------|----------------|---------------------------------|-----------|
| 0               | gender         | Gender of the applicant         | Object    |
| 1               | age            | Age of the applicant            | Object    |
| 2               | debt           | Total debt                      | Float64   |
| 3               | married        | Marital status                  | Object    |
| 4               | bankCustomer   | Bank customer status            | Object    |
| 5               | educationLevel | Education level                 | Object    |
| 6               | ethnicity      | Ethnicity                       | Object    |
| 7               | yearsEmployed  | Years of employment             | Float64   |
| 8               | priorDefault   | History of previous defaults    | Object    |
| 9               | employed       | Employment status               | Object    |
| 10              | creditScore    | Credit score                    | Int64     |
| 11              | citizen        | Citizenship status              | Object    |
| 12              | income         | Annual income                   | Int64     |
| 13              | approvalStatus | Approval status (target)        | Object    |
Dataset from the UCI Machine Learning Repository.

### Project Objectives
The main goal of this project is to build a machine learning model that can accurately predict the approvalStatus of credit card applications based on the provided features. I aimed to:

1. Perform Exploratory Data Analysis (EDA) to understand the numerical and categorical features of the dataset.
2. Preprocess the data by handling missing values and encoding categorical variables.
3. Apply feature scaling to enhance model performance.
4. Test multiple classification algorithms to identify the best-performing model.
5. Evaluate and optimize model accuracy with metrics such as precision, recall, and F1-score.

### Methods
#### 1. Exploratory Data Analysis (EDA)
##### 1.1 Data Cleaning
- First, we reviewed the dataset to identify missing values and found that it did not contain null values but had entries marked with a '?' symbol, likely indicating missing information. For simplicity, we removed any records with these placeholders and changed the data type of the 'age' column from Object to Float64.
```python
cc_apps.drop(cc_apps[cc_apps.isin(["?"]).any(axis=1)].index, inplace=True)
cc_apps["age"] = cc_apps["age"].astype("float")
cc_apps.info()
```
- The dataset columns were renamed for clarity (e.g., "age," "debt," "education level") to facilitate interpretation and analysis.
![image](https://github.com/user-attachments/assets/1139f692-3c29-4b6a-b2ec-1f70a04d5a69)

##### 1.2 Feature Analysis
- The data in numerical (e.g. debt, years employed). Using describe() functions, we gained a high-level summary of the data distribution and feature types.
```python
numerical_columns.describe()
```
![image](https://github.com/user-attachments/assets/c8cab214-b4a2-4d92-b679-5060c3b4d2fb)
I found that some numeric features have a high standard deviation; therefore, for the performance of the model, we need to standardize them.

##### 1.3 Visualization
- We used histograms to visualize the distribution of numerical features, which indicated skewness in age, debt, income, years employed, and credit score.
```python
numerical_columns.hist(figsize=(12,12),bins=20)
```
![image](https://github.com/user-attachments/assets/e79324ae-85f2-4833-b446-6803bcf6279c)
![image](https://github.com/user-attachments/assets/a547e165-5f4f-4bfd-adb8-3ac755306900)
![image](https://github.com/user-attachments/assets/d4b341bc-30ef-46d7-acd0-bb4002a77df5)

- Box plots were employed to examine the relationship between approval status and each numerical variable, highlighting outliers and trends in features like debt and years employed.
```python
for column in numerical_columns.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=cc_apps["approvalStatus"], y=cc_apps[column], showfliers=False, palette=['green', 'red'])
    plt.title(f'Boxplot of {column} by Approval Status')
    plt.xlabel("Approval Status")
    plt.ylabel(column)
    plt.show()
```
![image](https://github.com/user-attachments/assets/25e716c4-0d90-44b8-8c06-7b0138701a3b)
![image](https://github.com/user-attachments/assets/45f1bc42-f5f7-41b0-a899-c23d058cc02d)
![image](https://github.com/user-attachments/assets/bdb72f9a-2426-436d-ab38-b15e38a8d009)
![image](https://github.com/user-attachments/assets/77113485-40c4-439b-86b8-d27b7343901a)
![image](https://github.com/user-attachments/assets/0a9f179f-ffff-487f-91a6-47211b2fd754)
<p>I found that years of employment, credit score, and income impact the changes in credit card approval, as shown in the boxplot above.</p>

- Count plots were created for categorical variables to observe the class distribution across approval statuses, providing insight into the proportion of approved and declined applications for each category (e.g., gender, education level).
```python
for col in categorical_columns.columns:
    print(f"\033[1m{col}\033[0m\n")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=cc_apps, x=col, hue="approvalStatus", palette=['green', 'red'])
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f'Count of {col} by Approval Status')
    plt.legend(title="Approval Status")
    plt.show()
```
![image](https://github.com/user-attachments/assets/0164cda9-8687-4820-8003-be92b8c3e3e5)

In the categorical features, I gained insights into how they affect approval or decline rates per category. However, I would like to highlight the outstanding feature that clearly differentiates between categories regarding approval or decline: the history of previous defaults.

- Lastly, we generated a correlation heatmap to understand the relationships among numerical features, which revealed mild correlations between features like age and years employed.
```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cc_apps.corr(), ax=ax, annot=True, cmap="crest")
plt.show()
```
![image](https://github.com/user-attachments/assets/4bfba638-6c87-4baa-8949-bbfdaac4545b)

#### 2. Modeling
##### 2.1 Data Preprocessing
- Encoding techniques were used to convert categorical features to numerical representations, necessary for model training.
```python
cc_apps_encoded = pd.get_dummies(cc_apps, drop_first=True)
cc_apps_encoded.head()
```
![image](https://github.com/user-attachments/assets/a2bfd332-eb45-4809-94e9-8ea2430fd3dc)

- Define X is features and y is target that I want to predict and split to train and test to reduce overfitting in model



