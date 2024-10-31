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
- First, we reviewed the dataset to identify missing values and found that it did not contain null values but instead had entries marked with a '?' symbol, which likely indicate missing information. For simplicity, we removed any records with these placeholders.
