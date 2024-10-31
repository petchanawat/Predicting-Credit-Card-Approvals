# Predicting Credit Card Approvals (Classification - Supervised Learning)
### Overview
![Approval_image](https://github.com/user-attachments/assets/ea03e11e-cd58-4582-9168-5a2ef257189f)
Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this workbook, you will build an automatic credit card approval predictor using machine learning techniques, just like real banks do.</p>
In this project, I explore a machine learning classification problem focused on predicting credit card approval status based on applicant information. The dataset includes 690 samples with features such as demographic information, financial status, and employment history.</p>

### Dataset Description

| Old Column Name | Column Name    | Description                     | Data Type |

|-----------------|----------------|---------------------------------|-----------|
| 0               | gender         | Gender of the applicant         | Object    |
| 1               | age            | Age of the applicant            | Object   |
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

### Project Objectives
The main goal of this project is to build a machine learning model that can accurately predict the approvalStatus of credit card applications based on the provided features. I aimed to:

1. Preprocess the data, handling missing values and encoding categorical features.
2. Use feature scaling to improve model performance.
3. Test several classification algorithms to determine the best-performing model.
4. Evaluate and optimize model accuracy using metrics like precision, recall, and F1-score.
