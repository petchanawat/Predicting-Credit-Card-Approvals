# Predicting Credit Card Approvals (Classification - Supervised Learning)
### Overview
Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this workbook, you will build an automatic credit card approval predictor using machine learning techniques, just like real banks do.</p>
In this project, I explore a machine learning classification problem focused on predicting credit card approval status based on applicant information. The dataset includes 690 samples with features such as demographic information, financial status, and employment history.</p>

### Dataset Description

| Column         | Description                     | Data Type |
|----------------|---------------------------------|-----------|
| gender         | Gender of the applicant         | Object    |
| age            | Age of the applicant            | Float64   |
| debt           | Total debt                      | Float64   |
| married        | Marital status                  | Object    |
| bankCustomer   | Bank customer status            | Object    |
| educationLevel | Education level                 | Object    |
| ethnicity      | Ethnicity                       | Object    |
| yearsEmployed  | Years of employment             | Float64   |
| priorDefault   | History of previous defaults    | Object    |
| employed       | Employment status               | Object    |
| creditScore    | Credit score                    | Int64     |
| citizen        | Citizenship status              | Object    |
| income         | Annual income                   | Int64     |
| approvalStatus | Approval status (target)        | Int64     |
