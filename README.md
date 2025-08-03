# 📊 Risk Analytics in Banking using Exploratory Data Analysis (EDA)

## 📝 Introduction

This project is a case study that demonstrates the application of Exploratory Data Analysis (EDA) in a real-world business context, specifically within the banking and financial services industry.  
The goal is to use data to understand loan applicant behavior, identify risk patterns, and ultimately help reduce loan default rates through informed decision-making.

This case study also introduces basic concepts of risk analytics, focusing on how data can be used to minimize the risk of lending to unreliable applicants.

---

## 💼 Business Understanding

Loan providers often struggle to assess applicants with limited or no credit history. Some individuals may exploit this by defaulting on their loans.  
This project simulates a scenario where you work for a consumer finance company that offers various types of loans to urban customers.

When a loan application is received, the company must decide whether to approve or reject it. This decision involves two types of risk:

- False Negative: Approving a loan for someone likely to default (financial loss)
- False Positive: Rejecting a reliable applicant (loss of business opportunity)

The dataset includes:

- Clients with payment difficulties (late payments)
- Clients who paid on time
- Loan decisions: Approved, Cancelled, Refused, and Unused Offer

---

## 🎯 Business Objectives

The primary objective is to identify patterns and driver variables that indicate whether a client is likely to default. These insights can help the business to:

- Deny risky applicants  
- Reduce loan amounts  
- Increase interest rates for high-risk applicants  
- Avoid rejecting creditworthy customers  

By applying EDA, we aim to discover the key factors that influence loan repayment behavior and support risk-aware lending decisions.

---

## 📂 Dataset Description

The project uses the following datasets:

- application_data.csv  
  Contains information about clients at the time of loan application, including demographics, income, employment, and financial status.

- previous_application.csv  
  Includes past loan application records of clients and their final statuses (Approved, Cancelled, Refused, Unused Offer).

- columns_description.csv  
  A data dictionary that describes all variables in the datasets.

---

## 📁 Project Structure

This solution is divided into two Jupyter Notebooks:

1. `1_EDA_Application_Data.ipynb`  
   Performs EDA on the application_data.csv file to identify important features that may indicate loan default risk.

2. `2_Merged_Data_Analysis.ipynb`  
   Joins application_data.csv and previous_application.csv on the SK_ID_CURR field and conducts further analysis on the combined dataset.
