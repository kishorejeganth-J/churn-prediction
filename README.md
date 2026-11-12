Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave (churn), with a live web app.

Live demo: https://telco-churn-kj.streamlit.app/

Problem

Telecom companies lose revenue when customers leave. If the customers most likely to leave can be identified early, the company can try to keep them. This project builds a model that estimates each customer's churn probability from their account details.

Dataset

IBM Telco Customer Churn sample dataset (available on Kaggle): 7,043 customers and 21 columns covering account tenure, contract type, services used, billing and whether the customer churned. About 26.5% of customers churned, so the classes are imbalanced. The data file is not included in this repository; download it from Kaggle.

Approach
Cleaning: dropped the customerID column; converted TotalCharges from text to numbers; filled 11 blank values with 0 (these were new customers with tenure 0 who had not been billed yet).
Exploratory analysis: compared churn rates across contract type, internet service, payment method, tenure and monthly charges.
Features: one-hot encoded categorical columns (30 numeric features in total).
Split: stratified 80/20 train/test split (5,634 training and 1,409 test customers) so both sets keep the same churn rate.
Models: logistic regression (with feature scaling), random forest and XGBoost.
Evaluation: accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices on the held-out test set.
Threshold tuning: chose the decision threshold using 5-fold cross-validation on the training data only, then evaluated once on the test set.
Results

Default threshold (0.5), test set of 1,409 customers:

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.807	0.658	0.567	0.609	0.842
Random Forest	0.793	0.637	0.511	0.567	0.826
XGBoost	0.798	0.648	0.521	0.578	0.839

Logistic regression performed best, so it was used for the app.

Why accuracy is not enough: about 73.5% of customers do not churn, so a model that always predicts "stays" is already about 73.5% accurate. At the default threshold, the logistic regression model missed 162 of the 374 customers who actually churned (43%).

After threshold tuning (0.35):

Metric	Threshold 0.5	Threshold 0.35
Recall	0.567	0.709
Churners missed	162	109
Precision	0.658	0.544
False alarms	110	222

Lowering the threshold catches more customers who would leave, at the cost of more false alarms. Whether that trade-off is worth it depends on the cost of a retention offer compared with the cost of losing a customer.

The app

The Streamlit app takes a customer's details and returns the churn probability, and labels the customer high or low risk using the 0.35 threshold.

High risk example Low risk example

Run it locally

Python 3.12 is recommended. The saved model needs scikit-learn 1.6.1.

pip install -r requirements.txt
streamlit run app.py
Files
app.py - Streamlit app
churn_model.pkl - saved model, feature columns and decision threshold
requirements.txt - dependencies
notebook.ipynb - data cleaning, analysis and model training
Limitations
The model was evaluated on a single train/test split of one dataset.
In the app, total charges are estimated as tenure times monthly charges, while the model was trained on the dataset's actual total charges.
The decision threshold reflects an assumption that missing a churner costs more than a false alarm; a real company would set it from actual costs.
Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Streamlit, joblib
