# Hotel Booking Cancellation Prediction

This project analyzes hotel booking data and builds machine learning models to predict whether a booking will be canceled. The goal is to identify high-risk reservations using booking details such as lead time, deposit type, customer type, market segment, previous cancellations, Average Daily Rate, and special requests.

## Project Description

Hotel booking cancellations can create major business challenges for hotels. Cancellations can lead to empty rooms, lost revenue, poor demand forecasting, and difficulty planning staffing or room availability. By predicting which bookings are more likely to be canceled, hotels can make better operational and revenue management decisions.

In this project, I performed data cleaning, exploratory data analysis, preprocessing, and machine learning classification to predict booking cancellations. I trained and compared multiple models, including Logistic Regression, Random Forest, Decision Tree, and XGBoost.

The best-performing model was **XGBoost**, which achieved:

- **88.5% accuracy**
- **86.2% precision**
- **82.2% recall**
- **84.1% F1 score**

## Dataset

The dataset contains hotel booking records with information about each reservation.

Some important columns include:

- `hotel` - Type of hotel
- `lead_time` - Number of days between booking date and arrival date
- `arrival_date_month` - Month of arrival
- `stays_in_weekend_nights` - Number of weekend nights booked
- `stays_in_week_nights` - Number of week nights booked
- `adults`, `children`, `babies` - Number of guests
- `meal` - Meal package selected
- `country` - Country of origin
- `market_segment` - Market segment of the booking
- `distribution_channel` - Booking distribution channel
- `is_repeated_guest` - Whether the guest is a repeated guest
- `previous_cancellations` - Number of previous cancellations by the customer
- `previous_bookings_not_canceled` - Number of previous non-canceled bookings
- `reserved_room_type` - Room type reserved
- `assigned_room_type` - Room type assigned
- `booking_changes` - Number of booking changes made
- `deposit_type` - Deposit type used for the booking
- `customer_type` - Type of customer
- `adr` - Average Daily Rate
- `required_car_parking_spaces` - Number of parking spaces requested
- `total_of_special_requests` - Number of special requests made
- `is_canceled` - Target variable

The target variable is:

```text
is_canceled

where:

0 = booking was not canceled
1 = booking was canceled
Project Goal

The goal of this project is to build a classification model that can predict whether a hotel booking will be canceled.

This is a binary classification problem because the target variable has two possible outcomes:

Not canceled
Canceled
Tools and Libraries Used
Python
pandas
NumPy
matplotlib
seaborn
scikit-learn
XGBoost
Jupyter Notebook
Project Workflow

1. Data Loading

The dataset was loaded using pandas. I inspected the dataset using basic commands to understand the number of rows, columns, data types, missing values, and summary statistics.

df.head()
df.info()
df.describe()
df.isnull().sum()

2. Data Cleaning

The dataset was cleaned before modeling. The cleaning process included:

Checking missing values
Handling null or not-applicable values
Checking data types
Removing unnecessary columns
Removing leakage columns
Data Leakage Handling

Two columns were removed before modeling:

reservation_status
reservation_status_date

These columns were removed because they reveal the final outcome of the booking. For example, reservation_status can directly show whether a booking was canceled, checked out, or marked as a no-show.

Using these columns would cause data leakage, meaning the model would be trained using information that would not be available at the time of prediction. Removing them makes the model more realistic.

3. Exploratory Data Analysis

Exploratory Data Analysis was performed to better understand cancellation patterns.

The analysis focused on questions such as:

What percentage of bookings were canceled?
Which hotel type had more cancellations?
Does lead time affect cancellation?
Does deposit type affect cancellation?
Are repeated guests less likely to cancel?
Does market segment influence cancellation?
Do guests with more special requests cancel less often?
How does Average Daily Rate relate to cancellations?

Important features explored included:

hotel
lead_time
deposit_type
market_segment
distribution_channel
previous_cancellations
customer_type
adr
total_of_special_requests

4. Key EDA Findings

Some important patterns found during EDA were:

Bookings with longer lead times were more likely to be canceled.
Deposit type showed a strong relationship with cancellation behavior.
Customers with previous cancellations were more likely to cancel again.
Bookings with more special requests appeared less likely to be canceled.
Market segment and distribution channel showed different cancellation patterns.
Average Daily Rate had some relationship with cancellation behavior.
Repeated guests appeared to behave differently from first-time guests.

These findings helped guide the modeling process and gave business insight into what factors may influence cancellations.

5. Feature Preparation

The target variable was separated from the input features.

X = df.drop(columns=["is_canceled", "reservation_status", "reservation_status_date"])
y = df["is_canceled"]

The dataset contained both numerical and categorical features.

Numerical features included columns such as:

lead_time
adults
children
babies
previous_cancellations
booking_changes
days_in_waiting_list
adr
required_car_parking_spaces
total_of_special_requests

Categorical features included columns such as:

hotel
arrival_date_month
meal
country
market_segment
distribution_channel
reserved_room_type
assigned_room_type
deposit_type
customer_type

6. Preprocessing

A preprocessing pipeline was created using scikit-learn.

The preprocessing steps included:

Scaling numerical columns using StandardScaler
Encoding categorical columns using OneHotEncoder

This was necessary because machine learning models cannot directly use text-based categorical values such as "City Hotel", "Resort Hotel", "No Deposit", or "Online TA".

7. Train-Test Split

The data was split into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

An 80/20 split was used:

80% of the data was used for training
20% of the data was used for testing

The stratify=y argument was used to keep the cancellation ratio similar in both the training and testing sets.

8. Models Trained

Four classification models were trained and compared:

Logistic Regression
Random Forest Classifier
Decision Tree Classifier
XGBoost Classifier

9. Model Evaluation Metrics

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
Accuracy

Accuracy measures the percentage of total predictions that were correct.

Precision

Precision measures how many bookings predicted as canceled were actually canceled.

Recall

Recall measures how many actual cancellations the model correctly identified.

F1 Score

F1 score balances precision and recall. This is useful because both false positives and false negatives matter in cancellation prediction.

10. Model Results

Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	0.8190	0.8120	0.6656	0.7315
Random Forest	0.7655	0.9985	0.3674	0.5372
Decision Tree	0.8180	0.7956	0.6847	0.7360
XGBoost	0.8852	0.8619	0.8217	0.8413

11. Best Model

The best-performing model was XGBoost.

XGBoost achieved:

Accuracy: 88.5%
Precision: 86.2%
Recall: 82.2%
F1 Score: 84.1%

XGBoost performed best overall because it had the highest accuracy, recall, and F1 score among the models tested.

This means XGBoost was the strongest model for identifying hotel booking cancellations while maintaining a good balance between precision and recall.

12. Business Interpretation

The model can help hotels identify bookings that are likely to be canceled before the arrival date.

This can help hotels:

Improve room availability planning
Reduce revenue loss from cancellations
Adjust overbooking strategies
Identify high-risk reservations
Send reminders or incentives to customers
Better understand cancellation behavior
Improve demand forecasting

For example, bookings with long lead times, previous cancellations, certain deposit types, or specific market segments may be flagged as higher risk.

13. Project Structure

hotel-booking-cancellation/
│
├── hotel_booking_cancellation.ipynb
├── data.csv
├── README.md
└── requirements.txt

14. How to Run This Project

Clone the repository:

git clone https://github.com/nannurinirupamreddy/hotel_booking_cancellation.git

Navigate into the project folder:

cd hotel-booking-cancellation

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

Open the Jupyter Notebook:

jupyter notebook hotel_booking_cancellation.ipynb

Run all cells in order.

15. Requirements

You can create a requirements.txt file with:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter

16. Limitations

This project has a few limitations:

The model was trained on historical booking data, so future booking behavior may change.
External factors such as weather, local events, flight prices, and competitor hotel prices were not included.
Some booking features may only be available after certain customer interactions.
The model should be monitored and retrained if used in a real hotel system.

17. Future Improvements

Possible future improvements include:

Hyperparameter tuning for XGBoost
Cross-validation for more reliable model evaluation
Feature importance analysis
SHAP analysis for model explainability
Testing additional models such as LightGBM or CatBoost
Building a simple web app for cancellation prediction
Creating a dashboard to visualize cancellation risk

18. Final Conclusion

This project demonstrates how machine learning can be used to predict hotel booking cancellations using structured booking data.

After cleaning the data, performing exploratory analysis, removing leakage columns, preprocessing features, and comparing multiple classification models, XGBoost achieved the best performance with 88.5% accuracy and 84.1% F1 score.

The results show that hotel booking cancellations can be predicted using features such as lead time, deposit type, market segment, previous cancellations, ADR, customer type, and special requests.

This type of model can help hotels make better business decisions, improve room planning, and reduce revenue loss caused by cancellations.
