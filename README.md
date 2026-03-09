# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset.
2. Remove unnecessary columns and convert categorical data using dummy variables.
3. Separate the dataset into features (X) and target variable (Y).
4. Split the data into training and testing sets.
5. Create and train the Linear Regression model using training data.
6. Evaluate the model using 5-fold cross-validation and calculate average R² score.
7. Predict test data values and compute MSE, MAE, and R²; plot actual vs predicted prices. 

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
data= pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
data = data.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data,drop_first=True)
print(data.head())
X=data.drop('price',axis=1)
y=data['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
print('Name: K Sadhana')
print('Reg. No: 212225240128 ')
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model, X, y, cv=5)
print("Fold R² scores:", {f"{score:.4f}" for score in cv_scores})
print(f"Average R²: {cv_scores.mean():.4f}")
y_pred=model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

```

## Output:
<img width="740" height="248" alt="image" src="https://github.com/user-attachments/assets/69b95e6e-ea34-4e72-b8e5-a2d33e70ec9c" />

<img width="1089" height="683" alt="image" src="https://github.com/user-attachments/assets/914549c7-e9df-46b4-b6d5-831f49bcb729" />

## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
