import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'SquareFootage': [1500, 2500, 1800, 2000, 3000, 3500, 4000],
    'Bedrooms': [3, 4, 3, 3, 5, 4, 4],
    'Bathrooms': [2, 3, 2, 2, 3, 3, 4],
    'Price': [300000, 500000, 400000, 420000, 650000, 700000, 800000]
}


df = pd.DataFrame(data)


X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


new_data = pd.DataFrame({
    'SquareFootage': [3200],
    'Bedrooms': [4],
    'Bathrooms': [3]
})
new_prediction = model.predict(new_data)
print(f"Predicted Price for new data: {new_prediction[0]}")