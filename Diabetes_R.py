import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#%% Import data

diabetes = load_diabetes()
diabetes_df = pd.DataFrame(data = diabetes.data, columns = diabetes.feature_names)
diabetes_df['target'] = diabetes.target

#%% Visualize data
print(diabetes_df.head())

#%% Split data
X = diabetes_df.drop('target', axis = 1)
y = diabetes_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#%% Develop regressor models to compare

models = {"Linear Regression": LinearRegression(),
          "Ridge Regression": Ridge(),
          "Lasso Regression": Lasso(),
          "Random Forest Regressor": RandomForestRegressor(),
          "Support Vector Regressor": SVR()
          }

for model_name, model in models.items():
    
    # train
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    
    # evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # print
    print(f"Model: {model_name}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print()
    
    # visualize predicted vs. actual
    # plt.scatter(y_test, y_pred)
    plt.plot(np.arange(0,len(y_test)),y_test)
    plt.plot(np.arange(0,len(y_pred)),y_pred)
    plt.legend()
    # plt.xlabel("Actual Values")
    # plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted ({model_name}, r2 {r2:.2f})")
    plt.show()