import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("Ingesting dataset...")
df = pd.read_csv('car_data.csv')


df['Vehicle_Age'] = 2026 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True, errors='ignore')


X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_cols = ['Present_Price', 'Driven_kms', 'Owner', 'Vehicle_Age']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Gradient Boosting Pipeline...")
model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Variance Optimization Score (R2): {r2*100:.2f}%")


plt.figure(figsize=(8, 5))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6, 'color':'darkslateblue'}, line_kws={'color':'crimson'})
plt.title('Asset Valuation Evaluation: Actual vs. Predicted Target Values')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Model Price')
plt.tight_layout()


plt.savefig('evaluation_regression_plot.png')
print("Validation matrix diagram saved successfully as 'evaluation_regression_plot.png'!")