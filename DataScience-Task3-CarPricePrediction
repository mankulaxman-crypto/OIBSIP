# TASK 3 - Car Price Prediction with ML - Laxman Manku OIBSIP
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Synthetic Car Price Dataset (based on CarDekho / Car Price Kaggle structure)
np.random.seed(42)
n=1500
brands=['Maruti','Hyundai','Honda','Toyota','Ford','Mahindra','Tata','Kia']
fuel=['Petrol','Diesel','CNG','Electric']
owner=['First','Second','Third']
trans=['Manual','Automatic']

data={
    'Brand':np.random.choice(brands,n),
    'Year':np.random.randint(2005,2024,n),
    'Present_Price':np.round(np.random.uniform(2,40,n),2),
    'Kms_Driven':np.random.randint(5000,200000,n),
    'Fuel_Type':np.random.choice(fuel,n),
    'Seller_Type':np.random.choice(['Dealer','Individual'],n),
    'Transmission':np.random.choice(trans,n),
    'Owner':np.random.choice(owner,n)
}
df=pd.DataFrame(data)
df['Age']=2024-df['Year']
# Price formula + noise
df['Selling_Price']=df['Present_Price']*0.85 - df['Age']*0.2 - df['Kms_Driven']/100000 + np.random.randn(n)*0.5
df['Selling_Price']=df['Selling_Price'].clip(1,35)

print(df.head())
print(df.isnull().sum())
print(df.describe())

# EDA
sns.countplot(x='Fuel_Type', data=df)
plt.title('Fuel Type Count'); plt.show()
sns.scatterplot(x='Age', y='Selling_Price', data=df)
plt.title('Age vs Price'); plt.show()

# Encoding
df_enc=pd.get_dummies(df, columns=['Brand','Fuel_Type','Seller_Type','Transmission','Owner'], drop_first=True)

X=df_enc.drop(['Selling_Price','Year'],axis=1)
y=df_enc['Selling_Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Models
for name,model in [("Linear Regression",LinearRegression()),("Random Forest",RandomForestRegressor(n_estimators=100,random_state=42))]:
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    print(f"{name} R2: {r2_score(y_test,pred):.4f} RMSE: {np.sqrt(mean_squared_error(y_test,pred)):.4f} MAE: {mean_absolute_error(y_test,pred):.4f}")

# Feature Importance
rf=RandomForestRegressor().fit(X_train,y_train)
imp=pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
sns.barplot(x=imp.values, y=imp.index)
plt.title('Top 10 Important Features for Car Price'); plt.show()
print("Conclusion: Present_Price and Age are top factors")
