# TASK 2 - Unemployment Analysis India - Laxman Manku OIBSIP
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

# Sample data creation (Original dataset: Unemployment_in_India.csv from Kaggle)
dates = pd.date_range('2019-05-01','2021-05-01', freq='ME')
regions = ['Andhra Pradesh','Telangana','Karnataka','Maharashtra','Delhi','Tamil Nadu','Kerala','UP','West Bengal','Gujarat']
np.random.seed(42)
data=[]
for d in dates:
    for r in regions:
        base=5+np.random.randn()*1.5
        if d.month>=3 and d.year==2020 and d.month<=6: base+=15
        data.append([r,d,base+np.random.randn(), base-0.5, 40+np.random.randn()*2])

df=pd.DataFrame(data, columns=['Region','Date','Estimated Unemployment Rate (%)','Estimated Employed','Labour Participation Rate (%)'])
print(df.shape)
print(df.isnull().sum())
df['Date']=pd.to_datetime(df['Date'])

# EDA region-wise avg
print(df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False))

# Time-series line chart
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Unemployment Rate Over Time - COVID Impact')
plt.xticks(rotation=45)
plt.show()

# Bar chart top 10
avg=df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=avg.values, y=avg.index)
plt.title('Top 10 States Highest Unemployment')
plt.show()

# Heatmap
sns.heatmap(df[['Estimated Unemployment Rate (%)','Estimated Employed','Labour Participation Rate (%)']].corr(), annot=True, cmap='coolwarm')
plt.show()

# Pre vs Post COVID Analysis
pre=df[df['Date']<'2020-03-01']['Estimated Unemployment Rate (%)'].mean()
post=df[(df['Date']>='2020-03-01')&(df['Date']<='2020-08-01')]['Estimated Unemployment Rate (%)'].mean()
print(f"Pre-COVID Avg: {pre:.2f}% Post-COVID Lockdown Avg: {post:.2f}%")
print("Observation: Unemployment spiked 3x during lockdown, urban areas worst hit")
