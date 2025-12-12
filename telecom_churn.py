import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("🔹 First 5 rows of the dataset:")
print(data.head())

print("\n🔹 Dataset Info:")
print(data.info())

print("\n🔹 Missing Values:")
print(data.isnull().sum())

print("\n🔹 Data type of TotalCharges before:", data['TotalCharges'].dtype)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

print("\n🔹 Missing Values after converting TotalCharges:")
print(data.isnull().sum())

data = data.dropna(subset=['TotalCharges'])

print("\n✅ Final Data Shape after cleaning:", data.shape)

print("\n🔹 Data Types after cleaning:")
print(data.dtypes)

print("\n🔹 Churn value counts:")
print(data['Churn'].value_counts())

# ---------- MATPLOTLIB GRAPHS INSTEAD OF SEABORN ----------

# Bar plot for Churn distribution
plt.figure(figsize=(5,4))
data['Churn'].value_counts().plot(kind='bar')
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Gender vs Churn
plt.figure(figsize=(6,4))
data.groupby(['gender','Churn']).size().unstack().plot(kind='bar', stacked=False)
plt.title('Churn by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Tenure vs Churn (Histogram)
plt.figure(figsize=(6,4))
for label in ['Yes', 'No']:
    plt.hist(data[data['Churn']==label]['tenure'], alpha=0.5, label=label)
plt.title('Tenure vs Churn')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Monthly Charges vs Churn (Boxplot)
plt.figure(figsize=(6,4))
data.boxplot(column='MonthlyCharges', by='Churn')
plt.title('Monthly Charges vs Churn')
plt.suptitle('')
plt.show()

# Contract Type vs Churn
plt.figure(figsize=(6,4))
data.groupby(['Contract','Churn']).size().unstack().plot(kind='bar')
plt.title('Contract Type vs Churn')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.show()

# Churn Rate
churn_rate = (data['Churn'].value_counts(normalize=True)['Yes']) * 100
print(f"\n Churn Rate: {churn_rate:.2f}%")
