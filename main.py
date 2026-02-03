import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report


data = pd.read_csv('data.csv')
print(data.head(10))
print(data.info())

print(data['Loneliness'])

# Part 2
# Univariate Analysis
# Plot Lonliness histogram
plt.figure(figsize=(9, 6))
sns.histplot(data['Loneliness'], bins=5)
plt.title('Loneliness Distribution')
plt.xlabel('Loneliness')
plt.ylabel('Frequency')
plt.show()

# Plot Age distribution
plt.figure(figsize=(9, 6))
sns.histplot(data['Age'], bins=10)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot Finances distribution
plt.figure(figsize=(9, 6))
sns.histplot(data['Finances'], bins=5)
plt.title('Finances Distribution')
plt.ylabel('Frequency')
plt.xlabel('Finances')
plt.show()

# Plot Internet Usage
plt.figure(figsize=(9, 6))
sns.histplot(data['Internet usage'])
plt.title('Internet Usage')
plt.xlabel('Internet Usage')
plt.ylabel('Frequency')
plt.show()

# Plot Gender distribution
plt.figure(figsize=(9, 6))
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Plot Siblings distribution
plt.figure(figsize=(9, 6))
sns.histplot(data['Siblings'], bins=10)
plt.title('Siblings Distribution')
plt.xlabel('Number of Siblings')
plt.ylabel('Frequency')
plt.show()

# 2 Relationship to Target Variable
# Plot: Loneliness distribution by Gender
sns.set_style('whitegrid')
ct = pd.crosstab(data['Loneliness'], data["Gender"])
ax = ct.plot(kind='bar', figsize=(9, 6))
ax.set_title('Loneliness distribution by Gender')
ax.set_xlabel('Loneliness')
ax.set_ylabel('Count')
plt.show()

# Plot: Loneliness vs Age
plt.figure(figsize=(9, 6))
sns.boxplot(x='Loneliness', y='Age', data=data)
plt.title('Age Distribution by Loneliness Level')
plt.xlabel('Loneliness')
plt.ylabel('Age')
plt.show()

# Plot: Loneliness vs Finances
plt.figure(figsize=(9, 6))
sns.boxplot(x='Finances', y='Loneliness', data=data)
plt.title('Loneliness vs Finances')
plt.xlabel('Finances')
plt.ylabel('Loneliness')
plt.show()

# Plot: Loneliness vs Internet Usage
sns.set_style('whitegrid')
ct = pd.crosstab(data['Loneliness'], data["Internet usage"])
ax = ct.plot(kind='bar', figsize=(9, 6))
ax.set_title('Loneliness distribution by Internet Usage')
ax.set_xlabel('Loneliness')
ax.set_ylabel('Count')
plt.show()

# Plot: Loneliness vs Siblings
plt.figure(figsize=(9, 6))
sns.boxplot(x='Loneliness', y='Siblings', data=data)
plt.title('Siblings Distribution by Loneliness Level')
plt.xlabel('Loneliness')
plt.ylabel('Number of Siblings')
plt.show()

# 3 Correlation Analysis
numerical_cols = ['Loneliness', 'Age', 'Finances', 
                  'Siblings', "Parents' advice",
                  'Music', 'Movies', 'Techno',
                  'History', 'Pets', 'Mathematics', 'Spiders']
correlation_matrix = data[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.show()


# Preprocessing
data = data.drop(columns=['Techno', 'Siblings', 'Movies', 
                          'Finances', "Parents' advice", 'Mathematics'])

# Rows that contain any missing values
rows_with_missing = data[data.isnull().any(axis=1)]
print(rows_with_missing.head(100))

# Per-column missing counts
per_column_missing = data.isnull().sum()
per_column_table = pd.DataFrame({
    'missing_count': per_column_missing,
    'missing_percent': (per_column_missing / len(data) * 100).round(2)
})
print(per_column_table)

# Drop rows with missing Loneliness values
data = data.dropna(subset=['Loneliness'])

# Fill missing Age values with the column mean
n_age_missing = data['Age'].isnull().sum()
mean_age = data['Age'].mean().round()
data['Age'] = data['Age'].fillna(mean_age)

# Fill missing Gender values
n_missing = data['Gender'].isnull().sum()
np.random.seed(42)
missing_idx = data[data['Gender'].isnull()].index
choices = np.random.choice(['male', 'female'], size=n_missing)
data.loc[missing_idx, 'Gender'] = choices
filled_counts = data['Gender'].value_counts()

# One-hot encode Gender
gender_dummies = pd.get_dummies(data['Gender'], prefix='Gender')
data = pd.concat([data, gender_dummies], axis=1)
data = data.drop(columns=['Gender'])

# Ordinal-encode Internet usage into a new numeric column
internet_order = [
    'no time at all',
    'less than an hour a day',
    'few hours a day',
    'most of the day'
]
data['Internet_usage_ord'] = pd.Categorical(data['Internet usage'],
                                            categories=internet_order,
                                            ordered=True).codes

# Drop any remaining rows with missing values before modeling
data = data.dropna()

print("\nFinal dataset shape:", data.shape)
print("\nFinal columns:", data.columns.tolist())

# 4. Train/Test Split and Model Training
X = data.drop(columns=['Loneliness', 'Internet usage', 'Village - town'])
y = data['Loneliness']

print("\nFeatures used for modeling:", X.columns.tolist())
print(f"Training with {len(X)} samples")

# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n=== Model Performance ===")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Print feature coefficients
print("\n=== Feature Coefficients ===")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Classification Report (round predictions to nearest integer for classification)
y_test_pred_class = np.round(y_test_pred).astype(int)
y_train_pred_class = np.round(y_train_pred).astype(int)

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_test_pred_class))

print("\n=== Classification Report (Train Set) ===")
print(classification_report(y_train, y_train_pred_class))
