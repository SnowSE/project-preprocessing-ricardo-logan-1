import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('data.csv')
print(data.head(10))
print(data.info())

print(data['Loneliness'])

# Preprocessing
data = data.drop(columns=['Spiders'])

# Rows that contain any missing values
rows_with_missing = data[data.isnull().any(axis=1)]
print(rows_with_missing.head(100))

# Per-column missing counts and percentages
per_column_missing = data.isnull().sum()
per_column_table = pd.DataFrame({'missing_count': per_column_missing})
print(per_column_table)

# Fill missing Gender values
n_missing = data['Gender'].isnull().sum()
np.random.seed(42)
missing_idx = data[data['Gender'].isnull()].index
choices = np.random.choice(['male', 'female'], size=n_missing)
data.loc[missing_idx, 'Gender'] = choices
filled_counts = data['Gender'].value_counts()


# After Preprocessing
# Rows that contain any missing values
rows_with_missing = data[data.isnull().any(axis=1)]
print(rows_with_missing.head(100))

# Per-column missing counts and percentages
per_column_missing = data.isnull().sum()
per_column_table = pd.DataFrame({'missing_count': per_column_missing})
print(per_column_table)

# Plot: Loneliness distribution by Gender
sns.set_style('whitegrid')
ct = pd.crosstab(data['Loneliness'], data["Gender"])

ax = ct.plot(kind='bar', figsize=(9, 6))
ax.set_title('Loneliness distribution by Gender')
ax.set_xlabel('Loneliness')
ax.set_ylabel('Count')
plt.show()


