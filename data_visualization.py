import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/Users/melisacepni/PycharmProjects/miuul_project/term project/water_potability.csv")
water = pd.read_csv("/Users/melisacepni/PycharmProjects/miuul_project/water_quality.csv")

water.info()

# Grafik 1: Kayıp Veri Grafiği

missing_data_columns = data.columns[data.isnull().any()]

plt.figure(figsize=(8, 4))
msno.heatmap(data[missing_data_columns], cmap='Blues')
plt.show()

water.info()

# Grafik 2: Korelasyon Matrisi
plt.figure(figsize=(12, 8))
corr_matrix = water.corr()
sns.heatmap(corr_matrix, annot=True, cmap='Blues', square=True, linewidths=0.5, fmt=".3f", cbar_kws={"shrink": .8})
plt.show()

# Grafik 3: Histogram Grafikleri
sns.set_style("whitegrid")
columns = water.columns[:-1]
num_cols = 3
num_rows = len(columns) // num_cols + (len(columns) % num_cols > 0)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
for i, column in enumerate(columns):
    row = i // num_cols
    col = i % num_cols
    sns.histplot(water[column], kde=True, ax=axes[row, col], color='blue')
    axes[row, col].set_title(f'{column}', fontsize=12, fontweight='bold', color='darkblue')
    axes[row, col].set_xlabel("")
    axes[row, col].set_ylabel("")
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes.flatten()[j])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Grafik 4: Bar Grafikleri
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for idx, column in enumerate(water.columns[:-1]):
    ax = axes[idx // 4, idx % 4]
    sns.barplot(x='Potability', y=column, data=water, palette="Blues", ax=ax)
    ax.set_title(f'Average {column}', fontsize=12, color='darkblue')
    ax.set_xlabel('Potability')
    ax.set_ylabel(f'Average {column}')
for j in range(idx + 1, 12):
    fig.delaxes(axes[j // 4, j % 4])
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

# Grafik 5: İçilebilirlik Korelasyon Grafiği
plt.figure(figsize=(12, 8))
corr_matrix = water.corr()
sns.heatmap(corr_matrix[['Potability']], annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.show()

# Grafik 6: İçilebilirliğe göre değişkenlerin dağılım grafikleri
num_columns = water.select_dtypes(include=['float64', 'int64']).columns
fig, axes = plt.subplots(nrows=len(num_columns)//3 + 1, ncols=3, figsize=(15, 5*len(num_columns)//3))
axes = axes.flatten()
for i, column in enumerate(num_columns):
    sns.histplot(data=water, x=column, hue="Potability", kde=True, ax=axes[i], multiple="stack", color="skyblue")
    axes[i].set_title(f'{column} by Potability', color='darkblue')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# Grafik 7: İçilebilirlik Grafiği

water["Potability"].value_counts()

plt.figure(figsize=(8, 6))
sns.countplot(x='Potability', data=water, palette='Blues')
plt.xlabel('İçilebilirlik Durumu')
plt.ylabel('Frekans')
plt.show()