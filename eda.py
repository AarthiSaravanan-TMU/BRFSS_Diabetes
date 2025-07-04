# -*- coding: utf-8 -*-

# google mount
from google.colab import drive
drive.mount('/content/drive')

# import the library
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, chi2_contingency

# Load the datasets
df_012 = pd.read_csv("/content/drive/MyDrive/MRP/Dataset/diabetes_012_health_indicators_BRFSS2015.csv")

# Function to summarize missing values
def check_missing_values(df, name):
    """ Check for missing values count """
    print(f"Missing values in '{name}':")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("  No missing values.\n")
    else:
        print(missing[missing > 0], "\n")

check_missing_values(df_012, "diabetes_012_health_indicators_BRFSS2015.csv")

# Descriptive summary statistics=
summary_012 = df_012.describe(include="all")
summary_012.T

# Get value counts and calculate percentages
value_counts = df_012['Diabetes_012'].value_counts()
percentages = 100 * value_counts / value_counts.sum()

# Create a new DataFrame for plotting
plot_df = pd.DataFrame({
    'Category': value_counts.index,
    'Count': value_counts.values,
    'Percentage': percentages.values
}).sort_values(by='Count', ascending=False)

# Plot with correct order
plt.figure(figsize=(8, 6))
sns.barplot(
    x='Category',
    y='Count',
    data=plot_df,
    order=plot_df['Category'],
    palette='dark'
)

# Add percentage labels above the bars
for i, row in enumerate(plot_df.itertuples()):
    plt.text(i, row.Count, f"{row.Percentage:.1f}%", ha='center', va='bottom', fontsize=10)

plt.title('Target Count')
plt.xlabel('Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# correlation matrix
corr = df_012.corr('spearman')

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(14, 12))  # Make the figure bigger

# Draw the heatmap with larger annotations and 2 decimal formatting
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f",
            square=True, cbar_kws={"shrink": .9}, annot_kws={"size": 12})

plt.title("Heatmap of Feature Correlations", fontsize=16)
plt.tight_layout()
plt.show()

# Hypothesis testing
# Top 5 features are GenHlth, HighBP, BMI, DiffWalk, HighChol

# Split data by Diabetes_012 class
group0 = df_012[df_012["Diabetes_012"] == 0]
group1 = df_012[df_012["Diabetes_012"] == 1]
group2 = df_012[df_012["Diabetes_012"] == 2]

# ANOVA for continuous/ordinal features: GenHlth, BMI
anova_genhlth = f_oneway(group0["GenHlth"], group1["GenHlth"], group2["GenHlth"])
anova_bmi = f_oneway(group0["BMI"], group1["BMI"], group2["BMI"])

# Chi-square test for categorical features: HighBP, DiffWalk, HighChol
def chi_square(var):
    contingency = pd.crosstab(df_012[var], df_012["Diabetes_012"])
    return chi2_contingency(contingency)

chi_highbp = chi_square("HighBP")
chi_diffwalk = chi_square("DiffWalk")
chi_highchol = chi_square("HighChol")

# Summarize results
hypothesis_results = pd.DataFrame({
    "Feature": ["GenHlth", "BMI", "HighBP", "DiffWalk", "HighChol"],
    "Test": ["ANOVA", "ANOVA", "Chi-square", "Chi-square", "Chi-square"],
    "Statistic": [anova_genhlth.statistic, anova_bmi.statistic, chi_highbp[0], chi_diffwalk[0], chi_highchol[0]],
    "p-value": [anova_genhlth.pvalue, anova_bmi.pvalue, chi_highbp[1], chi_diffwalk[1], chi_highchol[1]]
})

hypothesis_results

# Outlier detection for continuous and quasi-continuous features
ordinalValues = [col for col in df_012.columns if df_012[col].nunique() >  15]
# Boxplot for continuous/quasi- continuous features
plt.figure(figsize=(8, 8))
box = df_012[ordinalValues].boxplot(rot=90, patch_artist=True)
for patch in box.artists:
    patch.set_facecolor('lightblue')
for median in box.lines[4::6]:
    median.set_color('red')
    median.set_linewidth(2)
plt.title("Outlier detection", fontsize=16)
plt.ylabel("Value Range")
plt.tight_layout()
plt.show()

