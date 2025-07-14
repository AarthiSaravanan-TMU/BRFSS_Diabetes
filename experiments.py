# -*- coding: utf-8 -*-

# google mount
from google.colab import drive
drive.mount('/content/drive')

# Installation
!pip install catboost

# import the library
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# Load the datasets
diabetes_012 = pd.read_csv("/content/drive/MyDrive/MRP/Dataset/diabetes_012_health_indicators_BRFSS2015.csv")
diabetes_012.head()

diabetes_012.isna().sum()

# Generate descriptive summary statistics for all datasets
summary_012 = diabetes_012.describe(include="all")
summary_012

df_012 = diabetes_012.copy()
df_012

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

# Define the BMI categorization function based on the provided image
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Healthy Weight"
    elif bmi < 30.0:
        return "Overweight"
    elif bmi < 35.0:
        return "Class 1 Obesity"
    elif bmi < 40.0:
        return "Class 2 Obesity"
    else:
        return "Class 3 Obesity"

# Apply categorization to BMI
df_012["BMI"] = df_012["BMI"].apply(categorize_bmi)

df_012.head()

df_012['BMI'].unique()

# OrdinalEncoder
# Define custom order for BMI categories
bmi_order = ["Underweight",
    "Healthy Weight",
    "Overweight",
    "Class 1 Obesity",
    "Class 2 Obesity",
    "Class 3 Obesity"]

# Apply OrdinalEncoder and shift encoding by +1
ordinal_encoder = OrdinalEncoder(categories=[bmi_order])
df_012[["BMI"]] = ordinal_encoder.fit_transform(df_012[["BMI"]]) + 1

# Show the mapping
df_012.head(5)

# Save the DataFrame to a CSV file
#df.to_csv("/content/drive/MyDrive/MRP/Dataset/diabetes_012_health_indicators_BRFSS2015_categorical.csv", index=False, encoding="utf-8")
#print("DataFrame saved to 'diabetes_012_health_indicators_BRFSS2015_categorical.csv'")

df = pd.read_csv("/content/drive/MyDrive/MRP/Dataset/diabetes_012_health_indicators_BRFSS2015_categorical.csv")
df.head()

"""1. Read the csv file
2. Applied BMI categories and then perform ordinal encoding
"""

df.info()

df = df.astype(int)
df.info()

df.head()

# Step 1: Split into features and target
X = df.drop(columns='Diabetes_012')
y = df['Diabetes_012']

# Step 2: Split off Test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Step 3: Split remaining 80% into Train (62.5% of 80%) and Val1+Val2 (37.5% of 80%)
X_train, X_val_temp, y_train, y_val_temp = train_test_split(
    X_temp, y_temp, test_size=0.375, random_state=42, stratify=y_temp
)

# Step 4: Split Val1 and Val2 equally (each 18.75% of total)
X_val1, X_val2, y_val1, y_val2 = train_test_split(
    X_val_temp, y_val_temp, test_size=0.5, random_state=42, stratify=y_val_temp
)

# Check shapes
print("Train:", X_train.shape)
print("Val1 :", X_val1.shape)
print("Val2 :", X_val2.shape)
print("Test :", X_test.shape)

"""# Without SMOTE - with Imbalanced nature"""

# Step 3: Feature Selection on SMOTE-applied training data

# Chi²
chi2_selector = SelectKBest(score_func=chi2, k='all').fit(X_train, y_train)
chi2_scores = pd.Series(chi2_selector.scores_, index=X_train.columns)


# Mutual Information
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all').fit(X_train, y_train)
mi_scores = pd.Series(mi_selector.scores_, index=X_train.columns)


# Information Gain (Entropy-Based)
ig_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
ig_model.fit(X_train, y_train)
ig_scores = pd.Series(ig_model.feature_importances_, index=X_train.columns)


# Normalize scores
scaler = MinMaxScaler()
chi2_scaled = pd.Series(scaler.fit_transform(chi2_scores.values.reshape(-1, 1)).flatten(), index=X_train.columns)
mi_scaled = pd.Series(scaler.fit_transform(mi_scores.values.reshape(-1, 1)).flatten(), index=X_train.columns)
ig_scaled = pd.Series(scaler.fit_transform(ig_scores.values.reshape(-1, 1)).flatten(), index=X_train.columns)

# Combine scores (equal weight)
combined_score = chi2_scaled + mi_scaled + ig_scaled
top_features = combined_score.sort_values(ascending=False).head(20)

# Create summary DataFrame
top_features_df = pd.DataFrame({
    "Feature": top_features.index,
    "CombinedScore": top_features.values,
    "Chi2_Score": chi2_scores[top_features.index].values,
    "MI_Score": mi_scores[top_features.index].values,
    "IG_Score": ig_scores[top_features.index].values
})

# Create rank-based DataFrame without AvgRank
ranked_features_df = pd.DataFrame({
    "Feature": chi2_scores.index,
    "Chi2_Rank": chi2_scores.rank(ascending=False).astype(int),
    "MI_Rank": mi_scores.rank(ascending=False).astype(int),
    "IG_Rank": ig_scores.rank(ascending=False).astype(int)
})

# Total number of features for Borda Count
num_features = len(ranked_features_df)

# Convert ranks to Borda scores (higher is better)
borda_scores = pd.DataFrame({
    "Feature": ranked_features_df["Feature"],
    "Chi2_Score": num_features - ranked_features_df["Chi2_Rank"] + 1,
    "MI_Score": num_features - ranked_features_df["MI_Rank"] + 1,
    "IG_Score": num_features - ranked_features_df["IG_Rank"] + 1
})
borda_scores["Borda_Total"] = borda_scores[["Chi2_Score", "MI_Score", "IG_Score"]].sum(axis=1)

# Sort Borda Count results
imbalance_BordaCount_df = borda_scores.sort_values("Borda_Total", ascending=False).reset_index(drop=True)
imbalance_BordaCount_df

# Plotting the Borda Count results
plt.figure(figsize=(12, 8))
plt.barh(imbalance_BordaCount_df['Feature'], imbalance_BordaCount_df['Borda_Total'])
plt.title("Feature Importance on imbalance dataset")
plt.xlabel("Borda Score)")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Highest score at the top
plt.tight_layout()
plt.show()

# Create Rank-Score plot data
chi2_ranked = chi2_scaled.sort_values(ascending=False)
mi_ranked = mi_scaled.sort_values(ascending=False)
ig_ranked = ig_scaled.sort_values(ascending=False)

# Prepare data for plotting
plot_data = [
    (range(1, len(chi2_ranked)+1), chi2_ranked.values, 'Chi²'),
    (range(1, len(mi_ranked)+1), mi_ranked.values, 'Mutual Information'),
    (range(1, len(ig_ranked)+1), ig_ranked.values, 'Information Gain')
]

# Plot
plt.figure(figsize=(12, 7))
markers = ['o', 's', '^']
for (ranks, scores, label), marker in zip(plot_data, markers):
    plt.plot(ranks, scores, label=label, marker=marker)

plt.xlabel("Feature Rank")
plt.ylabel("Normalized Score")
plt.title("Rank-Score Graphs on imbalanced dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define the new feature sets for Top-K
feature_sets = {}
for k in range(2, 17, 2):  # 2, 4, 6, ..., 16
    key = f'Top{k}'
    feature_sets[key] = top_features_df['Feature'].head(k).tolist()

# Define models
models = {
    'LogisticRegression': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(C=1, max_iter=1000))
    ]),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    'CatBoost': CatBoostClassifier(depth=4, verbose=0, random_state=42),
    'MLP': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42))
    ])
}

# Store results
results = []

# Evaluate each model on each feature set
for name, features in feature_sets.items():
    X_train_k = X_train[features]
    X_val_k = X_val1[features]

    for model_name, model in models.items():
        model.fit(X_train_k, y_train)
        preds = model.predict(X_val_k)

        # Calculate metrics
        #acc = accuracy_score(y_val1, preds)
        #prec = precision_score(y_val1, preds, average='weighted', zero_division=0)
        #rec = recall_score(y_val1, preds, average='weighted', zero_division=0)
        #f1 = f1_score(y_val1, preds, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_val1, preds, average='macro', zero_division=0)
        recall = recall_score(y_val1, preds, average='macro', zero_division=0)


        # If possible, get prediction probabilities for AUC
        try:
            probs = model.predict_proba(X_val_k)
            auc = roc_auc_score(y_val1, probs, multi_class='ovr', average='weighted')
        except:
            auc = np.nan

        '''results.append({
            'Model': model_name,
            'Feature_Set': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUC': auc
        })'''
        macro_f1 = f1_score(y_val1, preds, average='macro', zero_division=0)
        recall = recall_score(y_val1, preds, average='macro', zero_division=0)
        results.append({
            'Model': model_name,
            'Feature_Set': name,
            'Recall': recall,
            'Macro_F1': macro_f1,
            'AUC': auc
        })


# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df

# Define color palette
palette = sns.color_palette("tab10")

# Plot
plt.figure(figsize=(12, 6))
for i, model in enumerate(results_df['Model'].unique()):
    subset = results_df[results_df['Model'] == model]
    feature_counts = subset['Feature_Set'].str.extract('(\d+)').astype(int)
    plt.plot(
        feature_counts[0],
        subset['Macro_F1'],
        marker='o',
        label=model,
        color=palette[i]
    )

plt.title("Top-k feature with Macro F1 Score")
plt.xlabel("Top-k Features")
plt.ylabel("Macro F1 Score")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.show()

# Set color palette
sns.set_palette("tab10")

# Prepare AUC plot
plt.figure(figsize=(12, 6))

# Convert feature set names like 'Top2', 'Top4' to numeric values for x-axis
results_df['Num_Features'] = results_df['Feature_Set'].str.extract('(\d+)').astype(int)

# Plot AUC for each model
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    plt.plot(model_data['Num_Features'], model_data['AUC'], marker='o', label=model)

plt.title("Top-k feature with AUC Score")
plt.xlabel("Top-k features")
plt.ylabel("AUC Score")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.show()

"""# With SMOTE"""

# Step 2: Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Step 2 A: Check class balance after resampling
print("Before SMOTE-ENN:")
print(y_train.value_counts(normalize=True))
print("\nAfter SMOTE-ENN:")
print(pd.Series(y_train_sm).value_counts(normalize=True))

# Step 3: Feature Selection on SMOTE-applied training data

# Chi²
chi2_selector = SelectKBest(score_func=chi2, k='all').fit(X_train_sm, y_train_sm)
chi2_scores = pd.Series(chi2_selector.scores_, index=X_train.columns)


# Mutual Information
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all').fit(X_train_sm, y_train_sm)
mi_scores = pd.Series(mi_selector.scores_, index=X_train.columns)


# Information Gain (Entropy-Based)
ig_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
ig_model.fit(X_train_sm, y_train_sm)
ig_scores = pd.Series(ig_model.feature_importances_, index=X_train.columns)


# Normalize scores
scaler = MinMaxScaler()
chi2_scaled = pd.Series(scaler.fit_transform(chi2_scores.values.reshape(-1, 1)).flatten(), index=X_train.columns)
mi_scaled = pd.Series(scaler.fit_transform(mi_scores.values.reshape(-1, 1)).flatten(), index=X_train.columns)
ig_scaled = pd.Series(scaler.fit_transform(ig_scores.values.reshape(-1, 1)).flatten(), index=X_train.columns)

# Combine scores (equal weight)
combined_score = chi2_scaled + mi_scaled + ig_scaled
top_features = combined_score.sort_values(ascending=False).head(20)

# Create summary DataFrame
top_features_df = pd.DataFrame({
    "Feature": top_features.index,
    "CombinedScore": top_features.values,
    "Chi2_Score": chi2_scores[top_features.index].values,
    "MI_Score": mi_scores[top_features.index].values,
    "IG_Score": ig_scores[top_features.index].values
})

# Create rank-based DataFrame without AvgRank
ranked_features_df = pd.DataFrame({
    "Feature": chi2_scores.index,
    "Chi2_Rank": chi2_scores.rank(ascending=False).astype(int),
    "MI_Rank": mi_scores.rank(ascending=False).astype(int),
    "IG_Rank": ig_scores.rank(ascending=False).astype(int)
})

# Total number of features for Borda Count
num_features = len(ranked_features_df)

# Convert ranks to Borda scores (higher is better)
borda_scores = pd.DataFrame({
    "Feature": ranked_features_df["Feature"],
    "Chi2_Score": num_features - ranked_features_df["Chi2_Rank"] + 1,
    "MI_Score": num_features - ranked_features_df["MI_Rank"] + 1,
    "IG_Score": num_features - ranked_features_df["IG_Rank"] + 1
})
borda_scores["Borda_Total"] = borda_scores[["Chi2_Score", "MI_Score", "IG_Score"]].sum(axis=1)

# Sort Borda Count results
balanced_BordaCount_df = borda_scores.sort_values("Borda_Total", ascending=False).reset_index(drop=True)
balanced_BordaCount_df

# Plotting the Borda Count results
plt.figure(figsize=(12, 8))
plt.barh(balanced_BordaCount_df['Feature'], balanced_BordaCount_df['Borda_Total'])
plt.title("Feature Importance on balanced dataset")
plt.xlabel("Borda Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Highest score at the top
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

# Create Rank-Score plot data
chi2_ranked = chi2_scaled.sort_values(ascending=False)
mi_ranked = mi_scaled.sort_values(ascending=False)
ig_ranked = ig_scaled.sort_values(ascending=False)

# Prepare data for plotting
plot_data = [
    (range(1, len(chi2_ranked)+1), chi2_ranked.values, 'Chi²'),
    (range(1, len(mi_ranked)+1), mi_ranked.values, 'Mutual Information'),
    (range(1, len(ig_ranked)+1), ig_ranked.values, 'Information Gain')
]

# Plot
plt.figure(figsize=(12, 7))
markers = ['o', 's', '^']
for (ranks, scores, label), marker in zip(plot_data, markers):
    plt.plot(ranks, scores, label=label, marker=marker)

plt.xlabel("Feature Rank")
plt.ylabel("Normalized Score")
plt.title("Rank-Score Graphs on balanced")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#!pip install catboost

# Use pre-loaded top_features_df, X_train_sm, X_val, y_train_sm, y_val

# Define the new feature sets for Top-K
feature_sets = {}
for k in range(2, 17, 2):  # 2, 4, 6, ..., 16
    key = f'Top{k}'
    feature_sets[key] = top_features_df['Feature'].head(k).tolist()

# Define models
models = {
    'LogisticRegression': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(C=1, max_iter=1000))
    ]),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    'CatBoost': CatBoostClassifier(depth=4, verbose=0, random_state=42),
    'MLP': Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42))
    ])
}

# Store results
results = []

# Evaluate each model on each feature set
for name, features in feature_sets.items():
    X_train_k = X_train_sm[features]
    X_val_k = X_val1[features]

    for model_name, model in models.items():
        model.fit(X_train_k, y_train_sm)
        preds = model.predict(X_val_k)

        # Calculate metrics
        # acc = accuracy_score(y_val1, preds)
        # prec = precision_score(y_val1, preds, average='weighted', zero_division=0)
        # rec = recall_score(y_val1, preds, average='weighted', zero_division=0)
        # f1 = f1_score(y_val1, preds, average='weighted', zero_division=0)

        # If possible, get prediction probabilities for AUC
        try:
            probs = model.predict_proba(X_val_k)
            auc = roc_auc_score(y_val1, probs, multi_class='ovr', average='weighted')
        except:
            auc = np.nan

        '''results.append({
            'Model': model_name,
            'Feature_Set': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUC': auc
        })'''
        macro_f1 = f1_score(y_val1, preds, average='macro', zero_division=0)
        recall = recall_score(y_val1, preds, average='macro', zero_division=0)
        results.append({
            'Model': model_name,
            'Feature_Set': name,
            'Recall': recall,
            'Macro_F1': macro_f1,
            'AUC': auc
        })


# Convert to DataFrame
balanced_results_df = pd.DataFrame(results)
balanced_results_df

# Define color palette
palette = sns.color_palette("tab10")

# Plot
plt.figure(figsize=(12, 6))
for i, model in enumerate(balanced_results_df['Model'].unique()):
    subset = results_df[balanced_results_df['Model'] == model]
    feature_counts = subset['Feature_Set'].str.extract('(\d+)').astype(int)
    plt.plot(
        feature_counts[0],
        subset['Macro_F1'],
        marker='o',
        label=model,
        color=palette[i]
    )

plt.title("Top-k feature with Macro F1 Score")
plt.xlabel("Top-k Features")
plt.ylabel("Macro F1 Score")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.show()

# Set color palette
sns.set_palette("tab10")

# Prepare AUC plot
plt.figure(figsize=(12, 6))

# Convert feature set names like 'Top2', 'Top4' to numeric values for x-axis
balanced_results_df['Num_Features'] = balanced_results_df['Feature_Set'].str.extract('(\d+)').astype(int)

# Plot AUC for each model
for model in balanced_results_df['Model'].unique():
    model_data = balanced_results_df[balanced_results_df['Model'] == model]
    plt.plot(model_data['Num_Features'], model_data['AUC'], marker='o', label=model)

plt.title("Top-k feature with AUC Score")
plt.xlabel("Top-k features")
plt.ylabel("AUC Score")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.show()

"""Reasons for Choosing Top 10 Features
Peak Performance:

Logistic Regression achieves its highest AUC at Top 10 features.

Plateau Point for Others:

CatBoost, LightGBM, and MLP show performance plateau or small improvements beyond Top 10, indicating diminishing returns.

Performance Decreases After Top 10:

After Top 10, AUC drops or flattens for most models, especially Logistic Regression and MLP.

Avoiding Overfitting:

Adding more features beyond Top 10 doesn’t improve performance but increases model complexity and risk of overfitting.

Efficiency:

Top 10 gives a good balance between high AUC and low feature count, making the model faster and easier to interpret.

Consistency Across Models:

Most models achieve optimal or near-optimal AUC with Top 10, making it a robust cutoff across classifiers.
"""

# Highlight top 10 features
colors = ['tab:blue' if i >= 10 else 'tab:orange' for i in range(len(balanced_BordaCount_df))]

plt.figure(figsize=(12, 8))
plt.barh(balanced_BordaCount_df['Feature'], balanced_BordaCount_df['Borda_Total'], color=colors)
plt.title("Feature Importance")
plt.xlabel("Borda Count (Combined Rank)")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Highest score at the top
plt.tight_layout()
plt.show()

"""# Balanced and imbalanced features"""

# Highlight top 10 features
colors = ['tab:blue' if i >= 10 else 'tab:orange' for i in range(len(balanced_BordaCount_df))]

plt.figure(figsize=(12, 8))
plt.barh(balanced_BordaCount_df['Feature'], balanced_BordaCount_df['Borda_Total'], color=colors)
plt.title("Feature Importance on balanced dataset")
plt.xlabel("Borda Count")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Highest score at the top
plt.tight_layout()
plt.show()

# Highlight top 10 features
colors = ['tab:blue' if i >= 10 else 'tab:orange' for i in range(len(imbalance_BordaCount_df))]

plt.figure(figsize=(12, 8))
plt.barh(imbalance_BordaCount_df['Feature'], imbalance_BordaCount_df['Borda_Total'], color=colors)
plt.title("Feature Importance on imbalanced dataset")
plt.xlabel("Borda Count")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Highest score at the top
plt.tight_layout()
plt.show()

!pip install optuna

import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Assuming X_train_final, y_train_final, X_val2, y_val2, and feature_sets["Top10"] are defined
# Define top10
top10 = feature_sets['Top10']

# Combine training data (X_train + X_val1) if not already done
X_train_final = pd.concat([X_train, X_val1])
y_train_final = pd.concat([y_train, y_val1])

# Dictionary to store best Optuna trials
best_trials = {}

# Objective function for Optuna tuning
def objective(trial, model_name):
    if model_name == 'LogisticRegression':
        C = trial.suggest_loguniform('C', 1e-3, 1e2)
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(C=C, max_iter=1000))
        ])

    elif model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    elif model_name == 'LightGBM':
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        max_depth = trial.suggest_int('max_depth', 3, 12)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        model = LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth,
                               learning_rate=learning_rate, n_estimators=100,
                               random_state=42)

    elif model_name == 'CatBoost':
        depth = trial.suggest_int('depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        model = CatBoostClassifier(depth=depth, learning_rate=learning_rate,
                                   n_estimators=100, verbose=0, random_state=42, task_type='CPU')

    elif model_name == 'MLP':
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(64,), (128,), (64, 64)])
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                                  max_iter=300, random_state=42))
        ])

    model.fit(X_train_final[top10], y_train_final)
    preds = model.predict_proba(X_val2[top10])
    auc = roc_auc_score(y_val2, preds, multi_class='ovr', average='weighted')
    return auc

# Run optimization for all models
for model_name in ['LogisticRegression', 'RandomForest', 'LightGBM', 'CatBoost', 'MLP']:
    print(f"Optimizing {model_name}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=30, show_progress_bar=True)
    best_trials[model_name] = study.best_trial
    print(f"Best AUC for {model_name}: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

# Display the best hyperparameters and AUC scores
for model_name, trial in best_trials.items():
    print(f"Model: {model_name}")
    print(f"  Best AUC Score : {trial.value:.4f}")
    print(f"  Best Parameters: {trial.params}")
    print("-" * 50)
