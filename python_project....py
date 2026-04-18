

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 2026
@author: Kuruva Haritha
Project: Airline Passenger Satisfaction Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- 1. DASHBOARD THEME COLORS ---
BROWN = "#4B3621"
TAN = "#A68954"
SAND = "#C5B38A"
CREAM = "#F5F1E9"

# --- 2. DATA LOADING & CLEANING ---
path = r"C:\Users\harik\Downloads\air train.xlsx"
df = pd.read_excel(path)
df = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')
df.dropna(inplace=True)
print("Step 1: Dataset Loaded and Cleaned Successfully")

# --- 3. ENCODING ---
le = LabelEncoder()
encoded_df = df.copy()

for col in encoded_df.select_dtypes(include=['object']).columns:
    encoded_df[col] = encoded_df[col].astype(str)
    encoded_df[col] = le.fit_transform(encoded_df[col])

print("Step 2: Encoding successful! No more mixed types.")

# --- 4. MACHINE LEARNING MODEL ---
X = encoded_df.drop('satisfaction', axis=1)
y = encoded_df['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Generating predictions
y_pred = model.predict(X_test)
print("Step 3: Model Training Complete.")

# --- 5. VISUALIZATIONS ---
num_col = 'Flight Distance'

# A. HISTOGRAM
plt.figure(figsize=(8, 5), facecolor=CREAM)
plt.hist(df[num_col], bins=20, color=TAN, edgecolor=BROWN)
plt.title(f"Histogram of {num_col}", color=BROWN, fontweight='bold')
plt.show()

# B. BAR CHART (Top 10 Ages)
plt.figure(figsize=(8, 5), facecolor=CREAM)
df['Age'].value_counts().head(10).plot(kind='bar', color=TAN)
plt.title("Top 10 Passenger Ages", color=BROWN, fontweight='bold')
plt.show()

# C. PIE CHART (Solid Style)
plt.figure(figsize=(7, 7), facecolor=CREAM)
pie_data = df['Customer Type'].value_counts()
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=[TAN, SAND], startangle=140)
plt.title("Customer Type Distribution", color=BROWN, fontweight='bold')
plt.show()

# D. BOXPLOT
plt.figure(figsize=(8, 5), facecolor=CREAM)
sns.boxplot(x=df[num_col], color=SAND)
plt.title(f"Boxplot of {num_col}", color=BROWN, fontweight='bold')
plt.show()

# E. HEATMAP
plt.figure(figsize=(12, 8), facecolor=CREAM)
sns.heatmap(encoded_df.corr(), annot=False, cmap=sns.light_palette(TAN, as_cmap=True))
plt.title("Feature Correlation Matrix", color=BROWN, fontweight='bold')
plt.show()

# F. SCATTER PLOT
plt.figure(figsize=(8, 5), facecolor=CREAM)
plt.scatter(df.index[:500], df[num_col][:500], alpha=0.5, color=TAN)
plt.title("Scatter Plot (Sample 500)", color=BROWN, fontweight='bold')
plt.show()

# G. REGRESSION LINE
plt.figure(figsize=(8, 5), facecolor=CREAM)
sns.regplot(x='Departure Delay in Minutes', y='Arrival Delay in Minutes', 
            data=df.sample(500), scatter_kws={'color': TAN}, line_kws={'color': 'red'})
plt.title("Regression: Delay Analysis", color=BROWN, fontweight='bold')
plt.show()

# H. CATEGORY DISTRIBUTION
def categorize_val(val):
    if val < df[num_col].quantile(0.33): return "Low"
    elif val < df[num_col].quantile(0.66): return "Medium"
    else: return "High"

df['Category'] = df[num_col].apply(categorize_val)
cat_count = df['Category'].value_counts()
plt.figure(figsize=(8, 5), facecolor=CREAM)
plt.bar(cat_count.index, cat_count.values, color=[TAN, SAND, BROWN])
plt.title("Flight Distance Categories", color=BROWN, fontweight='bold')
plt.show()

# I. PAIR PLOT (Relationships)
print("Generating Pair Plot... please wait.")
pair_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'satisfaction']
sns.pairplot(df[pair_cols].sample(500), hue='satisfaction', palette=[TAN, BROWN])
plt.show()

# J. CONFUSION MATRIX 
plt.figure(figsize=(8, 6), facecolor=CREAM)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(TAN, as_cmap=True), 
            xticklabels=['Neutral/Dissat', 'Satisfied'], 
            yticklabels=['Neutral/Dissat', 'Satisfied'])
plt.title("Confusion Matrix", color=BROWN, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K. FEATURE IMPORTANCE
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6), facecolor=CREAM)
sns.barplot(x='Importance', y='Feature', data=feat_df, palette=sns.light_palette(TAN, reverse=True))
plt.title("Top 10 Satisfaction Factors", color=BROWN, fontweight='bold')
plt.show()

# --- FINAL OUTPUT ---
print("-" * 35)
print(f"FINAL PROJECT ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
print("-" * 35)