import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
csv_file = '/Users/abhinavdyansamantara/Desktop/Abhinav Dyan Samantara/FILES/PhishingLinkDetection/feature_extracted_dataset.csv'  # Replace this with the path to your CSV file
df = pd.read_csv(csv_file)

# Descriptive Analytics
print("First few rows of the dataset:\n", df.head())
print(f"Number of rows: {len(df)}")
print("Column data types:\n", df.dtypes)

# Check for missing values if present
print("\nMissing values per column:\n", df.isnull().sum())

# Distribution of class
print("\nClass Distribution:\n", df['label'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False)
plt.title("Class Distribution (0 = Legitimate, 1 = Phishing)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Feature summary statistics
print("\nSummary statistics for numeric columns:\n", df.describe())


for feature in ['url_length', 'num_subdomains', 'num_special_chars']:
    # Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(df[feature], kde=True, bins=30, color='blue')
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    
    # Box Plot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='label', y=feature, palette='coolwarm')
    plt.title(f"Box Plot of {feature} by Label")
    plt.xlabel("Label")
    plt.ylabel(feature)
    plt.show()

# Data Preprocessing
X = df[['url_length', 'num_subdomains', 'num_special_chars', 'is_https', 'has_phishing_keyword']]
y = df['label']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Initialization and Cross-Validation
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')

print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Output the best parameters
print("\nBest Parameters from Grid Search:", grid_search.best_params_)

# Evaluate with the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

# Classification Report
print("\nClassification Report with Best Model:\n", classification_report(y_test, y_pred_best))

# Step 6: Save the Best Model
joblib.dump(best_model, 'phishing_model_best.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully!")

# Feature Importance Plot
feature_importances = best_model.feature_importances_
features = X.columns
plt.barh(features, feature_importances, color='green')
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
print(f"ROC-AUC Score: {roc_auc}")

fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Correlation Matrix and Heatmap

# Drop non-numeric columns (like 'url') if they exist
numeric_df = df.select_dtypes(include=['number'])

# Calculate and display the correlation matrix
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Step 10: Data Class Distribution
print("\nFinal Class Distribution:\n", df['label'].value_counts())