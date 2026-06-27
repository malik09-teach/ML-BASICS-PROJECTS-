import os
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

def main():
    # --- 1. Project Setup and Data Acquisition ---
    print("--- 1. Setting up and downloading data ---")
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    csv_file = os.path.join(data_dir, 'Stars.csv')

    if not os.path.exists(csv_file):
        print("Downloading Stars.csv...")
        url = "https://github.com/YBIFoundation/Dataset/raw/main/Stars.csv"
        urllib.request.urlretrieve(url, csv_file)
        print("Download complete.")
    else:
        print("Stars.csv already exists.")

    # --- 2. Exploratory Data Analysis (EDA) ---
    print("\n--- 2. Exploratory Data Analysis ---")
    df = pd.read_csv(csv_file)
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nGenerating EDA plots...")
    os.makedirs('plots', exist_ok=True)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Select only numerical features for correlation matrix
    numerical_df = df.select_dtypes(include=['number'])
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

    # Class Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Star type', palette='viridis')
    plt.title('Distribution of Star Types')
    plt.savefig('plots/class_distribution.png')
    plt.close()

    print("Plots saved in 'plots' directory.")

    # --- 3. Data Preprocessing ---
    print("\n--- 3. Data Preprocessing ---")
    # Features and Target
    X = df.drop(['Star type', 'Star category'], axis=1, errors='ignore')
    y = df['Star type']

    # Identify categorical and numerical columns
    categorical_cols = ['Star color', 'Spectral Class']
    numerical_cols = ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)']

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # --- 4. Model Training & Evaluation ---
    print("\n--- 4. Model Training & Evaluation ---")

    # Model 1: Random Forest
    print("\nTraining Random Forest...")
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])
    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)

    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))

    # Save RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, rf_predictions)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    disp_rf.plot(cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig('plots/rf_confusion_matrix.png')
    plt.close()

    # Model 2: Support Vector Machine
    print("\nTraining Support Vector Machine (SVM)...")
    svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', SVC(random_state=42))])
    svm_pipeline.fit(X_train, y_train)
    svm_predictions = svm_pipeline.predict(X_test)

    print("SVM Classification Report:")
    print(classification_report(y_test, svm_predictions))

    # Save SVM Confusion Matrix
    cm_svm = confusion_matrix(y_test, svm_predictions)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
    disp_svm.plot(cmap='Blues')
    plt.title('SVM Confusion Matrix')
    plt.savefig('plots/svm_confusion_matrix.png')
    plt.close()

    # --- 5. Save the Model ---
    print("\n--- 5. Saving Model ---")
    model_filename = 'rf_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(rf_pipeline, f)
    print(f"Random Forest model saved to {model_filename}")

    # Also save SVM model since app.py is looking for svm_model.pkl
    svm_model_filename = 'svm_model.pkl'
    with open(svm_model_filename, 'wb') as f:
        pickle.dump(svm_pipeline, f)
    print(f"SVM model saved to {svm_model_filename}")

    print("\n--- Project Execution Complete ---")
    print("Check the 'plots' folder for visualizations.")

if __name__ == "__main__":
    main()
