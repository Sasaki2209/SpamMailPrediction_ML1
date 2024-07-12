import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and prepare data
mail_data = pd.read_csv('mail_data.csv')
mail_data['Category'] = mail_data['Category'].str.strip(',').replace({'spam': '0', 'ham': '1'}).astype(int)
X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Logistic Regression without regularization (very high C)
model_no_reg = LogisticRegression(C=1e12)
model_no_reg.fit(X_train_features, Y_train)
predictions_no_reg = model_no_reg.predict(X_test_features)

# Logistic Regression with regularization (default C=1.0)
model_with_reg = LogisticRegression(C=1.0)  # Typical regularization
model_with_reg.fit(X_train_features, Y_train)
predictions_with_reg = model_with_reg.predict(X_test_features)

# Evaluate models
print("Logistic Regression without Regularization:")
print(f"Accuracy: {accuracy_score(Y_test, predictions_no_reg)}")
print("Classification Report:\n", classification_report(Y_test, predictions_no_reg, target_names=['Spam', 'Ham']))
print("Confusion Matrix:\n", confusion_matrix(Y_test, predictions_no_reg))

print("\nLogistic Regression with Regularization:")
print(f"Accuracy: {accuracy_score(Y_test, predictions_with_reg)}")
print("Classification Report:\n", classification_report(Y_test, predictions_with_reg, target_names=['Spam', 'Ham']))
print("Confusion Matrix:\n", confusion_matrix(Y_test, predictions_with_reg))

# Plot confusion matrix for both models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(Y_test, predictions_no_reg), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix: No Regularization')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(Y_test, predictions_with_reg), annot=True, fmt='d', ax=axes[1], cmap='Reds')
axes[1].set_title('Confusion Matrix: With Regularization')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.show()

# Distribution of ham and spam emails
plt.figure(figsize=(10, 6))
sns.countplot(data=mail_data, x='Category')
plt.title('Distribution of Ham and Spam Emails')
plt.xlabel('Email Category (0 = Spam, 1 = Ham)')
plt.ylabel('Count')
plt.show()

# Learning curve function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves
plot_learning_curve(model_no_reg, 'Learning Curve (No Regularization)', X_train_features, Y_train, cv=5)
plot_learning_curve(model_with_reg, 'Learning Curve (With Regularization)', X_train_features, Y_train, cv=5)
plt.show()

