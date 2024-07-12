import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Load the dataset
mail_data = pd.read_csv('mail_data.csv')

# Show demo of dataset
print(mail_data)

# Strip any trailing commas or spaces from the 'Category' column
mail_data['Category'] = mail_data['Category'].str.strip(',')

# Replace 'spam' and 'ham' with 0 and 1 without downcasting warning
# First replace 'spam' and 'ham' with strings '0' and '1', then convert to integers explicitly
mail_data['Category'] = mail_data['Category'].replace({'spam': '0', 'ham': '1'})

# Convert the 'Category' column to integers
mail_data['Category'] = mail_data['Category'].astype(int)

# Prepare data for training
X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Initialize TF-IDF Vectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform on training data
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# Initialize and fit Logistic Regression model
# Here is how the Regularization is applied, 'LogisticRegression' in 'scikit-learn' applies L2 regularization.
logistic_model = LogisticRegression()
'''
 Here, C=1.0 is the default value. Lowering the value of C increases the strength of the regularization (more penalty)
 while increasing C reduces the regularization strength (less penalty).
 For example, C=0.1 would apply stronger regularization.
'''
#logistic_model = LogisticRegression(C=1.0)
#logistic_model = LogisticRegression(C=0.5, penalty='l2')  # Apply L2 regularization with specific strength
logistic_model.fit(X_train_features, Y_train)


# Predict on training data and calculate accuracy
prediction_on_training_data = logistic_model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print(f"Accuracy on training data (Logistic Regression): {accuracy_on_training_data}")

# Example of predicting on new input using Logistic Regression
input_mail = ["I've been searching for the right words to thank you..."]
input_data_features = feature_extraction.transform(input_mail)
logistic_prediction = logistic_model.predict(input_data_features)

if logistic_prediction[0] == 1:
    print('Ham mail (Logistic Regression)')
else:
    print('Spam mail (Logistic Regression)')

# Naive Bayes Classification
nb_model = MultinomialNB()
nb_model.fit(X_train_features, Y_train)

# Predict on training data and calculate accuracy
nb_prediction_on_training_data = nb_model.predict(X_train_features)
nb_accuracy_on_training_data = accuracy_on_training_data = accuracy_score(Y_train, nb_prediction_on_training_data)
print(f"Accuracy on training data (Naive Bayes): {nb_accuracy_on_training_data}")

# Example of predicting on new input using Naive Bayes
nb_prediction = nb_model.predict(input_data_features)

if nb_prediction[0] == 1:
    print('Ham mail (Naive Bayes)')
else:
    print('Spam mail (Naive Bayes)')

# Plot histogram of mail categories
plt.figure(figsize=(8, 6))
mail_data['Category'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Distribution of Ham and Spam Emails')
plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'], rotation=0)
plt.show()

# Create learning curve for Naive Bayes model
train_sizes, train_scores, test_scores = learning_curve(nb_model, X_train_features, Y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score', color='blue')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', color='red')
plt.title('Learning Curve for Naive Bayes Classifier')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

# Count ham and spam mail
model = LogisticRegression()
model.fit(X_train_features, Y_train)

raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

A = mail_data['Message']
input_mail = raw_mail_data['Message']

input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
data = prediction

count_1 = np.sum(data == 1) #ham
count_0 = np.sum(data == 0) #spam

print(f"number of ham mails: {count_1}")

print(f"number of spam mails: {count_0}")
