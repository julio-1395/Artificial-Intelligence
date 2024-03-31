from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = nb_model.predict(X_test_vec)

# Evaluate performance
accuracy = accuracy_score(y_test_enc, y_pred)
report = classification_report(y_test_enc, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

