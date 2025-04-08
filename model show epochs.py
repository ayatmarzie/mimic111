import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load the data
file_path = './mimic_database/mapped_elements/'  # Replace with your CSV file path
file_name = 'CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_3893_plus_notes.csv'

# Load the dataset
df = pd.read_csv(file_path + file_name)

# Preprocess the data
# Drop rows with NaN CKD column (if any)
df = df.dropna(subset=['CKD'])

# Encode CKD as a categorical target
label_encoder = LabelEncoder()
df['ICD9_CODE'] = label_encoder.fit_transform(df['CKD'])
del df['CKD']
del df['HADM_ID']
del df['HADMID_DAY']
del df['ADMITTIME']

num_classes = len(label_encoder.classes_)

# Separate features (X) and target (y)
X = df.drop(['ICD9_CODE'], axis=1)  # Drop target column from features
y = to_categorical(df['ICD9_CODE'], num_classes=num_classes)  # One-hot encode target

# Identify numeric columns for scaling
numeric_columns = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Convert non-numeric features to dummies
X = pd.get_dummies(X, drop_first=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Lists to store metrics over epochs
precision_scores = []
recall_scores = []
f1_micro_scores = []
f1_macro_scores = []

# Train the model and calculate metrics for each epoch
epochs = 20
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, verbose=1)

    # Generate predictions for the current epoch
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate metrics
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1_micro = f1_score(y_test_classes, y_pred_classes, average='micro')
    f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')

    # Append metrics to lists
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_micro_scores.append(f1_micro)
    f1_macro_scores.append(f1_macro)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Generate final classification report
y_pred_classes = np.argmax(model.predict(X_test), axis=1)
y_test_classes = np.argmax(y_test, axis=1)
target_names = [str(cls) for cls in label_encoder.classes_]
print(f"\nClassification Report of ICD9_CODE:")
print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))

# Save the model
model.save(f'icd9_classification_model_notes_3893.h5')

# Save classification report to file
with open(f"classification_report_notes_3893.txt", "w") as file:
    file.write(classification_report(y_test_classes, y_pred_classes, target_names=target_names))

# Plot the metrics
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), f1_micro_scores, label='F1-micro', color='blue')
plt.plot(range(1, epochs + 1), f1_macro_scores, label='F1-macro', color='orange')
plt.plot(range(1, epochs + 1), precision_scores, label='Precision', color='green')
plt.plot(range(1, epochs + 1), recall_scores, label='Recall', color='red')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('epachs for Sensorineural hearing loss, bilateral.png', format='eps', dpi=300, bbox_inches='tight')
plt.show()
