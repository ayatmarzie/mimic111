import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import accuracy_score

# Step 1: Load the data

file_path = './mimic_database/mapped_elements/'  # Replace with your CSV file path
file_name='CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_9604_plus_notes.csv'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.utils import to_categorical


# Load the dataset  # Update with your actual file path
#icd9_counts_df = pd.read_csv('the icd9s.txt', sep='\t')

df = pd.read_csv(file_path+file_name)




# Preprocess the data
# Drop rows with NaN ICD9_CODE (if any)
df = df.dropna(subset=['ICD9_CODE'])

# Encode ICD9_CODE as a categorical target
label_encoder = LabelEncoder()
df['ICD9_CODE'] = label_encoder.fit_transform(df['ICD9_CODE'])
#del df['CKD']
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

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Generate predictions
y_pred = model.predict(X_test)

# Convert predictions and true labels from one-hot encoding to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
model.save(f'icd9_classification_model_notes_9604.h5')
# Classification report
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))
print(f"\nClassification Report of icd9 code:")
print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))
with open(f"classification_report_notes_9604","w")as file:
    file.write(classification_report(y_test_classes, y_pred_classes, target_names=target_names))
