import pandas as pd
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Sample task data
data = {
    'Task': ['Complete report', 'Clean house', 'Buy groceries', 'Prepare meeting', 'Exercise', 'Call family'],
    'Urgency': [9, 2, 5, 8, 4, 6],  # Scale of 1-10
    'Importance': [8, 3, 5, 9, 6, 7],  # Scale of 1-10
    'Due_Date': [
        datetime.today(),
        datetime.today() + timedelta(days=7),
        datetime.today() + timedelta(days=2),
        datetime.today() + timedelta(hours=4),
        datetime.today() + timedelta(days=3),
        datetime.today() + timedelta(days=1)
    ]
}

df = pd.DataFrame(data)

# Step 2: Preprocessing - Add "Days Left" column
df['Days_Left'] = (df['Due_Date'] - datetime.now()).dt.days
df['Days_Left'] = df['Days_Left'].apply(lambda x: max(x, 0))  # Ensure no negative days

# Step 3: Rule-based system for prioritizing tasks
def rule_based_priority(task):
    if task['Urgency'] <= 3 and task['Importance'] <= 3:
        return 'Low Priority'
    elif task['Urgency'] >= 7 and task['Importance'] >= 7:
        return 'High Priority'
    return 'Medium Priority'

df['Rule_Based_Priority'] = df.apply(rule_based_priority, axis=1)

# Step 4: Encode priorities for training the decision tree
priority_map = {'Low Priority': 1, 'Medium Priority': 2, 'High Priority': 3}
df['Priority_Label'] = df['Rule_Based_Priority'].map(priority_map)

# Prepare features (Urgency, Importance, Days_Left) and target (Priority_Label)
X = df[['Urgency', 'Importance', 'Days_Left']]
y = df['Priority_Label']

# Step 5: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 7: Predict the priorities for the test set
y_pred = clf.predict(X_test)

# Step 8: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Model Accuracy: {accuracy * 100:.2f}%")

# Step 9: Apply the trained model to all tasks to predict their priority
df['Model_Predicted_Priority'] = clf.predict(df[['Urgency', 'Importance', 'Days_Left']])
df['Model_Predicted_Priority_Label'] = df['Model_Predicted_Priority'].map({1: 'Low Priority', 2: 'Medium Priority', 3: 'High Priority'})

# Step 10: Display the results
print("\nTask Prioritization (Rule-based vs. Model-based):")
print(df[['Task', 'Urgency', 'Importance', 'Days_Left', 'Rule_Based_Priority', 'Model_Predicted_Priority_Label']])
