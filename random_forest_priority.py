import pandas as pd
import os
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from queue import PriorityQueue

# ====================== CONSTANTS ======================
CSV_FILE = "job_descriptions.csv"
MODEL_FILE = "random_forest_model.joblib"

# ====================== DATABASE FUNCTIONS ======================
def load_dataset(csv_file=CSV_FILE):
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=['Name', 'Age', 'Gender', 'EdLevel', 'YearsCode', 'YearsCodePro',
                                 'Country', 'PreviousSalary', 'HaveWorkedWith', 'ComputerSkills',
                                 'MentalHealth', 'Employed', 'JobRole', 'Username', 'Password',
                                 'Status', 'Feedback', 'ApplicationDate', 'PriorityScore'])
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
    
    # Ensure all required columns exist
    for col in ['Name', 'Gender', 'EdLevel', 'Country', 'HaveWorkedWith', 'JobRole']:
        if col not in df.columns:
            df[col] = ""
    
    for col in ['Age', 'YearsCode', 'YearsCodePro', 'ComputerSkills', 'PreviousSalary', 'PriorityScore']:
        if col not in df.columns:
            df[col] = 0
    
    if 'Employed' not in df.columns:
        df['Employed'] = False
    if 'MentalHealth' not in df.columns:
        df['MentalHealth'] = "Fair"
    if 'Username' not in df.columns:
        df['Username'] = ""
    if 'Password' not in df.columns:
        df['Password'] = ""
    if 'Status' not in df.columns:
        df['Status'] = "Pending"
    if 'Feedback' not in df.columns:
        df['Feedback'] = ""
    if 'ApplicationDate' not in df.columns:
        df['ApplicationDate'] = datetime.now().strftime("%Y-%m-%d")
    
    return df

def save_dataset(df):
    df.to_csv(CSV_FILE, index=False)

# ====================== RANDOM FOREST MODEL FUNCTIONS ======================
def train_random_forest_model(df):
    # Prepare the data for training
    X = pd.get_dummies(df[['Age', 'Gender', 'EdLevel', 'YearsCode', 'YearsCodePro', 
                          'ComputerSkills', 'MentalHealth', 'Employed', 'JobRole']])
    y = df['PriorityScore']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    
    return model

def load_or_train_model(df):
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = train_random_forest_model(df)
    return model

def predict_priority_score(candidate_data, model, df):
    # Create a temporary DataFrame with the candidate's data
    temp_df = pd.DataFrame([candidate_data])
    
    # Ensure all categorical columns are present
    full_df = pd.concat([df, temp_df], ignore_index=True)
    
    # Prepare the data for prediction
    X_pred = pd.get_dummies(full_df[['Age', 'Gender', 'EdLevel', 'YearsCode', 'YearsCodePro', 
                                   'ComputerSkills', 'MentalHealth', 'Employed', 'JobRole']])
    X_pred = X_pred.tail(1)  # Get only the last row (our candidate)
    
    # Ensure all expected columns are present
    if os.path.exists(MODEL_FILE):
        expected_columns = joblib.load(MODEL_FILE).feature_names_in_
        for col in expected_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
    
        # Reorder columns to match training data
        X_pred = X_pred[expected_columns]
    
        # Predict the priority score
        predicted_score = model.predict(X_pred)[0]
    else:
        # Fallback to rule-based scoring if model doesn't exist
        predicted_score = calculate_priority_fallback(candidate_data)
    
    # Scale to 0-150 range
    predicted_score = np.clip(predicted_score, 0, 150)
    
    return predicted_score

def calculate_priority_fallback(row):
    """Fallback priority calculation if model isn't trained yet"""
    score = 0
    ed_scores = {"PhD": 30, "Master": 20, "Bachelor": 15, "High School": 5}
    score += ed_scores.get(row['EdLevel'], 0)
    score += min(row['YearsCodePro'] * 4, 40)
    score += min(row['ComputerSkills'] * 2, 20)
    
    age = row['Age']
    if 25 <= age <= 35: score += 20
    elif 36 <= age <= 45: score += 15
    elif 18 <= age <= 24: score += 10
    elif 46 <= age <= 55: score += 5
    
    job_weights = {
        "Web Developer": 30, "Data Scientist": 30, "DevOps Engineer": 30,
        "Project Manager": 25, "Business Analyst": 25, "UX Designer": 20,
        "Financial Analyst": 20, "Marketing Manager": 15, "Sales Executive": 15,
        "HR Specialist": 10
    }
    score += job_weights.get(row['JobRole'], 0)
    
    if row['MentalHealth'] == "Good": score += 10
    
    return min(score, 150)

# ====================== PRIORITY QUEUE CLASS ======================
class CandidatePriorityQueue:
    def __init__(self, model, df):
        self.pq = PriorityQueue()
        self.model = model
        self.df = df
    
    def add_candidate(self, candidate_data):
        """Add a candidate to the priority queue with their Random Forest predicted score"""
        # Predict the priority score using Random Forest
        priority_score = predict_priority_score(candidate_data, self.model, self.df)
        
        # We use negative score because Python's PriorityQueue is a min-heap
        # and we want highest scores first
        self.pq.put((-priority_score, candidate_data['Name'], candidate_data))
    
    def get_next_candidate(self):
        """Get the highest priority candidate from the queue"""
        if not self.pq.empty():
            return self.pq.get()[2]  # Return the candidate_data
        return None
    
    def size(self):
        return self.pq.qsize()
    
    def is_empty(self):
        return self.pq.empty()

# ====================== INITIALIZATION ======================
def initialize_priority_system():
    """Initialize the priority system and return model and queue"""
    df = load_dataset()
    model = load_or_train_model(df)
    priority_queue = CandidatePriorityQueue(model, df)
    return model, priority_queue