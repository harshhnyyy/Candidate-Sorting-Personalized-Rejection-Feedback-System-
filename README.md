# Candidate Sorting & Personalised Rejection Feedback System

An end-to-end AI pipeline that automates candidate shortlisting and generates personalised rejection feedback using machine learning and natural language processing.

## What it does
- Ingests structured candidate data and preprocesses it for model input
- Uses a Random Forest classifier to rank and shortlist candidates based on defined criteria
- Generates personalised, human-readable rejection feedback for non-shortlisted candidates using NLP
- Presents results through a clean Tkinter desktop interface

## Results
- Reduced manual screening effort by 60%
- Validated with precision and recall benchmarking
- Published in the International Research Journal on Advanced Engineering Hub (IRJAEH), May 2025

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Tkinter, Git

## How to run
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the application: python main.py

## Known limitations
- Trained on a specific dataset structure; new datasets require column name alignment
- Feedback generation is template-guided and works best with structured input fields
- Not tested on datasets exceeding 10,000 rows
