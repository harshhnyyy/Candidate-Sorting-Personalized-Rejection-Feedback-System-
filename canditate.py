import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from random_forest_priority import initialize_priority_system, predict_priority_score

# ====================== CONSTANTS ======================
JOB_ROLES = ["Web Developer", "Project Manager", "Business Analyst", "HR Specialist", 
             "Data Scientist", "UX Designer", "DevOps Engineer", "Marketing Manager", 
             "Financial Analyst", "Sales Executive"]
BG_COLOR = "#f5f5f5"
HEADER_COLOR = "#2E7D32"
BUTTON_COLOR = "#1B5E20"
TEXT_COLOR = "#333333"
ACCENT_COLOR = "#2E7D32"
ERROR_COLOR = "#c62828"
ENTRY_BG = "#ffffff"
CSV_FILE = "job_descriptions.csv"

# Initialize the priority system
model, priority_queue = initialize_priority_system()

# ====================== AI FEEDBACK GENERATOR ======================
def generate_ai_feedback(candidate_data):
    feedback = []
    role_requirements = {
        "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics", "Data Analysis"],
        "Web Developer": ["JavaScript", "HTML/CSS", "React", "Node.js", "Frontend"],
        "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Infrastructure"],
        "Project Manager": ["Leadership", "Agile", "Scrum", "Communication", "Planning"],
        "Business Analyst": ["SQL", "Excel", "Requirements", "Documentation", "Analysis"],
        "UX Designer": ["Figma", "User Research", "Wireframing", "Prototyping", "UI/UX"],
        "Marketing Manager": ["SEO", "Content", "Social Media", "Advertising", "Branding"],
        "Financial Analyst": ["Excel", "Financial Modeling", "Accounting", "Forecasting", "Analysis"],
        "Sales Executive": ["CRM", "Negotiation", "Communication", "Relationship", "Sales"],
        "HR Specialist": ["Recruitment", "Employee Relations", "HR Policies", "Interviewing", "Compliance"]
    }
    
    if candidate_data['JobRole'] in role_requirements:
        missing_skills = [skill for skill in role_requirements[candidate_data['JobRole']] 
                        if skill.lower() not in candidate_data['HaveWorkedWith'].lower()]
        if missing_skills:
            feedback.append(f"For {candidate_data['JobRole']} roles, we recommend gaining experience with: {', '.join(missing_skills)}")
    
    if candidate_data['YearsCodePro'] < 3:
        feedback.append(f"More professional experience would strengthen your application (currently {candidate_data['YearsCodePro']} years). Consider internships or freelance work.")
    elif candidate_data['YearsCodePro'] < 5:
        feedback.append(f"While you have {candidate_data['YearsCodePro']} years of experience, additional professional experience would make you more competitive.")
    
    if candidate_data['EdLevel'] == "High School":
        feedback.append("Consider pursuing higher education or professional certifications to be more competitive.")
    elif candidate_data['EdLevel'] == "Bachelor" and candidate_data['JobRole'] in ["Data Scientist", "DevOps Engineer"]:
        feedback.append("For this technical role, a Master's degree or specialized certifications could be beneficial.")
    
    if candidate_data['ComputerSkills'] < 5:
        feedback.append(f"Your computer skills rating ({candidate_data['ComputerSkills']}/10) could be improved through courses or certifications.")
    elif candidate_data['ComputerSkills'] < 8:
        feedback.append(f"Your computer skills are decent ({candidate_data['ComputerSkills']}/10), but reaching 8+ would make you more competitive.")
    
    if candidate_data['MentalHealth'] == "Poor":
        feedback.append("We noticed you reported poor mental health. Many companies offer wellness programs that could help.")
    
    if 'PreviousSalary' in candidate_data and candidate_data['PreviousSalary'] > 0:
        avg_salaries = {
            "Data Scientist": 120000, "Web Developer": 85000, "DevOps Engineer": 110000,
            "Project Manager": 95000, "Business Analyst": 80000, "UX Designer": 75000,
            "Financial Analyst": 90000, "Marketing Manager": 80000, "Sales Executive": 70000,
            "HR Specialist": 65000
        }
        if candidate_data['JobRole'] in avg_salaries:
            avg_salary = avg_salaries[candidate_data['JobRole']]
            ratio = candidate_data['PreviousSalary'] / avg_salary
            if ratio > 1.2:
                feedback.append(f"Your previous salary (${candidate_data['PreviousSalary']:,.0f}) is significantly higher than average for this role (${avg_salary:,.0f}).")
            elif ratio < 0.8:
                feedback.append(f"Your previous salary (${candidate_data['PreviousSalary']:,.0f}) is below average for this role (${avg_salary:,.0f}), which could work in your favor.")
    
    return "AI Feedback:\n- " + "\n- ".join(feedback) if feedback else "No specific feedback available. Your profile looks good overall, but the competition was particularly strong for this role."

# ====================== DATABASE FUNCTIONS ======================
def load_dataset(csv_file=CSV_FILE):
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=['Name', 'Age', 'Gender', 'EdLevel', 'YearsCode', 'YearsCodePro',
                                 'Country', 'PreviousSalary', 'HaveWorkedWith', 'ComputerSkills',
                                 'MentalHealth', 'Employed', 'JobRole', 'Username', 'Password',
                                 'Status', 'Feedback', 'ApplicationDate'])
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
    
    required_cols = ['Name', 'Age', 'Gender', 'EdLevel', 'YearsCode', 'YearsCodePro',
                    'Country', 'PreviousSalary', 'HaveWorkedWith', 'ComputerSkills',
                    'MentalHealth', 'Employed', 'JobRole']
    
    for col in required_cols:
        if col not in df.columns:
            if col in ['Name', 'Gender', 'EdLevel', 'Country', 'HaveWorkedWith', 'JobRole']:
                df[col] = ""
            elif col in ['Age', 'YearsCode', 'YearsCodePro', 'ComputerSkills', 'PreviousSalary']:
                df[col] = 0
            elif col == 'Employed':
                df[col] = False
            elif col == 'MentalHealth':
                df[col] = "Fair"
    
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

# ====================== INITIALIZE DATA ======================
df = load_dataset()
save_dataset(df)

encoders = {col: LabelEncoder() for col in ['EdLevel', 'Country', 'HaveWorkedWith', 'Gender', 'JobRole']}
for col, encoder in encoders.items():
    if col in df.columns:
        unique_values = df[col].astype(str).unique()
        encoder.fit(unique_values)
        df[f'{col}_enc'] = encoder.transform(df[col].astype(str))

def calculate_priority(row):
    """Calculate priority score using the Random Forest model"""
    global model, df
    candidate_data = row.to_dict()
    return predict_priority_score(candidate_data, model, df)

if 'PriorityScore' not in df.columns:
    df['PriorityScore'] = df.apply(calculate_priority, axis=1)

credentials = {}
for _, row in df.iterrows():
    if isinstance(row['Username'], str) and row['Username'].strip():
        credentials[row['Username'].strip().lower()] = row['Password']
credentials["admin"] = "admin123"

# ====================== GUI CLASSES ======================
class LoginPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Job Application Portal - Login")
        self.root.geometry("500x500")
        self.root.configure(bg=BG_COLOR)
        self.root.eval('tk::PlaceWindow . center')
        
        self.title_font = ("Arial", 24, "bold")
        self.label_font = ("Arial", 12)
        
        tk.Frame(root, bg=HEADER_COLOR, height=70).pack(fill=tk.X, side=tk.TOP)
        
        main_frame = tk.Frame(root, bg=BG_COLOR)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
        
        tk.Label(main_frame, text="Job Application Portal", 
                font=self.title_font, bg=BG_COLOR, fg=HEADER_COLOR).pack(pady=(0, 30))
        
        form_frame = tk.Frame(main_frame, bg=BG_COLOR)
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="Username:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, sticky="w", pady=5)
        self.username_entry = tk.Entry(form_frame, font=self.label_font, bg=ENTRY_BG, 
                                     fg=TEXT_COLOR, relief=tk.FLAT, borderwidth=2, width=25)
        self.username_entry.grid(row=0, column=1, pady=5, padx=10, ipady=5)
        
        tk.Label(form_frame, text="Password:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=1, column=0, sticky="w", pady=5)
        self.password_entry = tk.Entry(form_frame, font=self.label_font, bg=ENTRY_BG, 
                                     fg=TEXT_COLOR, show="*", relief=tk.FLAT, borderwidth=2, width=25)
        self.password_entry.grid(row=1, column=1, pady=5, padx=10, ipady=5)
        
        button_frame = tk.Frame(main_frame, bg=BG_COLOR)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Admin Login", command=self.admin_login, 
                 font=self.label_font, bg=BUTTON_COLOR, fg="white", 
                 relief=tk.FLAT, bd=0, padx=20, pady=8).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="User Login", command=self.user_login, 
                 font=self.label_font, bg=BUTTON_COLOR, fg="white", 
                 relief=tk.FLAT, bd=0, padx=20, pady=8).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Register", command=self.register, 
                 font=self.label_font, bg=BUTTON_COLOR, fg="white", 
                 relief=tk.FLAT, bd=0, padx=20, pady=8).pack(fill=tk.X, pady=5)
    
    def admin_login(self):
        if self.username_entry.get() == "admin" and self.password_entry.get() == "admin123":
            self.root.destroy()
            root = tk.Tk()
            AdminDashboard(root)
            root.mainloop()
        else:
            messagebox.showerror("Error", "Invalid Admin Credentials")
    
    def user_login(self):
        username = self.username_entry.get().strip().lower()
        password = self.password_entry.get().strip()
        
        if username in credentials:
            if credentials[username] == password:
                self.root.destroy()
                root = tk.Tk()
                UserDashboard(root, username)
                root.mainloop()
            else:
                messagebox.showerror("Error", "Incorrect password")
        else:
            messagebox.showerror("Error", "Username not found")
    
    def register(self):
        self.root.destroy()
        root = tk.Tk()
        RegistrationPage(root)
        root.mainloop()

class RegistrationPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Job Application Portal - Registration")
        self.root.geometry("800x700")
        self.root.configure(bg=BG_COLOR)
        
        self.title_font = ("Arial", 20, "bold")
        self.label_font = ("Arial", 11)
        self.header_font = ("Arial", 12, "bold")
        
        self.canvas = tk.Canvas(root, bg=BG_COLOR, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=BG_COLOR)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        header_frame = tk.Frame(self.scrollable_frame, bg=HEADER_COLOR, height=60)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="New User Registration", font=self.title_font, 
                bg=HEADER_COLOR, fg="white").pack(pady=15)
        
        form_frame = tk.Frame(self.scrollable_frame, bg=BG_COLOR, padx=20, pady=20)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(form_frame, text="Personal Information", font=self.header_font, 
                bg=BG_COLOR, fg=HEADER_COLOR).grid(row=0, column=0, sticky="w", pady=10)
        
        fields = [
            ("Full Name:", "name_entry", None),
            ("Age:", "age_entry", None),
            ("Gender:", "gender_var", ["Male", "Female", "Other"]),
            ("Education Level:", "edlevel_var", ["High School", "Bachelor", "Master", "PhD"]),
            ("Country:", "country_entry", None)
        ]
        
        for i, (label, var_name, options) in enumerate(fields):
            tk.Label(form_frame, text=label, font=self.label_font, 
                    bg=BG_COLOR, fg=TEXT_COLOR).grid(row=i+1, column=0, sticky="w", pady=5)
            
            if options:
                setattr(self, var_name, tk.StringVar(value=options[0]))
                if len(options) > 2:
                    ttk.Combobox(form_frame, textvariable=getattr(self, var_name), 
                                values=options, font=self.label_font, state="readonly").grid(row=i+1, column=1, sticky="ew", pady=5, padx=10)
                else:
                    frame = tk.Frame(form_frame, bg=BG_COLOR)
                    frame.grid(row=i+1, column=1, sticky="w", pady=5, padx=10)
                    for j, option in enumerate(options):
                        tk.Radiobutton(frame, text=option, variable=getattr(self, var_name), 
                                      value=option, font=self.label_font, bg=BG_COLOR).pack(side=tk.LEFT, padx=5)
            else:
                entry = tk.Entry(form_frame, font=self.label_font, bg=ENTRY_BG, 
                                fg=TEXT_COLOR, relief=tk.FLAT, borderwidth=1)
                entry.grid(row=i+1, column=1, sticky="ew", pady=5, padx=10, ipady=3)
                setattr(self, var_name, entry)
        
        tk.Label(form_frame, text="Professional Information", font=self.header_font, 
                bg=BG_COLOR, fg=HEADER_COLOR).grid(row=6, column=0, sticky="w", pady=10)
        
        prof_fields = [
            ("Total Coding Experience (years):", "years_code_entry", None),
            ("Professional Coding Experience (years):", "years_pro_entry", None),
            ("Technologies Worked With:", "skills_entry", None),
            ("Computer Skills (1-10):", "comp_skills_var", None),
            ("Mental Health:", "mental_var", ["Good", "Fair", "Poor"]),
            ("Currently Employed:", "employed_var", None),
            ("Previous Salary (USD):", "salary_entry", None)
        ]
        
        for i, (label, var_name, options) in enumerate(prof_fields):
            tk.Label(form_frame, text=label, font=self.label_font, 
                    bg=BG_COLOR, fg=TEXT_COLOR).grid(row=i+7, column=0, sticky="w", pady=5)
            
            if options:
                setattr(self, var_name, tk.StringVar(value=options[0]))
                frame = tk.Frame(form_frame, bg=BG_COLOR)
                frame.grid(row=i+7, column=1, sticky="w", pady=5, padx=10)
                for j, option in enumerate(options):
                    tk.Radiobutton(frame, text=option, variable=getattr(self, var_name), 
                                  value=option, font=self.label_font, bg=BG_COLOR).pack(side=tk.LEFT, padx=5)
            elif var_name == "comp_skills_var":
                setattr(self, var_name, tk.IntVar(value=5))
                tk.Scale(form_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                         variable=getattr(self, var_name), font=self.label_font, 
                         bg=BG_COLOR, fg=TEXT_COLOR).grid(row=i+7, column=1, sticky="ew", pady=5, padx=10)
            elif var_name == "employed_var":
                setattr(self, var_name, tk.BooleanVar(value=True))
                tk.Checkbutton(form_frame, variable=getattr(self, var_name), 
                              font=self.label_font, bg=BG_COLOR).grid(row=i+7, column=1, sticky="w", pady=5, padx=10)
            else:
                entry = tk.Entry(form_frame, font=self.label_font, bg=ENTRY_BG, 
                               fg=TEXT_COLOR, relief=tk.FLAT, borderwidth=1)
                entry.grid(row=i+7, column=1, sticky="ew", pady=5, padx=10, ipady=3)
                setattr(self, var_name, entry)
        
        tk.Label(form_frame, text="Job Application", font=self.header_font, 
                bg=BG_COLOR, fg=HEADER_COLOR).grid(row=14, column=0, sticky="w", pady=10)
        
        tk.Label(form_frame, text="Desired Job Role:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=15, column=0, sticky="w", pady=5)
        self.jobrole_var = tk.StringVar(value=JOB_ROLES[0])
        ttk.Combobox(form_frame, textvariable=self.jobrole_var, 
                    values=JOB_ROLES, font=self.label_font, state="readonly").grid(row=15, column=1, sticky="ew", pady=5, padx=10)
        
        tk.Label(form_frame, text="Account Information", font=self.header_font, 
                bg=BG_COLOR, fg=HEADER_COLOR).grid(row=16, column=0, sticky="w", pady=10)
        
        tk.Label(form_frame, text="Choose Username:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=17, column=0, sticky="w", pady=5)
        self.username_entry = tk.Entry(form_frame, font=self.label_font, bg=ENTRY_BG, 
                                     fg=TEXT_COLOR, relief=tk.FLAT, borderwidth=1)
        self.username_entry.grid(row=17, column=1, sticky="ew", pady=5, padx=10, ipady=3)
        
        button_frame = tk.Frame(form_frame, bg=BG_COLOR)
        button_frame.grid(row=18, column=0, columnspan=2, pady=20)
        
        tk.Button(button_frame, text="Submit Application", command=self.submit_application, 
                 font=self.header_font, bg=BUTTON_COLOR, fg="white", 
                 relief=tk.FLAT, bd=0, padx=20, pady=8).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Back to Login", command=self.back_to_login, 
                 font=self.header_font, bg=ERROR_COLOR, fg="white", 
                 relief=tk.FLAT, bd=0, padx=20, pady=8).pack(side=tk.LEFT, padx=10)
        
        form_frame.columnconfigure(1, weight=1)
    
    def validate_fields(self):
        required_fields = [
            (self.name_entry, "Full Name"),
            (self.age_entry, "Age"),
            (self.country_entry, "Country"),
            (self.years_code_entry, "Total Coding Experience"),
            (self.years_pro_entry, "Professional Coding Experience"),
            (self.skills_entry, "Technologies Worked With"),
            (self.salary_entry, "Previous Salary"),
            (self.username_entry, "Username")
        ]
        
        for field, name in required_fields:
            if not field.get().strip():
                messagebox.showerror("Error", f"Please fill in the {name} field")
                return False
        
        try:
            age = int(self.age_entry.get())
            if age < 18 or age > 70:
                messagebox.showerror("Error", "Age must be between 18 and 70")
                return False
        except ValueError:
            messagebox.showerror("Error", "Age must be a valid number")
            return False
        
        try:
            years_code = float(self.years_code_entry.get())
            years_pro = float(self.years_pro_entry.get())
            if years_pro > years_code:
                messagebox.showerror("Error", "Professional experience cannot exceed total coding experience")
                return False
        except ValueError:
            messagebox.showerror("Error", "Experience years must be valid numbers")
            return False
        
        try:
            salary = float(self.salary_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Salary must be a valid number")
            return False
        
        username = self.username_entry.get().strip().lower()
        if username in credentials:
            messagebox.showerror("Error", "Username already exists. Please choose a different one.")
            return False
        
        return True
    
    def submit_application(self):
        if not self.validate_fields():
            return
        
        new_user = {
            'Name': self.name_entry.get().strip(),
            'Age': int(self.age_entry.get()),
            'Gender': self.gender_var.get(),
            'EdLevel': self.edlevel_var.get(),
            'YearsCode': float(self.years_code_entry.get()),
            'YearsCodePro': float(self.years_pro_entry.get()),
            'Country': self.country_entry.get().strip(),
            'PreviousSalary': float(self.salary_entry.get()),
            'HaveWorkedWith': self.skills_entry.get().strip(),
            'ComputerSkills': self.comp_skills_var.get(),
            'MentalHealth': self.mental_var.get(),
            'Employed': self.employed_var.get(),
            'JobRole': self.jobrole_var.get(),
            'Username': self.username_entry.get().strip().lower(),
            'Password': f"{self.username_entry.get().strip().lower()}123",
            'Status': "Pending",
            'Feedback': "",
            'ApplicationDate': datetime.now().strftime("%Y-%m-%d")
        }
        
        global df, model, priority_queue
        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
        df['PriorityScore'] = df.apply(calculate_priority, axis=1)
        credentials[new_user['Username']] = new_user['Password']
        save_dataset(df)
        
        # Add to priority queue
        priority_queue.add_candidate(new_user)
        
        messagebox.showinfo("Success", "Application submitted successfully!\n"
                          f"Your password is: {new_user['Password']}")
        self.back_to_login()
    
    def back_to_login(self):
        self.root.destroy()
        root = tk.Tk()
        LoginPage(root)
        root.mainloop()

class AdminDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Recruiter Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_COLOR)
        
        self.title_font = ("Arial", 20, "bold")
        self.label_font = ("Arial", 12)
        self.small_font = ("Arial", 10)
        
        header_frame = tk.Frame(root, bg=HEADER_COLOR, height=70)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="Recruiter Dashboard", 
                font=self.title_font, bg=HEADER_COLOR, fg="white").pack(pady=15)
        
        main_frame = tk.Frame(root, bg=BG_COLOR)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        filter_frame = tk.Frame(main_frame, bg=BG_COLOR)
        filter_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(filter_frame, text="Search:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(filter_frame, textvariable=self.search_var, 
                               font=self.label_font, bg=ENTRY_BG, fg=TEXT_COLOR, 
                               relief=tk.FLAT, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind("<KeyRelease>", self.filter_candidates)
        
        tk.Label(filter_frame, text="Status:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=10)
        self.status_var = tk.StringVar(value="All")
        status_cb = ttk.Combobox(filter_frame, textvariable=self.status_var, 
                               values=["All", "Pending", "Approved", "Rejected"], 
                               font=self.label_font, state="readonly", width=12)
        status_cb.pack(side=tk.LEFT)
        status_cb.bind("<<ComboboxSelected>>", self.filter_candidates)
        
        tk.Label(filter_frame, text="Job Role:", font=self.label_font, 
                bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=10)
        self.jobrole_var = tk.StringVar(value="All")
        jobrole_cb = ttk.Combobox(filter_frame, textvariable=self.jobrole_var, 
                                values=["All"] + JOB_ROLES, 
                                font=self.label_font, state="readonly", width=15)
        jobrole_cb.pack(side=tk.LEFT)
        jobrole_cb.bind("<<ComboboxSelected>>", self.filter_candidates)
        
        tree_frame = tk.Frame(main_frame, bg=BG_COLOR)
        tree_frame.pack(expand=True, fill=tk.BOTH, pady=10)
        
        self.tree_scroll = tk.Scrollbar(tree_frame)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.candidate_tree = ttk.Treeview(
            tree_frame, yscrollcommand=self.tree_scroll.set, 
            selectmode="browse", columns=("Name", "Age", "Job Role", "Experience", "Status", "Priority"), 
            show="headings")
        
        self.tree_scroll.config(command=self.candidate_tree.yview)
        
        for col, width in [("Name", 200), ("Age", 60), ("Job Role", 150), 
                          ("Experience", 100), ("Status", 100), ("Priority", 80)]:
            self.candidate_tree.column(col, width=width, anchor=tk.CENTER)
            self.candidate_tree.heading(col, text=col)
        
        self.populate_treeview()
        self.candidate_tree.pack(expand=True, fill=tk.BOTH)
        
        button_frame = tk.Frame(main_frame, bg=BG_COLOR)
        button_frame.pack(pady=10)
        
        actions = [
            ("View Details", self.view_details),
            ("Approve", lambda: self.update_status("Approved")),
            ("Reject", lambda: self.update_status("Rejected")),
            ("Logout", self.logout)
        ]
        
        for text, cmd in actions:
            color = BUTTON_COLOR if text != "Logout" else ERROR_COLOR
            tk.Button(button_frame, text=text, command=cmd, 
                     font=self.label_font, bg=color, fg="white", 
                     relief=tk.FLAT, bd=0, padx=15, pady=5).pack(side=tk.LEFT, padx=5)
    
    def populate_treeview(self, status_filter="All", jobrole_filter="All", search_text=""):
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        filtered_df = df.sort_values('PriorityScore', ascending=False)
        
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
        if jobrole_filter != "All":
            filtered_df = filtered_df[filtered_df['JobRole'] == jobrole_filter]
        if search_text:
            search_text = search_text.lower()
            filtered_df = filtered_df[filtered_df['Name'].str.lower().str.contains(search_text) | 
                                     filtered_df['JobRole'].str.lower().str.contains(search_text) |
                                     filtered_df['Country'].str.lower().str.contains(search_text)]
        
        for _, row in filtered_df.iterrows():
            self.candidate_tree.insert("", tk.END, 
                values=(row['Name'], row['Age'], row['JobRole'], 
                       f"{row['YearsCodePro']} yrs", row['Status'], 
                       row['PriorityScore']))
    
    def filter_candidates(self, event=None):
        self.populate_treeview(
            self.status_var.get(), 
            self.jobrole_var.get(),
            self.search_var.get()
        )
    
    def get_selected_candidate(self):
        try:
            selected_item = self.candidate_tree.selection()[0]
            name = self.candidate_tree.item(selected_item)['values'][0]
            return df[df['Name'] == name].index[0]
        except:
            messagebox.showwarning("Warning", "Please select a candidate first")
            return None
    
    def view_details(self):
        selected_index = self.get_selected_candidate()
        if selected_index is None:
            return
        
        candidate = df.iloc[selected_index]
        details_window = tk.Toplevel(self.root)
        details_window.title("Candidate Details")
        details_window.geometry("700x500")
        details_window.configure(bg=BG_COLOR)
        
        header_frame = tk.Frame(details_window, bg=HEADER_COLOR)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text=candidate['Name'], 
                font=self.title_font, bg=HEADER_COLOR, fg="white").pack(pady=10)
        
        main_frame = tk.Frame(details_window, bg=BG_COLOR)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        score_frame = tk.Frame(main_frame, bg=BG_COLOR)
        score_frame.pack(side=tk.RIGHT, padx=20)
        
        score_canvas = tk.Canvas(score_frame, width=150, height=150, bg=BG_COLOR, highlightthickness=0)
        score_canvas.pack()
        
        score_canvas.create_oval(10, 10, 140, 140, fill="#E8F5E9", outline=HEADER_COLOR, width=3)
        score_canvas.create_text(75, 60, text="PRIORITY", font=("Arial", 12, "bold"), fill=TEXT_COLOR)
        score_canvas.create_text(75, 90, text=str(candidate['PriorityScore']), 
                               font=("Arial", 24, "bold"), fill=HEADER_COLOR)
        score_canvas.create_text(75, 120, text=f"/150", font=("Arial", 12), fill=TEXT_COLOR)
        
        details_frame = tk.Frame(main_frame, bg=BG_COLOR)
        details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(details_frame, bg=BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        left_labels = [
            f"Age: {candidate['Age']}",
            f"Gender: {candidate['Gender']}",
            f"Education: {candidate['EdLevel']}",
            f"Job Role: {candidate['JobRole']}",
            f"Country: {candidate['Country']}"
        ]
        
        for label in left_labels:
            tk.Label(left_frame, text=label, font=self.label_font, 
                    bg=BG_COLOR, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, pady=5)
        
        right_frame = tk.Frame(details_frame, bg=BG_COLOR)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_labels = [
            f"Professional Exp: {candidate['YearsCodePro']} yrs",
            f"Total Coding Exp: {candidate['YearsCode']} yrs",
            f"Skills: {candidate['HaveWorkedWith']}",
            f"Computer Skills: {candidate['ComputerSkills']}/10",
            f"Mental Health: {candidate['MentalHealth']}"
        ]
        
        for label in right_labels:
            tk.Label(right_frame, text=label, font=self.label_font, 
                    bg=BG_COLOR, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, pady=5)
        
        status_frame = tk.Frame(main_frame, bg=BG_COLOR)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(status_frame, text=f"Status: {candidate['Status']}", 
                font=self.label_font, bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
    
    def update_status(self, new_status):
        selected_index = self.get_selected_candidate()
        if selected_index is None:
            return
        
        df.at[selected_index, 'Status'] = new_status
        if new_status == "Rejected":
            candidate_data = df.iloc[selected_index]
            ai_feedback = generate_ai_feedback(candidate_data)
            df.at[selected_index, 'Feedback'] = ai_feedback
        
        save_dataset(df)
        self.populate_treeview(self.status_var.get(), self.jobrole_var.get())
        
        popup = tk.Toplevel(self.root)
        popup.title("Status Updated")
        popup.geometry("400x200")
        popup.configure(bg=BG_COLOR)
        
        color = ACCENT_COLOR if new_status == "Approved" else ERROR_COLOR
        tk.Label(popup, text=new_status.upper() + "!", 
                font=("Arial", 36), bg=BG_COLOR, fg=color).pack(expand=True)
        tk.Button(popup, text="OK", command=popup.destroy, 
                 font=self.label_font, bg=BUTTON_COLOR, fg="white").pack(pady=10)
    
    def logout(self):
        self.root.destroy()
        root = tk.Tk()
        LoginPage(root)
        root.mainloop()

class UserDashboard:
    def __init__(self, root, username):
        self.root = root
        self.username = username
        self.user_data = df[df['Username'] == username].iloc[0]
        
        self.root.title("User Dashboard")
        self.root.geometry("800x700")  # Increased height to accommodate feedback
        self.root.configure(bg=BG_COLOR)
        
        # Header
        header_frame = tk.Frame(root, bg=HEADER_COLOR, height=70)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="User Dashboard", 
                font=("Arial", 20, "bold"), bg=HEADER_COLOR, fg="white").pack(pady=15)
        
        # Main container with scrollbar
        main_canvas = tk.Canvas(root, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas, bg=BG_COLOR)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status frame
        status_frame = tk.Frame(scrollable_frame, bg="#E8F5E9", bd=2, relief=tk.GROOVE)
        status_frame.pack(fill=tk.X, pady=10, padx=20)
        tk.Label(status_frame, text=f"Application Status: {self.user_data['Status'].upper()}", 
                font=("Arial", 14, "bold"), bg="#E8F5E9", fg=TEXT_COLOR).pack(pady=10)
        
        # Priority score
        priority_frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
        priority_frame.pack(pady=10)
        self.show_priority_score(priority_frame, self.user_data['PriorityScore'])
        
        # Details frame
        details_frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
        details_frame.pack(fill=tk.X, pady=10, padx=20)
        
        # Left column
        left_frame = tk.Frame(details_frame, bg=BG_COLOR)
        left_frame.pack(side="left", fill=tk.BOTH, expand=True)
        
        left_labels = [
            f"Name: {self.user_data['Name']}",
            f"Age: {self.user_data['Age']}",
            f"Gender: {self.user_data['Gender']}",
            f"Education: {self.user_data['EdLevel']}",
            f"Job Role: {self.user_data['JobRole']}",
            f"Country: {self.user_data['Country']}"
        ]
        
        for label in left_labels:
            tk.Label(left_frame, text=label, font=("Arial", 11), 
                    bg=BG_COLOR, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, pady=5)
        
        # Right column
        right_frame = tk.Frame(details_frame, bg=BG_COLOR)
        right_frame.pack(side="left", fill=tk.BOTH, expand=True)
        
        right_labels = [
            f"Professional Exp: {self.user_data['YearsCodePro']} yrs",
            f"Total Coding Exp: {self.user_data['YearsCode']} yrs",
            f"Skills: {self.user_data['HaveWorkedWith']}",
            f"Computer Skills: {self.user_data['ComputerSkills']}/10",
            f"Mental Health: {self.user_data['MentalHealth']}",
            f"Employed: {'Yes' if self.user_data['Employed'] else 'No'}"
        ]
        
        for label in right_labels:
            tk.Label(right_frame, text=label, font=("Arial", 11), 
                    bg=BG_COLOR, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X, pady=5)
        
        # Feedback section - now fully visible without scrolling
        feedback_frame = tk.Frame(scrollable_frame, bg="#E3F2FD", bd=1, relief=tk.SOLID)
        feedback_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(feedback_frame, text="Application Feedback:", 
                font=("Arial", 14, "bold"), bg="#E3F2FD", fg=TEXT_COLOR).pack(anchor="w", padx=10, pady=5)
        
        feedback_text = tk.Text(feedback_frame, height=10, width=80, 
                              wrap=tk.WORD, font=("Arial", 11), 
                              bg="#E3F2FD", fg=TEXT_COLOR, bd=0)
        
        feedback = self.user_data['Feedback'] if self.user_data['Feedback'] else "Your application is still under review. No feedback available yet."
        feedback_text.insert(tk.END, feedback)
        feedback_text.config(state=tk.DISABLED)
        
        # Add scrollbar to feedback text
        scrollbar = tk.Scrollbar(feedback_frame, orient="vertical", command=feedback_text.yview)
        scrollbar.pack(side="right", fill="y")
        feedback_text.config(yscrollcommand=scrollbar.set)
        feedback_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Logout button - fixed at bottom right
        logout_frame = tk.Frame(scrollable_frame, bg=BG_COLOR)
        logout_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        tk.Button(logout_frame, text="LOGOUT", command=self.logout, 
                 font=("Arial", 14, "bold"), bg=ERROR_COLOR, fg="white", 
                 relief=tk.FLAT, padx=30, pady=5).pack(side=tk.RIGHT, padx=20)
        
        # Make sure feedback is visible by default
        main_canvas.yview_moveto(0)
    
    def show_priority_score(self, parent, score):
        fig, ax = plt.subplots(figsize=(4, 4), facecolor=BG_COLOR)
        wedge, _ = ax.pie([score, 150-score], 
                         colors=[self.get_score_color(score), "#f0f0f0"], 
                         startangle=90)
        plt.setp(wedge, width=0.4)
        ax.text(0, 0, f"{score}", ha='center', va='center', 
               fontsize=24, fontweight='bold', color=self.get_score_color(score))
        ax.axis('equal')
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    def get_score_color(self, score):
        if score >= 100: return "#4CAF50"
        elif score >= 75: return "#8BC34A"
        elif score >= 50: return "#FFC107"
        elif score >= 25: return "#FF9800"
        else: return "#F44336"
    
    def logout(self):
        self.root.destroy()
        root = tk.Tk()
        LoginPage(root)
        root.mainloop()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    root = tk.Tk()
    LoginPage(root)
    root.mainloop()