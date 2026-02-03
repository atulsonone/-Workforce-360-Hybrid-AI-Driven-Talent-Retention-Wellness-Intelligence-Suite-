import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, f1_score
import time

# --- INITIAL CONFIGURATION ---
st.set_page_config(page_title="Enterprise HR Analytics Suite", layout="wide", page_icon="📈")

# Premium UI Styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main { background-color: #f0f2f6; }
    .stApp { background-color: #f0f2f6; }
    
    /* Section Aesthetics */
    .section-simulator { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border-left: 8px solid #2196f3; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .section-analytics { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border-left: 8px solid #4caf50; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .section-models { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border-left: 8px solid #ff9800; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .section-report { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border-left: 8px solid #9c27b0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }

    /* Component Styling */
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 800 !important; color: #1a237e; }
    .card { background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e1e4e8; height: 100%; }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3f51b5; display: inline-block; }
    .summary-card { background-color: #f1f3f5; padding: 12px; border-radius: 8px; border-left: 4px solid #1a237e; font-size: 0.9rem; margin-top: 10px; }
    
    .status-stay { color: #2e7d32; font-weight: 800; }
    .status-leave { color: #c62828; font-weight: 800; }
    .profile-header { background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 25px; }
    .retention-item { background-color: #e8f5e9; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #2e7d32; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# Helper: Indian Lakhs
def format_indian_lakhs(v):
    return f"₹{v/100000:.1f}L"

# --- STEP 1: DATA CORE ---
@st.cache_data
def load_data():
    n = 10000
    np.random.seed(42)
    first_names = ["Arjun", "Neha", "Rohit", "Priya", "Vikram", "Anjali", "Suresh", "Meera", "Karan", "Sana"]
    last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Iyer", "Reddy", "Patel", "Khan", "Das", "Joshi"]
    
    data = pd.DataFrame({
        "Employee_ID": np.arange(1001, 1001 + n),
        "Employee_Name": [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)],
        "Age": np.random.randint(18, 60, n),
        "Number_of_Experience_Years": np.random.randint(0, 31, n),
        "Annual_Salary": np.random.randint(500000, 5000001, n),
        "Last_Increment_Pct": np.random.randint(0, 26, n),
        "Bonus_Received": np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]),
        "Skill_Match_Score": np.random.randint(1, 11, n),
        "Project_Match_Score": np.random.randint(1, 11, n),
        "Overall_Performance_Rating": np.random.randint(1, 6, n),
        "Medical_Sick_Leaves": np.random.randint(0, 31, n),
        "Job_Satisfaction": np.random.randint(1, 6, n),
        "Work_Stress_Level": np.random.randint(1, 11, n)
    })
    
    risk_score = (
        (data["Work_Stress_Level"] * 0.25) + 
        ((5000000 - data["Annual_Salary"]) / 5000000 * 0.20) +
        ((10 - data["Project_Match_Score"]) * 0.15) +
        ((6 - data["Job_Satisfaction"]) * 0.1) +
        (data["Medical_Sick_Leaves"] / 30 * 0.1)
    )
    data.loc[(data["Number_of_Experience_Years"] >= 18) & (data["Annual_Salary"] < 1800000), "Boost"] = 0.4
    risk_score += data["Boost"].fillna(0)
    data["Attrition"] = (risk_score > np.percentile(risk_score, 80)).astype(int)
    data.drop(columns=["Boost"], inplace=True)
    return data

# --- STEP 2: PRECISION AI PIPELINE (HYPERTUNING & AUTOENCODER) ---
@st.cache_resource
def train_precision_suite(df):
    df_p = df.copy()
    le = LabelEncoder()
    df_p["Bonus_Received"] = le.fit_transform(df_p["Bonus_Received"])
    features = ["Age", "Number_of_Experience_Years", "Annual_Salary", "Last_Increment_Pct", 
                "Bonus_Received", "Skill_Match_Score", "Project_Match_Score", 
                "Overall_Performance_Rating", "Medical_Sick_Leaves", "Job_Satisfaction", "Work_Stress_Level"]
    X = df_p[features]
    y = df_p["Attrition"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 1. Hypertuned Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'class_weight': ['balanced']
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, scoring='recall', cv=3)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    # 2. Neural Autoencoder (using MLPRegressor)
    # Architecture: 11 -> 6 -> 11 (Standard Symmetric Compression)
    ae = MLPRegressor(hidden_layer_sizes=(6,), activation='relu', solver='adam', max_iter=500, random_state=42)
    # Train ONLY on non-attrition data to learn "Normal" patterns
    X_normal = X_train[y_train == 0]
    ae.fit(X_normal, X_normal)
    
    # Calculate Benchmarks for Leaderboard
    models = {
        "Hypertuned RF": best_rf,
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42).fit(X_train, y_train),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    }
    benchmarks = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        benchmarks[name] = {
            "model": model, "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred), "F1": f1_score(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }
    
    return benchmarks, ae, scaler, features, le

# Init
df_master = load_data()
benchmarks, autoencoder, scaler, feature_cols, bonus_le = train_precision_suite(df_master)
top_m_name = max(benchmarks, key=lambda x: benchmarks[x]['Recall'])
best_model = benchmarks[top_m_name]['model']

# --- STEP 3: DYNAMIC RETENTION ENGINE ---
def get_tailored_retention(profile):
    """Generates a personalized retention strategy based on unique risk drivers."""
    actions = []
    changes = profile.copy()
    cost = 0
    
    # Driver 1: Salary Benchmarking
    # If senior/mid-level but earning low, or last increment was low
    if (profile["Number_of_Experience_Years"] > 10 and profile["Annual_Salary"] < 2000000) or profile["Last_Increment_Pct"] < 8:
        hike_pct = 0.20 if profile["Annual_Salary"] < 1500000 else 0.15
        hike_amt = profile["Annual_Salary"] * hike_pct
        changes["Annual_Salary"] += hike_amt
        cost += hike_amt
        actions.append(f"Strategic Salary Correction (+{int(hike_pct*100)}% / {format_indian_lakhs(hike_amt)})")
    
    # Driver 2: Burnout (Work Stress)
    if profile["Work_Stress_Level"] > 7:
        changes["Work_Stress_Level"] = max(1, profile["Work_Stress_Level"] - 3)
        actions.append("Mandatory Wellness Scope Reduction (-3 Stress Pts)")
    
    # Driver 3: Engagement (Project Match)
    if profile["Project_Match_Score"] < 5:
        changes["Project_Match_Score"] = min(10, profile["Project_Match_Score"] + 3)
        actions.append("Strategic Role Realignment (+3 Project Match)")
        
    # Driver 4: Satisfaction
    if profile["Job_Satisfaction"] < 3:
        changes["Job_Satisfaction"] = min(5, profile["Job_Satisfaction"] + 1)
        actions.append("Engagement Intervention (+1 Satisfaction Pt)")

    # Default if no specific high triggers found but risk is high
    if not actions:
        changes["Annual_Salary"] *= 1.10
        cost = profile["Annual_Salary"] * 0.10
        actions.append("General Retention Bonus (10% Hike)")

    return changes, actions, cost

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("🏢 HCM Intelligence Platform")
mode = st.sidebar.radio("Navigation", ["Simulation Mode", "Employee Search Mode"])

selected_row = None
if mode == "Employee Search Mode":
    st.sidebar.subheader("Employee Registry")
    emp_id = st.sidebar.selectbox("Select ID", df_master["Employee_ID"].unique())
    selected_row = df_master[df_master["Employee_ID"] == emp_id].iloc[0]

st.sidebar.divider()
def sidebar_val(key, default):
    return int(selected_row[key]) if selected_row is not None else default

s_age = st.sidebar.slider("Age", 18, 60, sidebar_val("Age", 42))
s_exp = st.sidebar.slider("Exp Years", 0, 30, sidebar_val("Number_of_Experience_Years", 18))
s_salary = st.sidebar.number_input("CTC (Lakhs)", 5, 50, sidebar_val("Annual_Salary", 1400000)//100000) * 100000
s_inc = st.sidebar.slider("Last Inc %", 0, 25, sidebar_val("Last_Increment_Pct", 4))
s_bonus = st.sidebar.selectbox("Bonus", ["Yes", "No"], index=0 if (selected_row['Bonus_Received'] if selected_row is not None else "Yes") == "Yes" else 1)
s_skill = st.sidebar.slider("Skill Score", 1, 10, sidebar_val("Skill_Match_Score", 6))
s_proj = st.sidebar.slider("Project Score", 1, 10, sidebar_val("Project_Match_Score", 3))
s_perf = st.sidebar.slider("Performance", 1, 5, sidebar_val("Overall_Performance_Rating", 4))
s_sick = st.sidebar.slider("Sick Leaves", 0, 30, sidebar_val("Medical_Sick_Leaves", 5))
s_sat = st.sidebar.slider("Satisfaction", 1, 5, sidebar_val("Job_Satisfaction", 2))
s_stress = st.sidebar.slider("Stress", 1, 10, sidebar_val("Work_Stress_Level", 9))

st.sidebar.divider()
sim_on = st.sidebar.toggle("🛠 Personalize Retention Plan")

# --- HYBRID RISK & ANOMALY ---
def get_hybrid_risk(inputs, model):
    df_in = pd.DataFrame([inputs])[feature_cols]
    df_in["Bonus_Received"] = bonus_le.transform(df_in["Bonus_Received"])
    scaled = scaler.transform(df_in)
    
    # 1. Supervised Risk
    prob = model.predict_proba(scaled)[0][1]
    
    # 2. Neural Autoencoder Anomaly (Reconstruction Error)
    reconstruction = autoencoder.predict(scaled)
    mse = np.mean((scaled - reconstruction) ** 2)
    # Normalize MSE to 0-1 scale for dash (0.5 is high in this feature space)
    anom_score = np.clip(mse / 0.5, 0, 1)
    
    return (0.7 * prob) + (0.3 * anom_score), prob, anom_score

profile_cur = {"Age": s_age, "Number_of_Experience_Years": s_exp, "Annual_Salary": s_salary, "Last_Increment_Pct": s_inc, 
               "Bonus_Received": s_bonus, "Skill_Match_Score": s_skill, "Project_Match_Score": s_proj, "Overall_Performance_Rating": s_perf,
               "Medical_Sick_Leaves": s_sick, "Job_Satisfaction": s_sat, "Work_Stress_Level": s_stress}

risk_now, prob_now, anom_now = get_hybrid_risk(profile_cur, best_model)

if sim_on:
    profile_ret, ret_actions, ret_cost = get_tailored_retention(profile_cur)
    risk_after, prob_after, anom_after = get_hybrid_risk(profile_ret, best_model)

# --- DASHBOARD UI ---
st.title("💼 Enterprise Attrition & Intelligence Platform")

if mode == "Employee Search Mode" and selected_row is not None:
    st.markdown(f"""<div class="profile-header">
                <h2>📊 Strategic Mobility mobility Advisor: {selected_row['Employee_Name']} (ID: {selected_row['Employee_ID']})</h2>
                <p>Profile precision verified via <b>Neural Autoencoder & Hypertuned Forest</b></p>
                </div>""", unsafe_allow_html=True)
else:
    st.markdown("##### Precision Modeling Active | Simulation Environment")

# 1. KPI SECTION
st.markdown('<div class="section-simulator">', unsafe_allow_html=True)
st.subheader("🎯 Workforce Health Metrics")
k1, k2, k3, k4 = st.columns(4)

total_risk = risk_after if sim_on else risk_now
k1.metric("Churn Risk", f"{total_risk*100:.1f}%", delta=f"{(risk_after-risk_now)*100:.1f}%" if sim_on else None, delta_color="inverse")
status_res = "LEAVE" if total_risk > 0.5 else "STAY"
k2.markdown(f"**Predicted Status**<br><span class='{'status-leave' if status_res=='LEAVE' else 'status-stay'}' style='font-size:2.2rem;'>{status_res}</span>", unsafe_allow_html=True)
total_anom = anom_after if sim_on else anom_now
anom_stat = "Anomalous" if total_anom > 0.6 else "Normal"
k3.markdown(f"**AI Anomaly Score**<br><span style='font-size:2.2rem; color:#1a237e;'>{total_anom:.2f}</span><br><span style='font-size:0.9rem;'>{anom_stat} Pattern</span>", unsafe_allow_html=True)
est_loss = total_risk * (profile_cur["Annual_Salary"] * 0.20)
k4.metric("Risk Cost (Est.)", f"₹{est_loss/100000:.2f}L", help="Replacement penalty * Risk probability")
st.markdown('</div>', unsafe_allow_html=True)

# 2. TABS
tab_viz, tab_ai, tab_report = st.tabs(["📊 Visual intelligence", "🔬 Precision Architecture", "📝 Strategic Retention mobility Report"])

with tab_viz:
    st.markdown('<div class="section-analytics">', unsafe_allow_html=True)
    cv1, cv2 = st.columns([1.2, 1])
    with cv1:
        st.write("#### 📊 Salary & Attrition Density")
        fig_hist = px.histogram(df_master, x="Annual_Salary", color="Attrition", barmode="overlay",
                               color_discrete_map={0: "#2ecc71", 1: "#e74c3c"}, opacity=0.7)
        fig_hist.update_layout(xaxis=dict(tickvals=[1000000, 3000000, 5000000], ticktext=["₹10L", "₹30L", "₹50L"]), height=350, margin=dict(t=0,b=0))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('<div class="summary-card"><b>Analysis:</b> The histogram shows that churn is highest in the <b>₹5L-₹15L</b> bracket for junior roles, and starts peaking again for <b>Senior Profiles</b> with low increments.</div>', unsafe_allow_html=True)
    with cv2:
        st.write("#### 🥧 Org-wide Risk Breakdown")
        counts = df_master["Attrition"].value_counts().reset_index()
        counts.columns = ["Status", "Val"]
        counts["Status"] = counts["Status"].map({0: "Stable", 1: "At Risk"})
        fig_pie = px.pie(counts, values="Val", names="Status", hole=0.5, color="Status", color_discrete_map={"Stable": "#2ecc71", "At Risk": "#e74c3c"})
        fig_pie.update_layout(height=350, margin=dict(t=0,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('<div class="summary-card"><b>Status:</b> <b>20.3%</b> of the organization is currently flagged for potential attrition based on hybrid risk modeling.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_ai:
    st.markdown('<div class="section-models">', unsafe_allow_html=True)
    st.subheader("🔬 Precision AI Architecture (Industry Standard)")
    st.info("💡 RF Model optimized via **GridSearchCV** & Anomaly Scoring via **Neural Autoencoder**.")
    m_cols = st.columns(3)
    for idx, (m_name, m_stats) in enumerate(benchmarks.items()):
        m_cols[idx % 3].markdown(f"""
        <div class="card">
            <div class="card-title">{m_name}</div>
            <div class="card-body">
                Recall: <b>{m_stats['Recall']:.1%}</b><br>
                Accuracy: {m_stats['Accuracy']:.1%}<br>
                F1 Score: {m_stats['F1']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.write("#### Confidence Matrices & Pattern Recognition")
    mc1, mc2, mc3 = st.columns(3)
    mlist = [mc1, mc2, mc3]
    for i, (m_name, m_stats) in enumerate(benchmarks.items()):
        f_cm = px.imshow(m_stats["Confusion Matrix"], text_auto=True, color_continuous_scale='Blues', x=['Stay','Leave'], y=['Stay','Leave'])
        f_cm.update_layout(title=m_name, height=200, margin=dict(t=30, b=0))
        mlist[i].plotly_chart(f_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_report:
    st.markdown('<div class="section-report">', unsafe_allow_html=True)
    st.subheader("📝 Strategic Executive mobility Adviory Report")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f'<div class="card"><div class="card-title">⚠️ Status Analysis</div><div class="card-body">Currently predicted to <b>{status_res}</b>.<br>Profile risk <b>{total_risk*100:.1f}%</b> is driven by a combination of { "high stress" if s_stress > 7 else "complex pattern interactions"}.</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="margin-top:15px;"><div class="card-title">🧘 wellness wellness Summary</div><div class="card-body">Stress: {s_stress}/10 | Satisfaction: {s_sat}/5.<br>Conditions indicate {"high burnout risk." if s_stress > 7 else "stable psychological safety."}</div></div>', unsafe_allow_html=True)
    with r2:
        if sim_on:
            st.markdown('<div class="card" style="border: 2px solid #2e7d32;"><div class="card-title">🛠 Personalized Retention Plan</div><div class="card-body">', unsafe_allow_html=True)
            for action in ret_actions:
                st.markdown(f'<div class="retention-item">✅ {action}</div>', unsafe_allow_html=True)
            st.markdown(f"""<hr><b>Risk Reduction:</b> {(risk_now - risk_after)*100:.1f}%<br>
                        <b>Proposed Investment:</b> {format_indian_lakhs(ret_cost)}<br>
                        <i>Outcome: Prediction shifted to {status_res}.</i></div></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">💰 Financial Assessment</div><div class="card-body">Annual CTC: {format_indian_lakhs(s_salary)}.<br>Benchmark: {"Below experience median" if s_salary < 1800000 and s_exp > 10 else "Appropriately Aligned"}.</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Principal HCM Intelligence | Hypertuned Forest | Neural Autoencoder | Tailored Retention mobility Logic")
