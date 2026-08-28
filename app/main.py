"""Polished Streamlit application for adaptive multi-subject learning research."""
from __future__ import annotations

import hashlib, hmac, secrets, sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from analytics.reports import dataset_statistics, performance_by_subject_topic
from config import GENERATED_DIR, MODELS_DIR, RESULTS_DIR, SimulationConfig
from data.questions.question_bank import build_question_bank
from database.db import connection, initialise_database, seed_question_bank
from experiments.experiment_1 import run_experiment_from_data
from ml.predict import predict_student_mastery
from recommendation.recommender import recommend_questions
from simulator.learning_simulator import run_simulation

NOTICE = "Synthetic records are simulated research data only — never real student-study evidence."
SUBJECTS = ["DSA", "DBMS", "Operating Systems", "Computer Networks", "Software Engineering"]

def style():
    st.markdown("""<style>
    .stApp{background:radial-gradient(circle at 8% 0,#172554 0,#0b1120 34%,#080d18 100%);color:#e5e7eb}[data-testid="stSidebar"]{background:linear-gradient(180deg,#101a32,#0a1020);border-right:1px solid #24324d}[data-testid="stMetric"]{background:#121e36;border:1px solid #253a62;border-radius:16px;padding:14px}.hero{padding:1.35rem 1.5rem;background:linear-gradient(115deg,#1d4ed8,#312e81 52%,#0f766e);border-radius:22px;border:1px solid #4f76d5;margin-bottom:1.2rem}.hero h1{margin:0;font-size:2rem}.hero p{margin:.4rem 0 0;color:#dbeafe}.insight{padding:1rem;border-radius:14px;background:#162643;border:1px solid #314a76;margin:.6rem 0}.section{color:#93c5fd;letter-spacing:.11em;font-size:.75rem;font-weight:700;text-transform:uppercase}.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:650}</style>""", unsafe_allow_html=True)

def hero(title, subtitle): st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16); digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 180000).hex(); return f"{salt}${digest}"
def password_ok(password, stored):
    if not stored or "$" not in stored: return False
    salt, digest = stored.split("$", 1); return hmac.compare_digest(password_hash(password, salt).split("$", 1)[1], digest)

@st.cache_data(show_spinner=False)
def questions(): return pd.DataFrame(build_question_bank())
@st.cache_data(show_spinner=False)
def generated_attempts():
    p = GENERATED_DIR / "attempts.csv"; return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def generated_truth():
    p = GENERATED_DIR / "synthetic_truth.csv"; return pd.read_csv(p) if p.exists() else pd.DataFrame()

def setup():
    initialise_database(); q = questions(); seed_question_bank(q.to_dict("records")); return q
def model_path():
    metric = RESULTS_DIR / "validation_model_comparison.csv"
    if metric.exists():
        p = MODELS_DIR / f"{pd.read_csv(metric).iloc[0].model}.joblib"
        if p.exists(): return p
    paths = sorted(MODELS_DIR.glob("*.joblib")); return paths[0] if paths else None
def user_attempts(student_id):
    with connection() as conn: return pd.read_sql_query("SELECT * FROM attempts WHERE student_id=? ORDER BY timestamp", conn, params=(student_id,))
def predictions(student_id, attempts, q):
    model = model_path()
    return pd.DataFrame() if model is None or attempts.empty else predict_student_mastery(model, attempts, q, student_id)
def create_user(name, password):
    with connection() as conn:
        no = conn.execute("SELECT COUNT(*) FROM students WHERE is_synthetic=0").fetchone()[0]+1; sid=f"U{no:05d}"
        conn.execute("INSERT INTO students(student_id,display_name,password_hash,is_synthetic,created_at) VALUES(?,?,?,?,?)",(sid,name.strip(),password_hash(password),0,datetime.now(timezone.utc).isoformat()))
    return sid
def log_in(sid, password):
    with connection() as conn: row=conn.execute("SELECT display_name,password_hash FROM students WHERE student_id=? AND is_synthetic=0",(sid,)).fetchone()
    if row and password_ok(password,row["password_hash"]): st.session_state.user_id=sid;st.session_state.user_name=row["display_name"];return True
    return False
def save_answer(sid, question, correct, seconds, confidence):
    with connection() as conn:
        n=conn.execute("SELECT COUNT(*) FROM attempts WHERE student_id=?",(sid,)).fetchone()[0]+1
        conn.execute("INSERT INTO attempts(student_id,question_id,subject,topic,difficulty,is_correct,time_taken_seconds,attempt_number,timestamp,session_id,confidence_rating,is_synthetic) VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",(sid,question.question_id,question.subject,question.topic,question.difficulty,int(correct),max(seconds,1),n,datetime.now(timezone.utc).isoformat(),f"{sid}_PRACTICE",confidence))
def initial_diagnostic(q, attempts):
    covered=set(attempts.subject) if not attempts.empty else set(); subject=next((x for x in SUBJECTS if x not in covered),SUBJECTS[0]); return q[(q.subject==subject)&(q.difficulty=="Easy")].sort_values("question_id").iloc[0]
def next_question(sid, attempts, q):
    covered = set(attempts.subject) if not attempts.empty else set()
    if len(covered) < len(SUBJECTS):
        return initial_diagnostic(q,attempts),"Coverage-first diagnostic: establish initial evidence in each subject before targeting a weak topic."
    p=predictions(sid,attempts,q)
    if p.empty:return initial_diagnostic(q,attempts),"Coverage-first diagnostic: build initial evidence across all subjects."
    recs=recommend_questions(sid,p,attempts,q,limit=1)
    if recs.empty:return initial_diagnostic(q,attempts),"Diagnostic revision question."
    rec=recs.iloc[0];return q[q.question_id==rec.question_id].iloc[0],rec.reason

def profile_page():
    hero("Learner profile","Secure local sign-in for personalized timed practice.")
    if st.session_state.get("user_id"):
        st.success(f"Signed in as {st.session_state.user_name} · {st.session_state.user_id}")
        if st.button("Sign out"): st.session_state.pop("user_id");st.session_state.pop("user_name");st.rerun()
        return
    left,right=st.columns(2)
    with left:
        st.markdown("<p class='section'>Sign in</p>",unsafe_allow_html=True)
        with st.form("login"): sid=st.text_input("Student ID",placeholder="U00001"); pw=st.text_input("Password",type="password"); submit=st.form_submit_button("Sign in",width="stretch")
        if submit:
            if log_in(sid.strip().upper(),pw):st.rerun()
            st.error("Invalid student ID or password.")
    with right:
        st.markdown("<p class='section'>Create account</p>",unsafe_allow_html=True)
        with st.form("register"): name=st.text_input("Display name"); pw=st.text_input("Choose password",type="password"); submit=st.form_submit_button("Create learner profile",width="stretch")
        if submit:
            if len(name.strip())<2 or len(pw)<6:st.error("Use a name and password of at least 6 characters.")
            else:
                sid=create_user(name,pw);st.session_state.user_id=sid;st.session_state.user_name=name.strip();st.rerun()

def dashboard_page(q):
    if not st.session_state.get("user_id"):st.info("Sign in from Learner Profile to access your dashboard.");return
    sid=st.session_state.user_id; attempts=user_attempts(sid);hero(f"Welcome back, {st.session_state.user_name}","Mastery estimates are calculated from your timed performance history.")
    if attempts.empty:st.markdown("<div class='insight'>Start the adaptive diagnostic. Your first answers establish a multi-subject mastery profile.</div>",unsafe_allow_html=True);return
    p=predictions(sid,attempts,q); cols=st.columns(4);cols[0].metric("Practice attempts",len(attempts));cols[1].metric("Observed accuracy",f"{attempts.is_correct.mean():.0%}");cols[2].metric("Avg. response",f"{attempts.time_taken_seconds.mean():.0f}s");cols[3].metric("Confidence",f"{attempts.confidence_rating.dropna().mean():.1f}/5" if attempts.confidence_rating.notna().any() else "—")
    if p.empty:st.warning("Train the synthetic baseline in ML Lab to enable mastery probabilities.");return
    st.subheader("Subject mastery")
    mastery=p.groupby("subject").mastery_probability.mean().reindex(SUBJECTS)
    for col,(subject,value) in zip(st.columns(5),mastery.items()):col.metric(subject.replace("Operating Systems","OS").replace("Computer Networks","CN").replace("Software Engineering","SE"),f"{value:.0%}");col.progress(float(value))
    st.subheader("Priority topics")
    for row in p.nsmallest(3,"mastery_probability").itertuples(index=False):
        reasons=row.explanation["reasons"] if isinstance(row.explanation,dict) else []
        st.markdown(f"<div class='insight'><b>{row.subject} · {row.topic}</b> — {row.mastery_probability:.0%} estimated mastery<br><span style='color:#bfdbfe'>{'; '.join(reasons)}</span></div>",unsafe_allow_html=True)
    st.caption("Model estimates begin from synthetic training data. Validate with independent real assessments before making research claims.")

def quiz_page(q):
    if not st.session_state.get("user_id"):st.info("Sign in to receive a personalized quiz.");return
    sid=st.session_state.user_id; attempts=user_attempts(sid);hero("Adaptive practice","Items are selected from the weakest predicted topic; difficulty follows recent performance.")
    if "active_question" not in st.session_state:
        item,why=next_question(sid,attempts,q);st.session_state.active_question=item.question_id;st.session_state.why=why;st.session_state.started=time.monotonic();st.session_state.answered=False
    question=q[q.question_id==st.session_state.active_question].iloc[0];st.markdown(f"<div class='insight'><b>Why this question?</b><br>{st.session_state.why}</div>",unsafe_allow_html=True);st.caption(f"{question.subject} · {question.topic} · {question.difficulty}");st.subheader(question.question)
    choices={"A":question.option_a,"B":question.option_b,"C":question.option_c,"D":question.option_d};answer=st.radio("Your answer",list(choices),format_func=lambda k:f"{k}. {choices[k]}",key=f"a_{question.question_id}");confidence=st.select_slider("Confidence",options=[1,2,3,4,5],value=3,format_func=lambda x:["Very low","Low","Moderate","High","Very high"][x-1]);elapsed=time.monotonic()-st.session_state.started;st.caption(f"Timer running · {int(elapsed)} seconds")
    if not st.session_state.answered and st.button("Submit answer",type="primary",width="stretch"):
        correct=answer==question.correct_answer;save_answer(sid,question,correct,elapsed,confidence);st.session_state.answered=True;st.success(f"Correct — {question.explanation}" if correct else f"Review — {question.explanation}");st.info(f"Saved {int(elapsed)}s and confidence {confidence}/5. Difficulty adapts after the next pattern of attempts.")
    if st.session_state.answered and st.button("Next personalized question",width="stretch"):st.session_state.pop("active_question");st.rerun()

def plan_page(q):
    if not st.session_state.get("user_id"):st.info("Sign in to generate your study plan.");return
    sid=st.session_state.user_id;attempts=user_attempts(sid);hero("Personalized study plan","Weak topics first, adaptive difficulty second, question novelty always considered.");p=predictions(sid,attempts,q)
    if p.empty:st.info("Complete a diagnostic question and train the baseline model in ML Lab.");return
    plan=recommend_questions(sid,p,attempts,q,limit=8)
    for i,row in enumerate(plan.itertuples(index=False),1):st.markdown(f"<div class='insight'><b>{i}. {row.topic}</b> <span style='color:#93c5fd'>({row.subject})</span><br>Target: <b>{row.recommended_difficulty}</b> · Estimated mastery: {row.mastery_probability:.0%}<br>{row.reason}</div>",unsafe_allow_html=True)
    st.download_button("Download study plan CSV",plan.to_csv(index=False).encode(),"personalized_study_plan.csv","text/csv")

def simulation_page():
    hero("Synthetic cohort simulator","Reproducible latent-skill profiles, response times, noise, practice effects and CSV export.");st.warning(NOTICE)
    with st.form("sim"):
        a,b,c=st.columns(3);students=a.select_slider("Students",[10,30,100,1000,10000],value=30);per=b.number_input("Attempts / student",10,500,60,10);seed=c.number_input("Random seed",0,value=42);learning=st.slider("Learning rate",.005,.10,.035,.005);ability=st.selectbox("Ability mix",["Typical","Emerging-heavy","Advanced-heavy"]);difficulty=st.selectbox("Difficulty mix",["Balanced","Foundation","Challenge"]);go=st.form_submit_button("Generate synthetic cohort",type="primary")
    if go:
        abilities={"Typical":{"emerging":.18,"developing":.38,"proficient":.30,"advanced":.14},"Emerging-heavy":{"emerging":.42,"developing":.34,"proficient":.18,"advanced":.06},"Advanced-heavy":{"emerging":.06,"developing":.18,"proficient":.36,"advanced":.40}};diffs={"Balanced":{"Easy":.4,"Medium":.4,"Hard":.2},"Foundation":{"Easy":.6,"Medium":.3,"Hard":.1},"Challenge":{"Easy":.2,"Medium":.45,"Hard":.35}}
        with st.spinner("Generating realistic synthetic histories..."):result=run_simulation(SimulationConfig(number_of_students=int(students),attempts_per_student=int(per),random_seed=int(seed),learning_rate=float(learning),ability_distribution=abilities[ability],difficulty_distribution=diffs[difficulty]))
        generated_attempts.clear();generated_truth.clear();st.success(f"Generated {len(result['attempts']):,} synthetic attempts.")
        for name,frame in [("students",result["students"]),("attempts",result["attempts"]),("synthetic_truth",result["truth"])]:st.download_button(f"Download {name}.csv",frame.to_csv(index=False).encode(),f"{name}.csv","text/csv")

def analytics_page():
    hero("Research analytics","Every number and chart is calculated from the current generated attempt dataset.");a=generated_attempts()
    if a.empty:st.info("Generate a synthetic cohort first.");return
    st.warning(NOTICE);stats=dataset_statistics(a);c=st.columns(4);c[0].metric("Students",stats["students"]);c[1].metric("Attempts",f"{stats['attempts']:,}");c[2].metric("Accuracy",f"{stats['overall_accuracy']:.1%}");c[3].metric("Mean response",f"{stats['mean_response_seconds']:.0f}s");subject,topic=performance_by_subject_topic(a);left,right=st.columns(2)
    with left:st.subheader("Accuracy by subject");st.bar_chart(subject.set_index("subject")["accuracy"]);st.subheader("Difficulty performance");st.bar_chart(a.groupby("difficulty").is_correct.mean().reindex(["Easy","Medium","Hard"]))
    with right:st.subheader("Weakest observed topics");st.dataframe(topic.sort_values("accuracy").head(12),hide_index=True,width="stretch");trend=a.assign(day=pd.to_datetime(a.timestamp).dt.date).groupby("day").is_correct.mean();st.subheader("Daily generated performance");st.line_chart(trend)
    st.download_button("Export generated attempts",a.to_csv(index=False).encode(),"generated_attempts.csv","text/csv")

def ml_page():
    hero("ML evaluation lab","Actual model fitting on generated attempt histories — never fabricated metrics.");st.warning(NOTICE);a=generated_attempts()
    if a.empty:st.info("Generate a cohort first.");return
    st.markdown("**Protocol:** history-only features → student-group split (70% train / 15% validation / 15% held-out test). Select by validation F1; report final metrics on separate test students.")
    truth=generated_truth()
    if truth.empty:
        st.error("The matching synthetic truth labels are missing. Generate the cohort again before training.");return
    if st.button("Train Logistic Regression, Random Forest & XGBoost",type="primary"):
        with st.spinner("Fitting all models on the current synthetic cohort..."):
            _,test,best=run_experiment_from_data(a,truth)
        st.success(f"Training complete on the current cohort. Validation-selected model: {best}.")
    test_path=RESULTS_DIR/"model_comparison.csv";validation=RESULTS_DIR/"validation_model_comparison.csv"
    if test_path.exists():
        if validation.exists():st.subheader("Validation model selection");st.dataframe(pd.read_csv(validation),hide_index=True,width="stretch")
        st.subheader("Held-out test metrics");st.dataframe(pd.read_csv(test_path),hide_index=True,width="stretch");roc=RESULTS_DIR/"test_roc_curves.png"
        if roc.exists():st.image(str(roc),caption="ROC curves from held-out test students")
        cols=st.columns(3)
        for col,name in zip(cols,["logistic_regression","random_forest","xgboost"]):
            image=RESULTS_DIR/f"{name}_confusion_matrix.png"
            if image.exists():col.image(str(image),caption=name.replace("_"," "))
        best=pd.read_csv(validation).iloc[0].model if validation.exists() else pd.read_csv(test_path).iloc[0].model;imp=RESULTS_DIR/f"{best}_feature_importance.csv"
        if imp.exists():st.subheader(f"Feature importance · {best}");st.bar_chart(pd.read_csv(imp).head(15).set_index("feature")["importance"])

def main():
    st.set_page_config(page_title="MasteryLab",page_icon="🎓",layout="wide",initial_sidebar_state="expanded");style();q=setup()
    with st.sidebar:st.markdown("## ◈ MasteryLab");st.caption("Adaptive Learning Research Platform");st.divider()
    pages={"Learner":[st.Page(profile_page,title="Profile & Login",icon="👤",url_path="profile"),st.Page(lambda:dashboard_page(q),title="Mastery Dashboard",icon="🎯",url_path="dashboard"),st.Page(lambda:quiz_page(q),title="Adaptive Quiz",icon="📝",url_path="adaptive-quiz"),st.Page(lambda:plan_page(q),title="Study Plan",icon="🗺️",url_path="study-plan")],"Research":[st.Page(simulation_page,title="Simulation Studio",icon="⚗️",url_path="simulation"),st.Page(analytics_page,title="Data Analytics",icon="📊",url_path="analytics"),st.Page(ml_page,title="ML Lab",icon="🧠",url_path="ml-lab")]}
    st.navigation(pages,position="sidebar").run()
if __name__=="__main__":main()
