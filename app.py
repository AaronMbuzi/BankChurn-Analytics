import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .success-text {
        color: #10B981;
        font-weight: bold;
    }
    .danger-text {
        color: #EF4444;
        font-weight: bold;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🏦 Bank Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses **Logistic Regression** to predict whether a bank customer will churn (leave the bank).
Upload your data or use the sample data to train the model and make predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Model Training", "🔮 Make Predictions", "📈 Model Performance", "📋 About"])

# Function to load and preprocess data (matches your code)
def load_and_preprocess_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            # Try reading the uploaded file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Please upload an Excel (.xlsx) or CSV (.csv) file")
                return None, None, None, None, None
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None, None, None, None, None
    else:
        # Use the sample data path from your code
        try:
            df = pd.read_excel('Churn_Modelling.xlsx', sheet_name='Churn_Modelling', header=0)
        except:
            st.warning("Sample file not found. Please upload your data file.")
            return None, None, None, None, None
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Prepare features and target
    if 'Exited' not in df.columns:
        st.error("Column 'Exited' not found in the dataset. This column should contain the target variable.")
        return None, None, None, None, None
    
    X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')
    y = df['Exited']
    
    # Convert boolean columns to int
    bool_cols = ['HasCrCard', 'IsActiveMember', 'IsHighValueCustomer']
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    # One-Hot Encode AgeGroup
    categorical_cols = ['AgeGroup']
    for col in categorical_cols:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    return df, X, y

# Function to train model
def train_model(X_train, X_test, y_train, y_test):
    # Scale numeric features
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'EstimatedSalary', 'BalanceSalaryRatio']
    
    # Filter to columns that exist
    num_cols = [col for col in num_cols if col in X_train.columns]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # Train Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:,1]
    
    return lr, scaler, y_pred, y_prob, X_train_scaled.columns

# Page 1: Model Training
if page == "📊 Model Training":
    st.markdown('<h2 class="sub-header">📊 Upload & Train Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your Churn_Modelling.xlsx file", type=['xlsx', 'csv'])
    
    with col2:
        test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
    
    if uploaded_file is not None or st.button("Use Sample Data Structure"):
        with st.spinner("Loading and preprocessing data..."):
            df, X, y = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            # Show dataset info
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", X.shape[1])
            with col3:
                churn_rate = (y.sum() / len(y)) * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            # Show data preview
            with st.expander("View Data Preview", expanded=True):
                tab1, tab2 = st.tabs(["First 10 Rows", "Data Info"])
                with tab1:
                    st.dataframe(df.head(10), use_container_width=True)
                with tab2:
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model button
            if st.button("🚀 Train Logistic Regression Model", use_container_width=True):
                with st.spinner("Training model..."):
                    lr, scaler, y_pred, y_prob, feature_names = train_model(X_train, X_test, y_train, y_test)
                    
                    # Calculate metrics
                    roc_score = roc_auc_score(y_test, y_prob)
                    cm = confusion_matrix(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Store in session state
                    st.session_state['model'] = lr
                    st.session_state['scaler'] = scaler
                    st.session_state['feature_names'] = feature_names
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = y_pred
                    st.session_state['y_prob'] = y_prob
                    st.session_state['roc_score'] = roc_score
                    st.session_state['cm'] = cm
                    st.session_state['report'] = report
                    
                    # Success message
                    st.success("✅ Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("ROC-AUC Score", f"{roc_score:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        accuracy = report['accuracy']
                        st.metric("Accuracy", f"{accuracy:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        precision = report['1']['precision']
                        st.metric("Precision (Churn)", f"{precision:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Make Predictions
elif page == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">🔮 Predict Customer Churn</h2>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first on the 'Model Training' page.")
    else:
        st.info("Enter customer details to predict churn probability")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            Age = st.number_input("Age", min_value=18, max_value=100, value=40)
            Tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
            Balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0)
        
        with col2:
            NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
            HasCrCard = st.selectbox("Has Credit Card", [1, 0])
            IsActiveMember = st.selectbox("Is Active Member", [1, 0])
            EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0)
        
        BalanceSalaryRatio = st.number_input("Balance/Salary Ratio", 
                                            value=Balance/EstimatedSalary if EstimatedSalary > 0 else 0,
                                            format="%.3f")
        
        # Add categorical inputs if they exist in the trained model
        AgeGroup_dummies = []
        if 'feature_names' in st.session_state:
            for feature in st.session_state['feature_names']:
                if 'AgeGroup' in feature:
                    AgeGroup_dummies.append(feature)
        
        if AgeGroup_dummies:
            AgeGroup = st.selectbox("Age Group", AgeGroup_dummies)
        
        if st.button("Predict Churn", use_container_width=True):
            # Create input array
            input_data = {
                'CreditScore': CreditScore,
                'Age': Age,
                'Tenure': Tenure,
                'Balance': Balance,
                'NumOfProducts': NumOfProducts,
                'HasCrCard': HasCrCard,
                'IsActiveMember': IsActiveMember,
                'EstimatedSalary': EstimatedSalary,
                'BalanceSalaryRatio': BalanceSalaryRatio
            }
            
            # Add AgeGroup dummies
            for dummy in AgeGroup_dummies:
                input_data[dummy] = 1 if dummy == AgeGroup else 0
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns match training data
            for col in st.session_state['feature_names']:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns
            input_df = input_df[st.session_state['feature_names']]
            
            # Scale features
            input_scaled = input_df.copy()
            num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                       'EstimatedSalary', 'BalanceSalaryRatio']
            num_cols = [col for col in num_cols if col in input_df.columns]
            input_scaled[num_cols] = st.session_state['scaler'].transform(input_df[num_cols])
            
            # Make prediction
            model = st.session_state['model']
            prob = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Churn Probability", f"{prob:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown('<p class="danger-text">Prediction: WILL CHURN 🚨</p>', unsafe_allow_html=True)
                    st.write("Customer is likely to leave the bank.")
                else:
                    st.markdown('<p class="success-text">Prediction: WILL STAY ✅</p>', unsafe_allow_html=True)
                    st.write("Customer is likely to stay with the bank.")
                st.markdown('</div>', unsafe_allow_html=True)

# Page 3: Model Performance
elif page == "📈 Model Performance":
    st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first on the 'Model Training' page.")
    else:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 ROC Curve", "📈 Confusion Matrix", "📋 Classification Report"])
        
        with tab1:
            st.subheader("ROC Curve")
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(st.session_state['y_test'], st.session_state['y_prob'])
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'Logistic Regression (AUC = {st.session_state["roc_score"]:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.2)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Confusion Matrix")
            
            # Plot confusion matrix
            cm = st.session_state['cm']
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Stay (0)', 'Churn (1)'],
                       yticklabels=['Stay (0)', 'Churn (1)'])
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14)
            
            st.pyplot(fig)
            
            # Metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Positives", tp)
            with col2:
                st.metric("False Positives", fp)
            with col3:
                st.metric("False Negatives", fn)
            with col4:
                st.metric("True Negatives", tn)
        
        with tab3:
            st.subheader("Classification Report")
            
            report = st.session_state['report']
            report_df = pd.DataFrame(report).transpose()
            
            # Display as a styled table
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance (Coefficients)")
            coef_df = pd.DataFrame({
                'Feature': st.session_state['feature_names'],
                'Coefficient': st.session_state['model'].coef_[0]
            }).sort_values(by='Coefficient', key=abs, ascending=False)
            
            st.dataframe(coef_df, use_container_width=True)

# Page 4: About
else:
    st.markdown('<h2 class="sub-header">📋 About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    This application predicts customer churn for a bank using **Logistic Regression**. 
    Churn prediction helps banks identify customers likely to leave so they can take proactive retention measures.
    
    ### 📊 Model Details
    - **Algorithm**: Logistic Regression
    - **Features Used**: Credit Score, Age, Tenure, Balance, Number of Products, etc.
    - **Target Variable**: Exited (1 = Churned, 0 = Stayed)
    - **Evaluation Metric**: ROC-AUC Score
    
    ### 🔧 Technical Implementation
    - **Data Preprocessing**: Standard Scaling, One-Hot Encoding
    - **Model Training**: Scikit-learn Logistic Regression
    - **Web Framework**: Streamlit
    - **Visualization**: Matplotlib, Seaborn
    
    ### 📁 Dataset Structure
    The model expects data with these key columns:
    - `CreditScore`, `Age`, `Balance`, `EstimatedSalary`
    - `HasCrCard`, `IsActiveMember` (as 0/1)
    - `AgeGroup` (categorical, one-hot encoded)
    - `Exited` (target variable)
    
    ### 🚀 How to Use
    1. Go to **Model Training** page
    2. Upload your dataset or use sample structure
    3. Train the model
    4. Make predictions on new data
    5. Evaluate model performance
    
    ### 👨‍💻 Developer
    Created as a portfolio project showcasing machine learning deployment skills.
    """)
    
    # Add download link for sample data structure
    st.markdown("---")
    st.subheader("Sample Data Structure")
    
    # Create sample data
    sample_data = {
        'CreditScore': [650, 720, 580],
        'Age': [40, 35, 50],
        'Balance': [50000, 75000, 25000],
        'EstimatedSalary': [60000, 80000, 40000],
        'Exited': [0, 1, 0]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)