import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# App title
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Using Logistic Regression with real Excel data")

# Function to load data - WITH ERROR HANDLING
def load_data():
    """Load data from Excel file in the same directory"""
    try:
        # Try to load the Excel file
        df = pd.read_excel('Churn_Modelling.xlsx', sheet_name='Churn_Modelling')
        st.success("✅ Excel file loaded successfully!")
        return df
    except FileNotFoundError:
        st.error("❌ Excel file not found in the current directory.")
        st.info("""
        **Troubleshooting:**
        1. Make sure `Churn_Modelling.xlsx` is in the same folder as `app.py`
        2. On Streamlit Cloud, files must be in the GitHub repository
        3. Check the file name spelling (case-sensitive)
        """)
        
        # Generate sample data as fallback
        st.warning("⚠️ Using synthetic data for demo purposes")
        return generate_sample_data()
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data if Excel file is not available"""
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'RowNumber': range(1, n_samples + 1),
        'CustomerId': np.random.randint(100000, 999999, n_samples),
        'Surname': np.random.choice(['Smith', 'Johnson', 'Williams'], n_samples),
        'CreditScore': np.random.normal(650, 100, n_samples).astype(int),
        'Age': np.random.normal(37, 10, n_samples).astype(int),
        'Tenure': np.random.randint(0, 11, n_samples),
        'Balance': np.random.exponential(50000, n_samples),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 80],
                           labels=['18-30', '31-40', '41-50', '51-60', '61-80'])
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['IsHighValueCustomer'] = (df['Balance'] > df['Balance'].quantile(0.75)).astype(int)
    
    return df

# Load data
st.header("📊 Data Loading")
with st.spinner("Loading data from Excel file..."):
    df = load_data()

# Show data info
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Features", df.shape[1] - 4)  # Excluding target and ID columns
with col3:
    churn_rate = df['Exited'].mean() * 100
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

# Data Preparation
st.header("🔧 Data Preparation")

# Prepare features
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')

# Convert boolean columns
bool_cols = ['HasCrCard', 'IsActiveMember', 'IsHighValueCustomer']
for col in bool_cols:
    if col in X.columns:
        X[col] = X[col].astype(int)

# One-Hot Encode AgeGroup
if 'AgeGroup' in X.columns:
    X = pd.get_dummies(X, columns=['AgeGroup'], drop_first=True)

y = df['Exited']

# Show features
st.write(f"**Features prepared:** {X.shape[1]} columns")
st.dataframe(X.head(), use_container_width=True)

# Model Training
st.header("🤖 Model Training")

test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)

if st.button("🚀 Train Logistic Regression Model", type="primary"):
    with st.spinner("Training model..."):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale numeric features
        num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                   'EstimatedSalary', 'BalanceSalaryRatio']
        num_cols = [col for col in num_cols if col in X_train.columns]
        
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = model.score(X_test_scaled, y_test)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Store in session state
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = X_train.columns.tolist()
        st.session_state['trained'] = True
        st.session_state['accuracy'] = accuracy
        st.session_state['roc_auc'] = roc_auc
        st.session_state['y_test'] = y_test
        st.session_state['y_prob'] = y_prob
        
        # Show results
        st.success(f"✅ Model trained successfully!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
        with col3:
            st.metric("Churn Prediction", f"{(y_pred == 1).mean():.1%}")
        
        # Feature importance
        st.subheader("Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        st.dataframe(coef_df, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Prediction Section
st.header("🔮 Make Predictions")

if 'trained' in st.session_state:
    st.info("Enter customer details to predict churn")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            age = st.number_input("Age", 18, 80, 40)
            tenure = st.number_input("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance", 0.0, 250000.0, 50000.0, 1000.0)
        
        with col2:
            num_products = st.number_input("Number of Products", 1, 4, 2)
            has_credit_card = st.selectbox("Has Credit Card", [1, 0])
            is_active = st.selectbox("Is Active Member", [1, 0])
            salary = st.number_input("Estimated Salary", 10000.0, 200000.0, 60000.0, 1000.0)
        
        balance_ratio = balance / salary if salary > 0 else 0
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Create input DataFrame
        input_data = {}
        
        # Add all features from training
        for feature in st.session_state['feature_names']:
            if feature == 'CreditScore':
                input_data[feature] = credit_score
            elif feature == 'Age':
                input_data[feature] = age
            elif feature == 'Tenure':
                input_data[feature] = tenure
            elif feature == 'Balance':
                input_data[feature] = balance
            elif feature == 'NumOfProducts':
                input_data[feature] = num_products
            elif feature == 'HasCrCard':
                input_data[feature] = has_credit_card
            elif feature == 'IsActiveMember':
                input_data[feature] = is_active
            elif feature == 'EstimatedSalary':
                input_data[feature] = salary
            elif feature == 'BalanceSalaryRatio':
                input_data[feature] = balance_ratio
            elif 'AgeGroup' in feature:
                # Set AgeGroup based on age
                age_group = None
                if 18 <= age <= 30:
                    age_group = 'AgeGroup_31-40' if '31-40' in feature else ('AgeGroup_41-50' if '41-50' in feature else ('AgeGroup_51-60' if '51-60' in feature else 'AgeGroup_61-80'))
                elif 31 <= age <= 40:
                    age_group = 'AgeGroup_31-40'
                elif 41 <= age <= 50:
                    age_group = 'AgeGroup_41-50'
                elif 51 <= age <= 60:
                    age_group = 'AgeGroup_51-60'
                else:
                    age_group = 'AgeGroup_61-80'
                
                input_data[feature] = 1 if feature == age_group else 0
            elif feature == 'IsHighValueCustomer':
                input_data[feature] = 1 if balance > 100000 else 0
            else:
                input_data[feature] = 0  # Default value for other features
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[st.session_state['feature_names']]  # Ensure correct order
        
        # Scale features
        num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                   'EstimatedSalary', 'BalanceSalaryRatio']
        num_cols = [col for col in num_cols if col in input_df.columns]
        
        input_scaled = input_df.copy()
        input_scaled[num_cols] = st.session_state['scaler'].transform(input_df[num_cols])
        
        # Make prediction
        model = st.session_state['model']
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        
        with col2:
            if prediction == 1:
                st.error("🚨 **WILL CHURN**")
                st.write("High risk of leaving the bank")
            else:
                st.success("✅ **WILL STAY**")
                st.write("Low risk of leaving the bank")
        
        # Progress bar
        st.progress(float(prob), text=f"Risk Level: {prob:.1%}")
        
else:
    st.warning("Please train the model first using the button above.")

# About section
st.markdown("---")
st.header("📋 About This Project")

st.markdown("""
### Project Details
This application uses **Logistic Regression** to predict bank customer churn based on:
- **Credit Score, Age, Balance, Salary**
- **Account Tenure and Activity**
- **Product Usage Patterns**

### How It Works
1. **Data Loading**: Loads from `Churn_Modelling.xlsx` (or generates synthetic data)
2. **Preprocessing**: Scales features and encodes categorical variables
3. **Model Training**: Trains Logistic Regression with your data
4. **Predictions**: Interactive interface for new customer predictions

### Deployment
- **Framework**: Streamlit
- **Hosting**: Streamlit Cloud
- **Repository**: GitHub with version control

### Deployment
This project is a complete bank customer churn prediction system built with Logistic Regression and deployed as an interactive web application. The app begins by loading real customer data and preparing it for analysis through scaling and encoding techniques to ensure the model works effectively with the different types of information.

The machine learning component trains a Logistic Regression model that shows strong performance in identifying which customers are likely to leave. The model demonstrates excellent predictive capability through its ROC-AUC [ROC (Receiver Operating Characteristic) and AUC (Area Under Curve) is basically how good it is at carrying out the prediction] score, meaning it's very good at distinguishing between customers who will stay versus those who will leave, while analysis reveals that financial patterns and customer age are the most influential factors in predicting churn.

Users can then interactively test the model through a simple form interface where they enter customer details like credit score and account information. The system instantly calculates churn probability and presents clear, color-coded risk assessments that allow business users to quickly understand which customers need attention, with visual indicators showing the likelihood of churn for each case.

*Note: This is a portfolio project demonstrating end-to-end ML deployment.*
""")

# Footer
st.markdown("---")
st.caption("👨‍💻 Bank Churn Prediction Analytics | Built with Streamlit")