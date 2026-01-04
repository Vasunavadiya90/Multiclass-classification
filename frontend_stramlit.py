import streamlit as st
import pandas as pd
import numpy as np
from src.pipelines.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .stTitle {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in-out;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #a0aec0;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    /* Card Styles */
    .stForm {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 1s ease-in-out;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #667eea;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Input Fields */
    .stNumberInput, .stTextInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        font-family: 'Outfit', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: scaleIn 0.5s ease-in-out;
    }
    
    .result-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-score {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .result-description {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 1rem;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Column gap */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='stTitle'>💳 Credit Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced AI-powered credit risk assessment using machine learning</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About")
    st.markdown("""
    This application uses advanced machine learning algorithms to predict credit scores based on financial and credit-related features.
    
    **Prediction Categories:**
    - 🔴 **Poor** (0): High Risk
    - 🟡 **Standard** (1): Moderate Risk
    - 🟢 **Good** (2): Low Risk
    """)
    
    st.markdown("### 📈 How it Works")
    st.markdown("""
    1. Fill in the financial details
    2. Submit for analysis
    3. Get instant credit score prediction
    4. View detailed risk assessment
    """)
    
    st.markdown("### 💡 Tips")
    st.info("Ensure all information is accurate for the best prediction results.")

# Main form
with st.form("prediction_form"):
    st.markdown("<div class='section-header'>📋 Personal Information</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=26, help="Your current age")
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=81817.0, step=1000.0, help="Your total annual income")
    
    with col2:
        monthly_inhand_salary = st.number_input("Monthly In-hand Salary ($)", min_value=0.0, value=80471.0, step=100.0, help="Your monthly take-home salary")
        amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=80.587, step=10.0, help="Average monthly investment amount")
    
    st.markdown("<div class='section-header'>🏦 Banking Information</div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    
    with col3:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=5, help="Total number of bank accounts")
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=5, help="Total number of credit cards")
    
    with col4:
        num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=20, value=4, help="Total number of active loans")
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=455.0, step=50.0, help="Average monthly account balance")
    
    with col5:
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=809.88, step=50.0, help="Total outstanding debt")
        total_emi_per_month = st.number_input("Total EMI per Month ($)", min_value=0.0, value=49.575854, step=10.0, help="Total monthly EMI payments")
    
    st.markdown("<div class='section-header'>💳 Credit Information</div>", unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    
    with col6:
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=24.366, step=0.1, help="Percentage of credit limit used")
        credit_history_age = st.number_input("Credit History Age (months)", min_value=0, max_value=500, value=258, help="Age of your credit history in months")
    
    with col7:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1, help="Average interest rate on loans")
        changed_credit_limit = st.number_input("Changed Credit Limit ($)", min_value=0.0, value=11.37, step=1.0, help="Recent changes to credit limit")
    
    with col8:
        credit_mix = st.selectbox("Credit Mix", ["Poor", "Standard", "Good"], index=2, help="Quality of your credit mix")
        payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"], index=1, help="Do you pay minimum amount only?")
    
    st.markdown("<div class='section-header'>⚠️ Payment Behavior</div>", unsafe_allow_html=True)
    col9, col10, col11 = st.columns(3)
    
    with col9:
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, max_value=100, value=12, help="Average days delayed from due date")
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, max_value=50, value=7, help="Total number of delayed payments")
    
    with col10:
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=50, value=2, help="Recent credit inquiries")
        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "High_spent_Small_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Small_value_payments"
            ],
            index=2,
            help="Your typical payment behavior pattern"
        )
    
    with col11:
        st.write("")  # Spacer
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🔮 Predict Credit Score")

# Handle prediction
if submit_button:
    with st.spinner("🔄 Analyzing your financial data..."):
        try:
            # Create custom data object
            custom_data = CustomData(
                Delay_from_due_date=delay_from_due_date,
                Num_of_Delayed_Payment=num_of_delayed_payment,
                Num_Credit_Inquiries=num_credit_inquiries,
                Credit_Utilization_Ratio=credit_utilization_ratio,
                Credit_History_Age=credit_history_age,
                Payment_of_Min_Amount=payment_of_min_amount,
                Amount_invested_monthly=amount_invested_monthly,
                Monthly_Balance=monthly_balance,
                Credit_Mix=credit_mix,
                Payment_Behaviour=payment_behaviour,
                Age=age,
                Annual_Income=annual_income,
                Num_Bank_Accounts=num_bank_accounts,
                Num_Credit_Card=num_credit_card,
                Interest_Rate=interest_rate,
                Num_of_Loan=num_of_loan,
                Monthly_Inhand_Salary=monthly_inhand_salary,
                Changed_Credit_Limit=changed_credit_limit,
                Outstanding_Debt=outstanding_debt,
                Total_EMI_per_month=total_emi_per_month
            )
            
            # Convert to dataframe
            pred_df = custom_data.to_df()
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            prediction = int(results[0])
            
            logging.info(f"Prediction completed: {prediction}")
            
            # Display results with beautiful styling
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Define score mappings
            score_mapping = {
                0: {"label": "Poor", "emoji": "🔴", "color": "#ef4444", "description": "High Risk - Significant improvement needed"},
                1: {"label": "Standard", "emoji": "🟡", "color": "#f59e0b", "description": "Moderate Risk - Room for improvement"},
                2: {"label": "Good", "emoji": "🟢", "color": "#10b981", "description": "Low Risk - Excellent credit profile"}
            }
            
            score_info = score_mapping[prediction]
            
            # Create result card
            st.markdown(f"""
            <div class='result-card' style='background: linear-gradient(135deg, {score_info['color']} 0%, {score_info['color']}dd 100%);'>
                <div class='result-title'>Your Predicted Credit Score</div>
                <div class='result-score'>{score_info['emoji']} {score_info['label']}</div>
                <div class='result-description'>{score_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create gauge chart for visualization
            st.markdown("<br>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit Score Level", 'font': {'size': 24, 'family': 'Outfit'}},
                gauge={
                    'axis': {'range': [None, 2], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': score_info['color']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 1], 'color': '#fecaca'},
                        {'range': [1, 2], 'color': '#fed7aa'},
                        {'range': [2, 3], 'color': '#bbf7d0'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'family': "Inter"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if prediction == 0:  # Poor
                st.error("""
                **Immediate Actions Required:**
                - Reduce credit utilization below 30%
                - Pay all bills on time for the next 6 months
                - Avoid new credit inquiries
                - Consider debt consolidation options
                - Set up automatic payments to avoid delays
                """)
            elif prediction == 1:  # Standard
                st.warning("""
                **Improvement Suggestions:**
                - Keep credit utilization under 30%
                - Continue paying bills on time
                - Maintain a diverse credit mix
                - Reduce outstanding debt gradually
                - Avoid unnecessary credit inquiries
                """)
            else:  # Good
                st.success("""
                **Maintain Your Excellent Score:**
                - Continue current payment habits
                - Keep credit utilization low
                - Monitor your credit report regularly
                - Maintain your diverse credit mix
                - Consider credit limit increases responsibly
                """)
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            logging.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; opacity: 0.7; font-size: 0.9rem;'>
    <p>Built with ❤️ using Streamlit | Credit Score Prediction Model v1.0</p>
    <p>© 2026 Vasu Navadiya | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
