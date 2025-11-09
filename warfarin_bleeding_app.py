import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import sys

# Check Python version
st.sidebar.markdown(f"**Python Version:** {sys.version}")

# Set page configuration
st.set_page_config(
    page_title="Warfarin Bleeding Risk Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and encoders"""
    try:
        model = joblib.load('saved_model/warfarin_bleeding_rf_model.pkl')
        label_encoders = joblib.load('saved_model/label_encoders.pkl')
        feature_columns = joblib.load('saved_model/feature_columns.pkl')
        st.sidebar.success("✅ Model loaded successfully")
        return model, label_encoders, feature_columns
    except FileNotFoundError as e:
        st.sidebar.error(f"❌ Model files not found: {e}")
        st.sidebar.info("Please ensure the model is trained and saved in the 'saved_model' directory")
        return None, None, None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None, None, None

def predict_bleeding_risk(patient_data, model, label_encoders, feature_columns):
    """Predict bleeding risk for a new patient"""
    try:
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Encode categorical variables
        for col in ['sex', 'cyp2c9_genotype', 'vkorc1_genotype']:
            if col in patient_df.columns and col in label_encoders:
                patient_df[col] = label_encoders[col].transform([patient_data[col]])[0]
        
        # Ensure all features are present and in correct order
        for feature in feature_columns:
            if feature not in patient_df.columns:
                patient_df[feature] = 0
        
        patient_features = patient_df[feature_columns]
        
        # Make prediction
        probability = model.predict_proba(patient_features)[0, 1]
        prediction = model.predict(patient_features)[0]
        
        # Determine risk category
        if probability > 0.3:
            risk_category = "High"
            recommendation = "🚨 High Risk - Consider dose adjustment, frequent monitoring, or alternative therapy"
        elif probability > 0.15:
            risk_category = "Medium"
            recommendation = "⚠️ Medium Risk - Increased monitoring recommended"
        else:
            risk_category = "Low"
            recommendation = "✅ Low Risk - Standard monitoring appropriate"
        
        return {
            'bleeding_probability': probability,
            'risk_category': risk_category,
            'prediction': 'Major Bleeding' if prediction == 1 else 'No Major Bleeding',
            'recommendation': recommendation,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_risk_gauge(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Bleeding Risk Score", 'font': {'size': 24}},
        delta = {'reference': 20, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 15], 'color': 'green'},
                {'range': [15, 30], 'color': 'yellow'},
                {'range': [30, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">💊 Warfarin Bleeding Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("### Egyptian Population Pharmacogenomics Model")
    
    # Sidebar info
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This tool predicts the risk of major bleeding in Egyptian patients 
    taking warfarin using clinical and genetic factors.
    
    **Features used:**
    - Demographic data
    - Clinical history
    - Pharmacogenomics (CYP2C9, VKORC1)
    - Treatment metrics
    """)
    
    # Load model
    model, label_encoders, feature_columns = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ **Model not loaded** 
        
        Please ensure you have:
        1. Trained the model using the training script
        2. Saved the model files in the 'saved_model' directory
        3. Files required:
           - warfarin_bleeding_rf_model.pkl
           - label_encoders.pkl  
           - feature_columns.pkl
        """)
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        # Create form for patient data
        with st.form("patient_form"):
            # Demographics
            st.markdown("#### Demographics")
            age = st.slider("Age", 20, 95, 65)
            sex = st.selectbox("Sex", ["M", "F"])
            
            # Clinical Features
            st.markdown("#### Clinical History")
            previous_bleeding = st.selectbox("Previous Bleeding History", [0, 1], 
                                           format_func=lambda x: "Yes" if x == 1 else "No")
            hypertension = st.selectbox("Hypertension", [0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No")
            renal_disease = st.selectbox("Renal Disease", [0, 1], 
                                       format_func=lambda x: "Yes" if x == 1 else "No")
            liver_disease = st.selectbox("Liver Disease", [0, 1], 
                                       format_func=lambda x: "Yes" if x == 1 else "No")
            
            # Medications
            st.markdown("#### Medications")
            amiodarone_use = st.selectbox("Amiodarone Use", [0, 1], 
                                        format_func=lambda x: "Yes" if x == 1 else "No")
            antiplatelet_use = st.selectbox("Antiplatelet Use (Aspirin/Clopidogrel)", [0, 1], 
                                          format_func=lambda x: "Yes" if x == 1 else "No")
            
            # Pharmacogenomics
            st.markdown("#### Pharmacogenomics")
            cyp2c9_genotype = st.selectbox("CYP2C9 Genotype", 
                                         ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"])
            vkorc1_genotype = st.selectbox("VKORC1 Genotype", ["GG", "AG", "AA"])
            
            # Warfarin Treatment
            st.markdown("#### Warfarin Treatment")
            most_recent_inr = st.slider("Most Recent INR", 1.0, 5.0, 2.5, 0.1)
            warfarin_dose_mg_day = st.slider("Warfarin Dose (mg/day)", 0.5, 10.0, 4.5, 0.1)
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Bleeding Risk")
    
    with col2:
        st.markdown("### 📊 Risk Assessment")
        
        if submitted:
            # Prepare patient data
            patient_data = {
                'age': age,
                'sex': sex,
                'previous_bleeding': previous_bleeding,
                'hypertension': hypertension,
                'renal_disease': renal_disease,
                'liver_disease': liver_disease,
                'amiodarone_use': amiodarone_use,
                'antiplatelet_use': antiplatelet_use,
                'cyp2c9_genotype': cyp2c9_genotype,
                'vkorc1_genotype': vkorc1_genotype,
                'most_recent_inr': most_recent_inr,
                'warfarin_dose_mg_day': warfarin_dose_mg_day
            }
            
            # Make prediction
            with st.spinner("Calculating bleeding risk..."):
                result = predict_bleeding_risk(patient_data, model, label_encoders, feature_columns)
            
            if not result['success']:
                st.error(f"❌ Prediction failed: {result['error']}")
            else:
                # Display results
                probability = result['bleeding_probability']
                risk_category = result['risk_category']
                
                # Risk Gauge
                st.plotly_chart(create_risk_gauge(probability), use_container_width=True)
                
                # Risk Category
                st.markdown(f"### Risk Category: {risk_category}")
                
                if risk_category == "High":
                    st.markdown('<div class="risk-high">🚨 HIGH BLEEDING RISK</div>', unsafe_allow_html=True)
                elif risk_category == "Medium":
                    st.markdown('<div class="risk-medium">⚠️ MEDIUM BLEEDING RISK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low">✅ LOW BLEEDING RISK</div>', unsafe_allow_html=True)
                
                # Probability
                st.metric("Bleeding Probability", f"{probability:.1%}")
                
                # Recommendation
                st.markdown("### 💡 Clinical Recommendation")
                st.info(result['recommendation'])
                
                # Key Risk Factors
                st.markdown("### 🔍 Key Risk Factors")
                top_factors = []
                if previous_bleeding:
                    top_factors.append("📌 Previous bleeding history")
                if liver_disease:
                    top_factors.append("📌 Liver disease")
                if renal_disease:
                    top_factors.append("📌 Renal disease")
                if amiodarone_use:
                    top_factors.append("📌 Amiodarone use")
                if antiplatelet_use:
                    top_factors.append("📌 Antiplatelet medication use")
                if most_recent_inr > 3.0:
                    top_factors.append(f"📌 High INR ({most_recent_inr})")
                if cyp2c9_genotype in ["*1/*3", "*2/*2", "*2/*3", "*3/*3"]:
                    top_factors.append(f"📌 CYP2C9 {cyp2c9_genotype} (poor metabolizer)")
                if vkorc1_genotype in ["AG", "AA"]:
                    top_factors.append(f"📌 VKORC1 {vkorc1_genotype} (high sensitivity)")
                
                if top_factors:
                    for factor in top_factors[:6]:
                        st.write(factor)
                else:
                    st.write("✅ No major risk factors identified")
        
        else:
            # Default view before submission
            st.info("👆 Please fill out the patient information form and click 'Predict Bleeding Risk' to see the assessment.")
            
            # Display model info
            st.markdown("### ℹ️ Model Information")
            st.write(f"**Model Type:** Random Forest Classifier")
            st.write(f"**Number of Features:** {len(feature_columns)}")
            st.write(f"**Trained on:** Egyptian patient population")
            
            # Feature importance (if available)
            try:
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                st.markdown("### 📈 Feature Importance")
                fig = px.bar(feature_importance, x='importance', y='feature', 
                            orientation='h', title='Feature Importance in Prediction')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not display feature importance chart")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This tool is for educational and research purposes only. 
    Clinical decisions should be made by qualified healthcare professionals.
    
    **Technical Info**: Built with Streamlit, Scikit-learn, and Plotly
    """)

if __name__ == "__main__":
    main()
