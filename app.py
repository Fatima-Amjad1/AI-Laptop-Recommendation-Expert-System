import streamlit as st
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. SET UP THE WEB PAGE ---
st.set_page_config(page_title="AI Laptop Advisor", page_icon="💻", layout="centered")
st.title("🤖 AI Laptop Advisor v1.0")
st.write("Find the perfect laptop optimized for your specific budget and needs.")

# --- 2. LOAD AND CLEAN DATA ---
# CRITICAL: Make sure 'laptop.csv' is uploaded to your GitHub repository!
@st.cache_data
def load_and_prep_data():
    data = pd.read_csv("laptop.csv")
    
    # Clean prices
    data['Price'] = data['Price'].replace('[^0-9]', '', regex=True).astype(int)
    
    # Budget Logic
    def budget_calc(price):
        if price < 50000: return "Low"
        elif price < 100000: return "Medium"
        else: return "High"
    data['Budget'] = data['Price'].apply(budget_calc)
    
    # Drop unused columns if present
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
        
    # Usage Logic
    def usage_calc(row):
        if "RTX" in str(row['Graphics']) or "GTX" in str(row['Graphics']):
            return "Gaming"
        elif "16" in str(row['Ram']):
            return "Programming"
        else:
            return "Study"
    data['Usage'] = data.apply(usage_calc, axis=1)
    
    return data

try:
    data = load_and_prep_data()
except FileNotFoundError:
    st.error("❌ 'laptop.csv' not found. Please upload it to your GitHub repository!")
    st.stop()

# --- 3. ENCODE AND TRAIN THE MODEL ---
le_budget = LabelEncoder()
le_usage = LabelEncoder()
le_model = LabelEncoder()
le_ram = LabelEncoder()
le_ssd = LabelEncoder()
le_gpu = LabelEncoder()

data['Budget_num']   = le_budget.fit_transform(data['Budget'])
data['Usage_num']    = le_usage.fit_transform(data['Usage'])
data['Model_num']    = le_model.fit_transform(data['Model'])
data['Ram_num']      = le_ram.fit_transform(data['Ram'])
data['SSD_num']      = le_ssd.fit_transform(data['SSD'])
data['Graphics_num'] = le_gpu.fit_transform(data['Graphics'])

X = data[['Budget_num', 'Usage_num']]
y = data[['Model_num', 'Ram_num', 'SSD_num', 'Graphics_num']]

model = DecisionTreeClassifier()
model.fit(X, y)

# --- 4. USER INTERFACE (SIDEBAR / DROPDOWNS) ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    budget_in = st.selectbox("💰 Select Your Budget:", ["Low", "Medium", "High"])

with col2:
    usage_in = st.selectbox("🎮 Select Your Primary Usage:", ["Study", "Programming", "Gaming"])

# --- 5. PREDICTION AND DISPLAY LOGIC ---
if st.button("🔎 Scan Database for Best Match", type="primary", use_container_width=True):
    with st.spinner("Analyzing performance configurations..."):
        time.sleep(1) # Replicates your scanning animation delay
        
        # Process input
        b_num = le_budget.transform([budget_in])[0]
        u_num = le_usage.transform([usage_in])[0]
        
        # Predict
        test_data = pd.DataFrame([[b_num, u_num]], columns=['Budget_num', 'Usage_num'])
        prediction = model.predict(test_data)
        
        # Decode results
        final_model = le_model.inverse_transform([int(prediction[0][0])])[0]
        final_ram   = le_ram.inverse_transform([int(prediction[0][1])])[0]
        final_ssd   = le_ssd.inverse_transform([int(prediction[0][2])])[0]
        final_gpu   = le_gpu.inverse_transform([int(prediction[0][3])])[0]
        
    # Beautiful Web App Display Container
    st.balloons()
    st.success("✅ MATCH FOUND!")
    
    with st.container(border=True):
        st.subheader(f"🏆 {final_model}")
        st.markdown("---")
        st.markdown(f"🧠 **RAM:** {final_ram}")
        st.markdown(f"💾 **Storage:** {final_ssd}")
        st.markdown(f"🖥️ **Graphics:** {final_gpu}")
        
    st.caption(f"Configuration optimized natively for {usage_in} tasks within a {budget_in} budget profile.")
