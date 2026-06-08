import gradio as gr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. LOAD AND CLEAN DATA ---
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

# Initialize and train model globally when app starts
data = load_and_prep_data()

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

# --- 2. PREDICTION FUNCTION FOR GRADIO ---
def predict_laptop(budget_in, usage_in):
    try:
        # Process inputs
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
        
        # Format a gorgeous presentation layout for the web screen
        result_html = f"""
        <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; border: 2px solid #ffb6c1;">
            <h3 style="color: #d15c7a; margin-top: 0;">🏆 Recommended Match: {final_model}</h3>
            <hr style="border: 0; height: 1px; background: #ffb6c1; margin: 10px 0;">
            <p style="font-size: 16px; margin: 5px 0;">🧠 <b>RAM:</b> {final_ram}</p>
            <p style="font-size: 16px; margin: 5px 0;">💾 <b>Storage:</b> {final_ssd}</p>
            <p style="font-size: 16px; margin: 5px 0;">🖥️ <b>Graphics:</b> {final_gpu}</p>
            <p style="font-size: 13px; color: #777; margin-top: 15px; font-style: italic;">
                Configuration optimized for {usage_in} tasks within a {budget_in} budget profile.
            </p>
        </div>
        """
        return result_html
    except Exception as e:
        return f"<p style='color:red;'>Error processing request: {str(e)}</p>"

# --- 3. GRADIO INTERFACE LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="pink")) as demo:
    gr.Markdown("# 🤖 AI Laptop Advisor v1.0")
    gr.Markdown("Find the perfect laptop optimized for your specific budget and workload requirements.")
    
    with gr.Row():
        budget_input = gr.Dropdown(choices=["Low", "Medium", "High"], label="💰 Select Your Budget", value="Medium")
        usage_input = gr.Dropdown(choices=["Study", "Programming", "Gaming"], label="🎮 Select Your Primary Usage", value="Study")
        
    submit_btn = gr.Button("🔎 Scan Database for Best Match", variant="primary")
    
    output_html = gr.HTML(label="Match Result")
    
    # Link button to function
    submit_btn.click(fn=predict_laptop, inputs=[budget_input, usage_input], outputs=output_html)

# Launch the app
demo.launch()
