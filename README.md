# AI Laptop Expert System 

###  The Problem
I noticed that many students struggle to choose the right laptop. With so many technical specs (*RAM, GPU, SSD*), it’s easy to either overspend or buy a machine that can’t handle their workload. I wanted to build a tool that makes these decisions easier.

### The Solution
**This is an AI-driven Expert System that acts as a tech consultant.**

* **Input:** The user selects their Budget and intended Usage from an intuitive web interface.
* **Process:** A Decision Tree Classifier (Scikit-Learn) analyzes a dataset of 1,000+ laptops.
* **Output:** The system dynamically predicts the exact Model, RAM, Storage, and GPU configuration needed.

---

### Key Technical Features
* **Data Cleaning:** Handled messy currency strings, removed special characters, and dynamically categorized raw prices into numerical Budget tiers using Pandas.
* **Label Encoding:** Applied multi-variable Label Encoding to convert all categorical text data into numerical inputs natively understood by the machine learning algorithm.
* **Multi-Output Prediction:** Engineered the system using Scikit-Learn's multi-output classification capabilities to predict four distinct hardware features simultaneously from a single input array.

---

### Model Architecture
The project leverages **Supervised Machine Learning**. The underlying pipeline processes a curated laptop dataset, encodes high-cardinality string features into discrete mathematical values, and maps them through a **Decision Tree Classifier** optimized to split nodes based on standard information gain metrics (Gini Impurity). 

---

### Tech Stack & Deployment
* **Core Language:** Python
* **Data & ML Libraries:** Scikit-Learn, Pandas
* **Web UI Framework:** Streamlit
* **Live Deployment Platform:** Hugging Face Spaces / GitHub

---

### How It Runs on the Web
This repository is configured to deploy directly to the cloud. The platform automatically scans `requirements.txt` to spin up a standalone environment, executes `app.py` via an active Python process, and serves the app over HTTPS.

#### Local Development:
If you want to run this web app locally on your machine instead of the cloud:
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the web server: `streamlit run app.py`
