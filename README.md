**AI Laptop Expert System**
 ***The Problem***
I noticed that many students struggle to choose the right laptop. With so many technical specs *(RAM, GPU, SSD)* , it’s easy to either overspend or buy a machine that can’t handle their workload. I wanted to build a tool that makes these decisions easier.

💡 ***The Solution***
**This is an AI-driven Expert System that acts as a tech consultant.**

**Input**: The user provides their Budget and intended Usage (Gaming, Office, Coding).

**Process**: A Decision Tree Classifier (Scikit-Learn) analyzes a dataset of 500+ laptops.
**Output**: The system predicts the exact Model, RAM, Storage, and GPU configuration needed.

🛠️ ***Key Technical Features***
**Data Cleaning**: Handled currency formatting and categorized raw prices into Budget tiers using Python.

**Label Encoding**: Converted categorical text data into a format the Machine Learning model could understand.

**Multi-Output Prediction**: Configured the model to predict four different hardware specifications simultaneously.

**Model Architecture**
The project uses Supervised Learning. I curated a dataset of 500+ laptops, applied Label Encoding to handle categorical data, and used Multi-output Classification to predict four hardware specs simultaneously.

**How to Run**
1-Clone this repo.
2-Open the .ipynb file in Google Colab or Jupyter.
3-Upload laptop_data.csv to the environment.
4-Run the cells!

🚀 **Tech Stack**
Language: Python -
Library: Scikit-Learn, Pandas  -
Environment: Google Colab/ Jupyter
