#  EV Adoption & Analysis Dashboard

An interactive, data-driven machine learning dashboard to analyze Electric Vehicle (EV) adoption patterns, perform market segmentation, and provide intelligent strategic business recommendations. Built completely with **Python** & **Streamlit**.

##  Key Features

###  Exploratory Data Analysis (EDA)
- **Dynamic KPI Tracking:** Live calculations of Total EV Sales, Average Sales per record, and Average Market Share.
- **Interactive Visualizations:** Stunning Plotly line trends (yearly/monthly), state-wise choropleths, category distributions, and correlation matrices. 
- **Extensive Filters:** Dynamically update the entire dashboard by filtering on Year, State, and Vehicle Category.

###  Machine Learning & Data Mining
- **Market Segmentation (K-Means Clustering):** Automatically groups data dynamically into **High**, **Medium**, and **Low Adoption Segments** using a 3-Dimensional K-Means clustering algorithm. 
- **Association Rule Mining:** Discovers underlying *If-Then* relationships between regions, vehicle categories, and sales volumes utilizing the **Apriori** and **FP-Growth** algorithms. Includes interactive Support, Confidence, and Lift thresholds!

###  AI Strategy Advisor
- **Context-Aware Chatbot:** Features a fully integrated AI Chatbot powered by the **Groq Llama-3 API**. 
- **Data-Driven Strategy:** The AI is implicitly fed your current dashboard's live metrics (total sales, market share) to offer highly accurate, tailored strategic advice for infrastructure, investments, and policies.

---

##  Tech Stack
- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-Learn (K-Means), MLxtend (Apriori & FP-Growth)
- **Data Visualization:** Plotly (Express, Graph Objects), Matplotlib, Seaborn
- **AI Integration:** Groq API, Requests

---

##  Running the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/Thejashree0308/EV_adoption.git
cd EV_adoption
```

### 2. Install the necessary dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Setup your API Key
The AI Strategy Advisor requires a Groq API Key to function. 
1. Create a `.streamlit` folder in the root directory.
2. Inside that folder, create a file named `secrets.toml`.
3. Add your key like this:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Run the application!
```bash
streamlit run app.py
```
Open the provided `localhost` URL in your browser to start exploring EV adoption data!

---
*Created for data mining, AI integration, and market analysis.*
Live link : https://ev-adoption.onrender.com/
