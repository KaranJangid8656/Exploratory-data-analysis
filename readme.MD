🚀 Interactive EDA Dashboard

An interactive, visually rich Exploratory Data Analysis (EDA) dashboard built with Streamlit. Upload .csv or .xlsx datasets and instantly explore your data through automatic summaries, distribution plots, and correlation heatmaps — all presented in a modern dark-themed UI.

🌟 Features

📁 Upload .csv or .xlsx datasets

📊 Dataset overview: row/column count, missing values

🗾 Column data types & memory usage display

📈 Summary statistics: mean, median, std deviation, etc.

📉 Distribution plots: histograms, box plots

🔥 Correlation heatmap & scatter plot matrix

🌐 Missing values heatmap

🌑 Sleek dark mode theme with customized UI

🧠 Sidebar navigation to switch between analysis views

📄 Export selected visuals or summary reports (coming soon)

🛠 Tech Stack

Tool

Purpose

Streamlit

Interactive web app framework

Pandas

Data manipulation & analysis

Plotly

Interactive and dynamic visualizations

Seaborn

Statistical data visualization

Matplotlib

General plotting support

OpenPyXL

Excel file reading (.xlsx support)

📦 Installation

Clone the repository

git clone https://github.com/your-username/interactive-eda-dashboard.git
cd interactive-eda-dashboard

Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/macOS

Install the required dependencies

pip install -r requirements.txt

Run the Streamlit app

streamlit run app.py

📂 Project Structure

interactive-eda-dashboard/
├── app.py                  # Main Streamlit application
├── utils/
│   └── eda_utils.py        # Reusable EDA functions
├── assets/                 # Custom themes, logos, icons
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation

📸 Screenshots

Add screenshots or a demo GIF here to showcase the dashboard in action.

🤝 Contributing

Contributions are welcome! Here's how you can help:

Report bugs or request features via Issues

Fork the repo and submit a Pull Request

Suggest UI/UX improvements

📃 License

This project is licensed under the MIT License — see the LICENSE file for details.

