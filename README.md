# 📊 Web Table Extraction & Cleaning Tool (v1.0)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://viliwebtotablev1.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)

## 🚀 Overview
**WebToTable** is an automated, end-to-end Data Engineering (ETL) pipeline designed to seamlessly extract, clean, analyze, and export tabular data from any web page. It transforms the tedious process of manual web scraping into a rapid, zero-setup workflow accessible directly from your browser without writing a single line of code.

## 💡 The Problem It Solves
Data analysts spend an estimated 80% of their time finding and cleaning data. Web scraping typically involves dealing with raw HTML, handling connection timeouts, and writing repetitive Pandas data-cleaning logic for every new dataset. WebToTable automates this entire pipeline, allowing professionals to focus on data analysis rather than data extraction.

## 🏗️ The Development Journey (How it was built)
This project was engineered through a rigorous, three-phase evolutionary process to ensure structural integrity and scalability:
1. **The Static Prototype:** The engine originated as a hardcoded Python script to test the core logic of web scraping and DataFrame manipulation without user interaction.
2. **The Dynamic CLI Engine:** The logic was then upgraded into a dynamic command-line interface. By utilizing standard `input()` and `print()` functions, the engine allowed users to manually pass URLs, specify table indices, and trigger cleaning operations interactively via the terminal.
3. **The Interactive Web App:** In the final architectural shift, the engine was completely migrated from terminal standard I/O to a reactive graphical user interface. By replacing terminal commands with Streamlit components, the script was transformed into a globally accessible web application.

## ✨ Core Features & Cleaning Mechanics
Once the backend HTTP requests locate the standard HTML `<table>` tags, the extracted data is passed through a dynamic cleaning pipeline. 

Users can opt for **Strict Cleaning**, which drops any row containing at least one missing value (NaN) to ensure 100% complete records. Alternatively, the engine offers **Intelligent Imputation**, which segregates columns by data type—filling missing numeric data with the mathematical mean or zeros, and replacing missing text with null values. Finally, the application generates a comprehensive real-time statistical summary (Data Profiling) including structural data types and descriptive statistics.

## 💾 Universal Export Capabilities
The processed DataFrame can be exported directly from the browser into multiple production-ready formats:
*   **CSV & Excel:** Standard formats for immediate use in BI tools like Tableau and Power BI.
*   **JSON & Dict:** Key-value structured formats, ideal for migrating data directly into NoSQL databases.
*   **HDF5 & LaTeX:** High-performance formats designed specifically for storing massive amounts of numerical data and academic reporting.

## ⚠️ Technical Limitations
While highly efficient for standard tabular data, the extraction mechanism relies strictly on scanning HTML `<table>` tags. It cannot scrape data structured inside nested `<div>` grids. Additionally, because the application uses static HTTP requests, websites that load their tabular data dynamically via client-side JavaScript (e.g., React, Angular) will return empty results.

## 💻 Tech Stack
*   **Core Engine & Logic:** Python
*   **Data Manipulation:** Pandas
*   **Web Scraping & Networking:** Requests, BeautifulSoup
*   **Frontend & Deployment:** Streamlit

## 🌐 Live Demo
Experience the tool live here: **[WebToTable Professional Engine](https://viliwebtotablev1.streamlit.app/)**

### 🧪 Test it yourself:
1. Copy this URL: `https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population`
2. Paste it into the **Target Web URL** field in the sidebar.
3. Select Table Number `1`.
4. Check **Enable Data Analysis Report**.
5. Click **Extract and Process Table** and watch the engine work in seconds.

## 👨‍💻 Developer
Developed by **Tariq Zeyad Jawad (Vili)** 
Physicist & Data Analyst merging analytical logic with data engineering.
 
