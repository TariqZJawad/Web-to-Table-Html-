import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import io

# Page Configuration & Professional Theme Setup
st.set_page_config(
    page_title="WebToTable Pro | Advanced Data Scraper",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    .social-icons {
        display: flex;
        gap: 15px;
        font-size: 24px;
        margin-top: 10px;
    }
    .social-icons a {
        color: #58a6ff;
        text-decoration: none;
    }
    .social-icons a:hover {
        color: #79c0ff;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State to prevent resetting on download/clicks
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar Control Panel
st.sidebar.header("Configuration Panel")

url = st.sidebar.text_input(
    "Target Web URL", 
    value="",
    placeholder="Paste your URL here..."
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

table_num = st.sidebar.number_input("Table Number (1-based index)", min_value=1, value=1, step=1)
in_table = table_num - 1

clean_option = st.sidebar.selectbox(
    "Cleaning Mode",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "1: Drop rows with NaN",
        2: "2: Fill NaN with None",
        3: "3: Fill NaN with 0",
        4: "4: Numeric -> Mean, Text -> None",
        5: "5: Numeric -> 0, Text -> None"
    }[x]
)

analyze = st.sidebar.checkbox("Enable Data Analysis Report", value=True)
heading = st.sidebar.checkbox("Show DataFrame Head", value=True)

table_name = st.sidebar.text_input("Output File Name", value="work")
table_type = st.sidebar.selectbox(
    "File Format / Export Option", 
    ["csv", "excel", "json", "clipboard", "dict", "hdf", "latex"]
)

# Sidebar Contact Information with Clickable Icons
st.sidebar.markdown("---")
st.sidebar.markdown("### Contact Me")
st.sidebar.markdown("**Tariq Zeyad Jawad**")
st.sidebar.markdown("📞 +9647717447089")
st.sidebar.markdown("✉️ tariq.z.jawad4@gmail.com")

st.sidebar.markdown("""
    <div class="social-icons">
        <a href="https://www.linkedin.com/in/tariq-jawad?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" title="LinkedIn">🌐 LinkedIn</a>
        <a href="https://github.com/TariqZJawad" target="_blank" title="GitHub">💻 GitHub</a>
    </div>
""", unsafe_allow_html=True)

# Main Application Interface
st.title("📊 WebToTable Professional Engine")
st.markdown("Extract, clean, analyze, and export tabular data seamlessly from any web page.")

# Reset Button to clear session
if st.button("🔄 Reset / Clear Session"):
    st.session_state.df = None
    st.session_state.processed = False
    st.rerun()

# Core Extraction Function
def process_scraping():
    if not url:
        st.error("Please enter a valid target web URL.")
        return

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        
        table = None
        if isinstance(in_table, int):
            if len(tables) <= in_table:
                st.error("There is no table in this sequence.")
                return
            table = tables[in_table]
        elif isinstance(in_table, str):
            for t in tables:
                if in_table.lower() in t.text.lower():
                    table = t
                    break
            if table is None:
                st.error(f"Sorry, the table containing it could not be found: {in_table}")
                return
                
        titles = table.find_all('th')
        col_titles = [title.text.strip() for title in titles]
        col_titles = [col for col in col_titles if col != '']
        
        if not col_titles:
            first_row = table.find('tr')
            if first_row:
                col_titles = [cell.text.strip() for cell in first_row.find_all(['td', 'th'])]

        df = pd.DataFrame(columns=col_titles[:len(col_titles)])
        col_data = table.find_all('tr')
        
        for row in col_data[1:]:
            row_data = row.find_all(['td', 'th'])
            ind_row_data = [data.text.strip() for data in row_data]
            if len(ind_row_data):
                if len(ind_row_data) < len(df.columns):
                    ind_row_data += [''] * (len(df.columns) - len(ind_row_data))
                elif len(ind_row_data) > len(df.columns):
                    ind_row_data = ind_row_data[:len(df.columns)]    
                length = len(df)
                df.loc[length] = ind_row_data
                
        df = df.replace('', np.nan)
        df = df.drop_duplicates()

        # Cleaning Logic & Automatic Type Inference / Conversion for all columns
        for col in df.columns:
            # Clean common formatting like commas or percentage signs before conversion
            cleaned_series = df[col].astype(str).str.replace(',', '').str.replace('%', '')
            converted_col = pd.to_numeric(cleaned_series, errors='coerce')
            if converted_col.notna().sum() > 0 and converted_col.notna().sum() >= 0.5 * len(df):
                df[col] = converted_col

        if clean_option in [1, 2, 3, 4, 5]:
            for col in df.columns:
                temp_col = pd.to_numeric(df[col], errors='coerce')
                if clean_option == 1:
                    df = df.dropna()
                elif clean_option == 2:
                    df = df.fillna(value=None)
                elif clean_option == 3:
                    df = df.fillna(0)
                elif clean_option == 4:
                    if temp_col.notna().sum() > 0:
                        mean_val = temp_col.mean()
                        df[col] = df[col].fillna(mean_val)
                    else:
                        df[col] = df[col].fillna(value=None)
                elif clean_option == 5:
                    if temp_col.notna().sum() > 0:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(value=None)

        # Save to Session State
        st.session_state.df = df
        st.session_state.processed = True
        st.success("Table built successfully, cleaned, and types inferred!")

    except (ConnectionError, Timeout, TooManyRedirects) as e:
        st.error(f"Connection error: {e}")

# Trigger Extraction Execution Button
if st.button("🚀 Extract and Process Table"):
    with st.spinner("Processing web data... Please wait."):
        process_scraping()

# Render Results from Session State (Persist on download/clicks)
if st.session_state.processed and st.session_state.df is not None:
    df = st.session_state.df
    
    # 1. Analysis Report
    if analyze:
        st.markdown("---")
        st.subheader("📋 Data Analysis Report")
        
        num_rows, num_cols = df.shape
        total_cells = num_rows * num_cols
        missing_cells = df.isna().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", num_rows)
        col2.metric("Columns", num_cols)
        col3.metric("Total Cells", total_cells)
        col4.metric("Missing Cells (%)", f"{missing_cells} ({missing_percentage:.2f}%)")
        
        st.markdown("#### Statistical Description (Describe)")
        st.dataframe(df.describe(include='all'), use_container_width=True)
        
        # Replaced text info with a clean structured table showing column names and inferred data types
        st.markdown("#### DataFrame Structure (Columns & Data Types)")
        info_df = pd.DataFrame({
            "Column Name": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Data Type": [str(dtype) for dtype in df.dtypes.values]
        })
        st.dataframe(info_df, use_container_width=True)
        
        describe_str = df.describe(include='all').to_string()
        info_str = info_df.to_string(index=False)
        
        report_text = f"""=== DATA ANALYSIS REPORT ===
Rows: {num_rows}
Columns: {num_cols}
Total Cells: {total_cells}
Missing Cells: {missing_cells} ({missing_percentage:.2f}%)

--- Column Structure & Data Types ---
{info_str}

--- Statistical Description ---
{describe_str}
"""
        st.download_button(
            label="📥 Download Complete Analysis Report & Structure (.txt)",
            data=report_text,
            file_name=f"{table_name}_report.txt",
            mime="text/plain"
        )

    # 2. DataFrame Head Preview
    if heading:
        st.markdown("---")
        st.subheader("🔍 DataFrame Preview (Head)")
        st.dataframe(df.head(), use_container_width=True)

    # 3. Export Options
    if not table_name:
        table_name = "exported_table"

    st.markdown("---")
    st.subheader("💾 Export Options")
    
    if table_type == 'csv':
        out_filename = f"{table_name}.csv"
        st.download_button("📥 Download CSV File", df.to_csv(index=False).encode('utf-8'), out_filename, "text/csv")
    elif table_type == 'excel':
        out_filename = f"{table_name}.xlsx"
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        st.download_button("📥 Download Excel File", excel_buffer.getvalue(), out_filename, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif table_type == 'json':
        out_filename = f"{table_name}.json"
        st.download_button("📥 Download JSON File", df.to_json(index=False), out_filename, "application/json")
    elif table_type == 'clipboard':
        df.to_clipboard()
        st.info("Data copied to clipboard successfully!")
    elif table_type == 'dict':
        st.write(df.to_dict())
    elif table_type == 'hdf':
        st.info("HDF export format is configured for local storage backend.")
    elif table_type == 'latex':
        out_filename = f"{table_name}.tex"
        latex_code = df.to_latex()
        st.code(latex_code, language="latex")
        st.download_button("📥 Download LaTeX Code", latex_code, out_filename, "text/plain")
