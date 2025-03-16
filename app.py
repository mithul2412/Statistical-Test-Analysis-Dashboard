import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback

# Try importing individual question modules with error handling
st.set_page_config(page_title="Faculty Analysis Dashboard", layout="wide")

# Change these lines in app.py:
try:
    # Import individual question modules
    import_errors = []
    modules_available = []
    
    try:
        from scripts import q1_current_salaries
        modules_available.append("q1")
    except Exception as e:
        import_errors.append(f"Error importing q1_current_salaries: {str(e)}")
    
    try:
        from scripts import q2_starting_salaries
        modules_available.append("q2")
    except Exception as e:
        import_errors.append(f"Error importing q2_starting_salaries: {str(e)}")
        
    try:
        from scripts import q3_salary_increases
        modules_available.append("q3")
    except Exception as e:
        import_errors.append(f"Error importing q3_salary_increases: {str(e)}")
        
    try:
        from scripts import q4_promotion_decisions
        modules_available.append("q4")
    except Exception as e:
        import_errors.append(f"Error importing q4_promotion_decisions: {str(e)}")
    
    # Global variables 
    custom_palette = {'M': 'skyblue', 'F': 'lightcoral'}
    
    def display_data_overview(df):
        st.header("Data Overview")
        st.write("Select which analysis to run from the sidebar. Here's a preview of your data:")
        
        # Display the data preview
        st.write("### Data Preview")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.write("### Basic Statistics")
        st.write(f"Number of Records: {len(df)}")
        
        # Count by sex
        if 'sex' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                sex_counts = df['sex'].value_counts()
                st.write("#### Count by Sex")
                st.dataframe(sex_counts)
            
            with col2:
                # Create a simple pie chart
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%',
                       colors=[custom_palette.get(sex, '#999999') for sex in sex_counts.index])
                ax.set_title('Distribution by Sex')
                st.pyplot(fig)
        
        # Column information - improved display
        st.write("### Column Information")
        
        # Get column info
        columns_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isna().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(columns_df)
        
        # Data description
        st.write("### Numerical Columns Statistics")
        st.dataframe(df.describe())
    
    def main():
        st.title("Faculty Analysis Dashboard")
        
        # Display module import status
        # with st.expander("Debug Information"):
        #     st.write("### Available Modules")
        #     for module in modules_available:
        #         st.success(f"Module {module} successfully imported")
            
        #     st.write("### Import Errors")
        #     if import_errors:
        #         for error in import_errors:
        #             st.error(error)
        #     else:
        #         st.success("No import errors detected")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        
        # File upload - with caching using session_state
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
            
        uploaded_file = st.sidebar.file_uploader("Upload data file (space or tab-separated)", type=["txt", "csv"])
        
        # Update the session state if a new file is uploaded
        if uploaded_file is not None:
            file_contents = uploaded_file.getvalue()
            
            # If a new file was uploaded, or if the session state is empty, update it
            if st.session_state.uploaded_file is None or st.session_state.uploaded_file.getvalue() != file_contents:
                st.session_state.uploaded_file = uploaded_file
        
        # Use the cached file if available
        if st.session_state.uploaded_file is not None:
            uploaded_file = st.session_state.uploaded_file
            
            # Read the uploaded file
            try:
                # Reset the file pointer each time
                uploaded_file.seek(0)
                
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, sep='\\s+')
                    
                # Reset the file pointer again
                uploaded_file.seek(0)
                
                # Options in sidebar - only show modules that were successfully imported
                analysis_options = ["Data Overview"]
                if "q1" in modules_available:
                    analysis_options.append("Q1: Sex Bias in Recent Year")
                if "q2" in modules_available:
                    analysis_options.append("Q2: Sex Bias in Starting Salaries")
                if "q3" in modules_available:
                    analysis_options.append("Q3: Sex Bias in Salary Increases")
                if "q4" in modules_available:
                    analysis_options.append("Q4: Sex Bias in Promotion Decisions")
                
                app_mode = st.sidebar.radio("Select Analysis", analysis_options)
                
                if app_mode == "Data Overview":
                    display_data_overview(df)
                
                elif app_mode == "Q1: Sex Bias in Recent Year" and "q1" in modules_available:
                    st.header("Question 1: Does sex bias exist at the university in the most current year available (1995)?")
                    try:
                        q1_current_salaries.run_analysis(uploaded_file)
                    except Exception as e:
                        st.error(f"Error running Q1 analysis: {str(e)}")
                        st.exception(e)
                    
                elif app_mode == "Q2: Sex Bias in Starting Salaries" and "q2" in modules_available:
                    st.header("Question 2: Has sex bias existed in the starting salaries of faculty members(salaries in the year hired)?")
                    try:
                        q2_starting_salaries.run_analysis(uploaded_file)
                    except Exception as e:
                        st.error(f"Error running Q2 analysis: {str(e)}")
                        st.exception(e)
                    
                elif app_mode == "Q3: Sex Bias in Salary Increases" and "q3" in modules_available:
                    st.header("Question 3: Has sex bias existed in granting salary increases between 1990 and 1995?")
                    try:
                        q3_salary_increases.run_analysis(uploaded_file)
                    except Exception as e:
                        st.error(f"Error running Q3 analysis: {str(e)}")
                        st.exception(e)
                
                elif app_mode == "Q4: Sex Bias in Promotion Decisions" and "q4" in modules_available:
                    st.header("Question 4: Has sex bias existed in granting promotions from Associate to full Professor?")
                    try:
                        q4_promotion_decisions.run_analysis(uploaded_file)
                    except Exception as e:
                        st.error(f"Error running Q4 analysis: {str(e)}")
                        st.exception(e)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Please ensure your file is in the correct format (CSV or tab/space-separated).")
        
        else:
            st.header("Faculty Analysis Dashboard")
            st.write("""
            This application allows you to analyze faculty data for potential gender bias in:
            
            1. **Current Salaries** - Is there a gender gap in overall compensation?
            2. **Starting Salaries** - Are there differences in initial compensation?
            3. **Salary Increases** - Are raises distributed equitably over time?
            4. **Promotion Decisions** - Do promotion rates and timing differ by gender?
            
            To begin, upload a faculty dataset using the file uploader in the sidebar.
            """)
            
            st.info("""
            ### Expected Data Format
            
            The data file should contain faculty information with the following columns:
            
            - **id**: Faculty identifier
            - **year**: Year of observation
            - **sex**: Gender (M/F)
            - **deg**: Degree (e.g., PhD, Masters)
            - **field**: Academic field (e.g., Arts, Sciences, Prof)
            - **rank**: Academic rank (e.g., Full, Assoc, Assist)
            - **admin**: Administrative duties (0/1)
            - **salary**: Annual salary
            - **startyr**: Start year (for Questions 2, 3 & 4)
            - **yrdeg**: Year degree was obtained (for Question 4)
            
            The file should be in CSV format or space/tab-separated text.
            """)
    
    if __name__ == "__main__":
        main()
except Exception as e:
    st.error(f"Critical error in app initialization: {str(e)}")
    st.code(traceback.format_exc())
