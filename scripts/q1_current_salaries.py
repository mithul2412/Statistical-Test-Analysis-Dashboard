import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

# Define palette for consistent colors
custom_palette = {'M': 'skyblue', 'F': 'lightcoral'}

def load_data(uploaded_file):
    """Load and preprocess data for current salaries analysis"""
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\\s+')
        
        # Convert 2-digit years to 4-digits if needed
        if 'year' in df.columns and df['year'].max() < 100:
            df['year'] = df['year'].apply(lambda x: 1900 + x if x < 100 else x)
            
        # Ensure sex column is treated as a string
        if 'sex' in df.columns:
            df['sex'] = df['sex'].astype(str)
            
        # Filter for the most recent year if needed
        if 'year' in df.columns:
            latest_year = df['year'].max()
            df_latest = df[df['year'] == latest_year]
            if len(df_latest) > 10:  # Make sure we have enough data
                # st.info(f"Filtered for the most recent year in dataset: {latest_year}")
                return df_latest
        
        # If we can't filter by year, return all data
        return df
        
    except Exception as e:
        st.error(f"Error processing file for Question 1: {str(e)}")
        st.exception(e)
        return None

def exploratory_analysis(df):
    """Perform exploratory analysis with visualizations"""
    # st.subheader("Exploratory Analysis")
    
    try:
        # PART 1: ALL TABLES FIRST
        # st.write("## Summary Tables")
        
        # Summary statistics by gender
        if 'sex' in df.columns and 'salary' in df.columns:
            st.write("### Summary Statistics")
            
            # Group and calculate summary statistics
            summary_stats = df.groupby('sex')['salary'].agg(['mean', 'median', 'std', 'count'])
            
            # Format the summary table
            summary_stats = summary_stats.round(2)
            summary_stats.columns = ['Mean Salary', 'Median Salary', 'Std Deviation', 'Count']
            summary_stats.index.name = 'Gender'
            
            st.dataframe(summary_stats)
            
            # Distribution by rank and sex if rank column exists
            if 'rank' in df.columns:
                st.write("### Distribution by Rank and Sex")
                
                # Create count table
                rank_sex_counts = pd.crosstab(df['rank'], df['sex'], margins=True, margins_name='Total')
                st.write("#### Count by Rank and Sex")
                st.dataframe(rank_sex_counts)
                
                # Create proportion table (percentage within rank)
                rank_sex_prop = pd.crosstab(df['rank'], df['sex'], normalize='index') * 100
                rank_sex_prop = rank_sex_prop.round(1)
                st.write("#### Percentage by Rank (row %)")
                st.dataframe(rank_sex_prop)
            
            # Distribution by administrative role and sex if admin column exists
            if 'admin' in df.columns:
                st.write("### Distribution by Administrative Role and Sex")
                
                # Create admin role labels for better readability
                df_admin = df.copy()
                admin_labels = {0: 'No Admin Role', 1: 'Admin Role'}
                df_admin['admin_label'] = df_admin['admin'].map(admin_labels)
                
                # Create count table
                admin_sex_counts = pd.crosstab(df_admin['admin_label'], df_admin['sex'], margins=True, margins_name='Total')
                st.write("#### Count by Administrative Role and Sex")
                st.dataframe(admin_sex_counts)
                
                # Create proportion table (percentage within admin category)
                admin_sex_prop = pd.crosstab(df_admin['admin_label'], df_admin['sex'], normalize='index') * 100
                admin_sex_prop = admin_sex_prop.round(1)
                st.write("#### Percentage by Administrative Role (row %)")
                st.dataframe(admin_sex_prop)
            
            # Distribution by field and sex if field column exists
            if 'field' in df.columns:
                st.write("### Distribution by Field and Sex")
                
                # Create count table for field
                field_sex_counts = pd.crosstab(df['field'], df['sex'], margins=True, margins_name='Total')
                st.write("#### Count by Field and Sex")
                st.dataframe(field_sex_counts)
                
                # Create proportion table for field
                field_sex_prop = pd.crosstab(df['field'], df['sex'], normalize='index') * 100
                field_sex_prop = field_sex_prop.round(1)
                st.write("#### Percentage by Field (row %)")
                st.dataframe(field_sex_prop)
            
            # PART 2: ALL VISUALIZATIONS
            st.write("### Visualizations")
            
            # Create two columns for the first row of charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot: Salary by Sex and Rank
                # st.write("### Salary Distribution by Rank and Sex")
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x='rank', y='salary', hue='sex', palette=custom_palette, ax=ax1)
                ax1.set_title('Salary Distribution by Rank and Sex')
                ax1.set_xlabel('Rank')
                ax1.set_ylabel('Salary ($)')
                ax1.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig1)
            
            with col2:
                # Density Plot: Salary by Sex
                # st.write("### Salary Distribution Density by Sex")
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for sex, color in custom_palette.items():
                    if sex in df['sex'].unique():
                        sns.kdeplot(data=df[df['sex'] == sex], x='salary', fill=True, 
                                  label=f"{sex} (n={len(df[df['sex'] == sex])})", 
                                  color=color, alpha=0.6, ax=ax2)
                
                ax2.set_title('Salary Density Plot by Sex')
                ax2.set_xlabel('Salary ($)')
                ax2.set_ylabel('Density')
                ax2.legend(title='Gender')
                ax2.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig2)
            
            # Create two columns for the second row of charts
            col3, col4 = st.columns(2)
            
            with col3:
                # Bar chart for rank distribution
                if 'rank' in df.columns:
                    # st.write("### Proportion by Rank and Sex")
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    
                    # Calculate proportions for each rank
                    rank_counts = df.groupby(['rank', 'sex']).size().unstack(fill_value=0)
                    rank_props = rank_counts.div(rank_counts.sum(axis=1), axis=0)
                    
                    rank_props.plot(kind='bar', stacked=False, ax=ax3, color=[custom_palette.get(i, '#999999') for i in rank_props.columns])
                    ax3.set_title('Proportion of Men/Women by Rank')
                    ax3.set_xlabel('Rank')
                    ax3.set_ylabel('Proportion')
                    ax3.legend(title='Sex')
                    ax3.grid(True, linestyle='--', alpha=0.6)
                    
                    st.pyplot(fig3)
            
            with col4:
                # Bar chart for admin role distribution
                if 'admin' in df.columns:
                    # st.write("### Proportion by Administrative Role and Sex")
                    
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    
                    # Calculate proportions for each admin category
                    admin_counts = df_admin.groupby(['admin_label', 'sex']).size().unstack(fill_value=0)
                    admin_props = admin_counts.div(admin_counts.sum(axis=1), axis=0)
                    
                    admin_props.plot(kind='bar', stacked=False, ax=ax4, color=[custom_palette.get(i, '#999999') for i in admin_props.columns])
                    ax4.set_title('Proportion of Men/Women by Administrative Role')
                    ax4.set_xlabel('Administrative Role')
                    ax4.set_ylabel('Proportion')
                    ax4.legend(title='Sex')
                    ax4.grid(True, linestyle='--', alpha=0.6)
                    
                    st.pyplot(fig4)
        else:
            st.error("Required columns 'sex' and 'salary' not found in the dataset.")
    
    except Exception as e:
        st.error(f"Error in exploratory analysis: {str(e)}")
        st.exception(e)

def statistical_tests(df):
    """Run statistical tests for current salaries analysis"""
    st.subheader("Statistical Tests")
    
    # Check for required columns
    required_cols = ['sex', 'salary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for statistical tests: {', '.join(missing_cols)}")
        return
    
    try:
        # Create a binary sex variable
        df['sex_binary'] = df['sex'].map({'F': 0, 'M': 1})
        
        # 1. Two-Sample T-Test
        st.write("### T-Test: Male vs Female Salaries")
        
        # Extract values for each group
        male_salaries = df[df['sex'] == 'M']['salary']
        female_salaries = df[df['sex'] == 'F']['salary']
        
        # Calculate means for display
        male_mean = male_salaries.mean()
        female_mean = female_salaries.mean()
        mean_diff = male_mean - female_mean
        
        # Perform t-test
        t_stat, p_val = ttest_ind(male_salaries, female_salaries, equal_var=False)
        
        # Create a DataFrame for clean display
        ttest_results = pd.DataFrame({
            'Statistic': ['Male Mean Salary', 'Female Mean Salary', 'Mean Difference', 'T-statistic', 'P-value', 'Significant (α=0.05)'],
            'Value': [
                f"${male_mean:.2f}",
                f"${female_mean:.2f}",
                f"${mean_diff:.2f}",
                f"{t_stat:.4f}",
                f"{p_val}",
                "Yes" if p_val < 0.05 else "No"
            ]
        })
        
        st.dataframe(ttest_results)
        
        # Interpretation
        if p_val < 0.05:
            if mean_diff > 0:
                st.write(f"**Interpretation**: There is a statistically significant difference in salaries. On average, men earn ${mean_diff:.2f} more than women (p={p_val}).")
            else:
                st.write(f"**Interpretation**: There is a statistically significant difference in salaries. On average, women earn ${abs(mean_diff):.2f} more than men (p={p_val}).")
        else:
            st.write(f"**Interpretation**: There is no statistically significant difference in salaries between men and women (p={p_val}).")
                
        # 2. Simple Linear Regression (Unadjusted)
        st.write("### Simple Linear Regression: Sex Effect on Salary (Unadjusted)")
        
        # Adding a constant to the independent variable for the intercept
        X_simple = sm.add_constant(df['sex_binary'])
        y = df['salary']
        
        # Fit the simple model
        model_simple = sm.OLS(y, X_simple).fit()
        
        # Extract key results
        intercept = model_simple.params['const']
        sex_coef = model_simple.params['sex_binary']
        sex_pval = model_simple.pvalues['sex_binary']
        r_squared = model_simple.rsquared
        adj_r_squared = model_simple.rsquared_adj
        f_stat = model_simple.fvalue
        f_pval = model_simple.f_pvalue
        
        # Create a DataFrame for model statistics
        model_simple_stats = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F p-value', 'AIC', 'BIC', 'Observations'],
            'Value': [
                f"{r_squared:.4f}",
                f"{adj_r_squared:.4f}",
                f"{f_stat:.4f}",
                f"{f_pval}",
                f"{model_simple.aic:.4f}",
                f"{model_simple.bic:.4f}",
                f"{model_simple.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model_simple_stats)
        
        # Create a DataFrame for the coefficients
        coef_table_simple = pd.DataFrame({
            'Variable': ['Intercept', 'Sex (Male=1)'],
            'Coefficient': [
                f"{intercept:.2f}", 
                f"{sex_coef:.2f}"
            ],
            'Std Error': [
                f"{model_simple.bse['const']:.2f}", 
                f"{model_simple.bse['sex_binary']:.2f}"
            ],
            't-value': [
                f"{model_simple.tvalues['const']:.4f}", 
                f"{model_simple.tvalues['sex_binary']:.4f}"
            ],
            'P-value': [
                "<0.001", 
                f"{model_simple.pvalues['sex_binary']}"
            ],
            'CI Lower 95%': [
                f"{model_simple.conf_int().loc['const'][0]:.2f}",
                f"{model_simple.conf_int().loc['sex_binary'][0]:.2f}"
            ],
            'CI Upper 95%': [
                f"{model_simple.conf_int().loc['const'][1]:.2f}",
                f"{model_simple.conf_int().loc['sex_binary'][1]:.2f}"
            ]
        })
        
        st.write("**Coefficients:**")
        st.dataframe(coef_table_simple)
        
        # Interpretation
        if sex_pval < 0.05:
            if sex_coef > 0:
                st.write(f"**Interpretation**: Sex is a significant predictor of salary. Men earn ${sex_coef:.2f} more than women on average (p={sex_pval}).")
            else:
                st.write(f"**Interpretation**: Sex is a significant predictor of salary. Women earn ${abs(sex_coef):.2f} more than men on average (p={sex_pval}).")
        else:
            st.write(f"**Interpretation**: Sex is not a significant predictor of salary (p={sex_pval}).")
        
        # Display full model results in an expander
        with st.expander("View Full Simple Regression Results"):
            st.text(model_simple.summary().as_text())
        
        # 3. Multiple Linear Regression (Adjusted)
        st.write("### Multiple Linear Regression: Sex Effect on Salary (Adjusted)")
        
        # Check which control variables are available
        categorical_cols = []
        if 'rank' in df.columns:
            categorical_cols.append('rank')
        if 'admin' in df.columns:
            categorical_cols.append('admin')
        if 'deg' in df.columns:
            categorical_cols.append('deg')
        if 'field' in df.columns:
            categorical_cols.append('field')
            
        if categorical_cols:
            # Create a copy of the data for adjusted analysis
            df_adj = df.copy()
            
            # Ensure sex_binary is created
            df_adj['sex_binary'] = df_adj['sex'].map({'F': 0, 'M': 1})
            
            # Creating dummy variables for categorical predictors
            df_adj = pd.get_dummies(df_adj, columns=categorical_cols, drop_first=True)
            
            # Define columns to exclude
            exclude_cols = ['salary', 'case', 'id', 'sex', 'year', 'yrdeg', 'startyr']
            
            # Define independent variables (exclude unnecessary columns)
            X_adj = df_adj.drop(columns=exclude_cols)
            
            # Add constant term
            X_adj = sm.add_constant(X_adj)
            
            # Convert boolean columns to integers if needed
            bool_cols = X_adj.select_dtypes(include=['bool']).columns
            if len(bool_cols) > 0:
                X_adj[bool_cols] = X_adj[bool_cols].astype(int)
            
            # Define dependent variable
            y_adj = df_adj['salary']
            
            # Fit multiple regression model
            model_adj = sm.OLS(y_adj, X_adj).fit()
            
            # Extract key results
            adj_sex_coef = model_adj.params['sex_binary']
            adj_sex_pval = model_adj.pvalues['sex_binary']
            
            # Create a DataFrame for model statistics
            model_adj_stats = pd.DataFrame({
                'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F p-value', 'AIC', 'BIC', 'Observations'],
                'Value': [
                    f"{model_adj.rsquared:.4f}",
                    f"{model_adj.rsquared_adj:.4f}",
                    f"{model_adj.fvalue:.4f}",
                    f"{model_adj.f_pvalue}",
                    f"{model_adj.aic:.4f}",
                    f"{model_adj.bic:.4f}",
                    f"{model_adj.nobs}"
                ]
            })
            
            st.write("**Model Statistics:**")
            st.dataframe(model_adj_stats)
            
            # Create a table of coefficients
            coef_data = []
            for var in model_adj.params.index:
                if var == 'const':
                    var_name = 'Intercept'
                elif var == 'sex_binary':
                    var_name = 'Sex (Male=1)'
                else:
                    var_name = var.replace('_', ' ').capitalize()
                
                coef_data.append({
                    'Variable': var_name,
                    'Coefficient': f"{model_adj.params[var]:.2f}",
                    'Std Error': f"{model_adj.bse[var]:.2f}",
                    't-value': f"{model_adj.tvalues[var]:.4f}",
                    'P-value': f"{model_adj.pvalues[var]}",
                    'Significant': "Yes" if model_adj.pvalues[var] < 0.05 else "No"
                })
            
            # Create DataFrame for all coefficients
            coef_df_adj = pd.DataFrame(coef_data)
            
            # First show the sex effect
            sex_effect = coef_df_adj[coef_df_adj['Variable'] == 'Sex (Male=1)'].copy()
            st.write("**Sex Effect (Adjusted):**")
            st.dataframe(sex_effect)
            
            # Show all significant variables
            sig_vars = coef_df_adj[coef_df_adj['Significant'] == 'Yes'].copy()
            st.write("**Significant Variables:**")
            st.dataframe(sig_vars)
            
            # Display all coefficients in an expander
            with st.expander("View All Coefficients"):
                st.dataframe(coef_df_adj)
            
            # Interpretation
            if adj_sex_pval < 0.05:
                if adj_sex_coef > 0:
                    st.write(f"**Interpretation**: After controlling for other factors, men earn ${adj_sex_coef:.2f} more than women on average (p={adj_sex_pval}).")
                else:
                    st.write(f"**Interpretation**: After controlling for other factors, women earn ${abs(adj_sex_coef):.2f} more than men on average (p={adj_sex_pval}).")
            else:
                st.write(f"**Interpretation**: After controlling for other factors, there is no significant difference in salary between men and women (p={adj_sex_pval}).")
            
            # Display full model results in an expander
            with st.expander("View Full Multiple Regression Results"):
                st.text(model_adj.summary().as_text())
            
            # 4. Interaction Effects
            st.write("### Interaction Effects")
            
            if 'admin' in df.columns:
                st.write("#### Interaction between Sex and Administrative Duties")
                
                # Creating a copy of the data for interaction analysis
                df_int_admin = df_adj.copy()
                
                # Find admin dummy variable
                admin_cols = [col for col in df_int_admin.columns if col.startswith('admin_')]
                
                if admin_cols:
                    # Use the first admin dummy column
                    admin_col = admin_cols[0]
                    
                    # Add interaction term between sex and admin
                    df_int_admin['sex_admin'] = df_int_admin['sex_binary'] * df_int_admin[admin_col]
                    
                    # Define variables to include in the model
                    rank_cols = [col for col in df_int_admin.columns if col.startswith('rank_')]
                    deg_cols = [col for col in df_int_admin.columns if col.startswith('deg_')]
                    field_cols = [col for col in df_int_admin.columns if col.startswith('field_')]
                    
                    # Create list of predictors
                    predictors = ['sex_binary', admin_col, 'sex_admin'] + rank_cols + deg_cols + field_cols
                    
                    # Define X and y
                    X_int_admin = df_int_admin[predictors]
                    X_int_admin = sm.add_constant(X_int_admin)
                    y_int_admin = df_int_admin['salary']
                    
                    # Convert any boolean columns to integers
                    bool_cols = X_int_admin.select_dtypes(include=['bool']).columns
                    if len(bool_cols) > 0:
                        X_int_admin[bool_cols] = X_int_admin[bool_cols].astype(int)
                    
                    # Fit interaction model
                    model_int_admin = sm.OLS(y_int_admin, X_int_admin).fit()
                    
                    # Extract interaction term results
                    int_coef = model_int_admin.params['sex_admin']
                    int_pval = model_int_admin.pvalues['sex_admin']
                    
                    # Create interaction results table
                    int_results = pd.DataFrame({
                        'Term': ['Sex × Admin Interaction'],
                        'Coefficient': [f"{int_coef:.2f}"],
                        'Std Error': [f"{model_int_admin.bse['sex_admin']:.2f}"],
                        't-value': [f"{model_int_admin.tvalues['sex_admin']:.4f}"],
                        'P-value': [f"{int_pval}"],
                        'Significant': ["Yes" if int_pval < 0.05 else "No"]
                    })
                    
                    st.dataframe(int_results)
                    
                    # Interpretation
                    if int_pval < 0.05:
                        if int_coef > 0:
                            st.write(f"**Interpretation**: There is a significant interaction between sex and administrative duties. Male administrators earn ${int_coef:.2f} more than would be expected from the main effects alone (p={int_pval}).")
                        else:
                            st.write(f"**Interpretation**: There is a significant interaction between sex and administrative duties. Male administrators earn ${abs(int_coef):.2f} less than would be expected from the main effects alone (p={int_pval}).")
                    else:
                        st.write(f"**Interpretation**: There is no significant interaction between sex and administrative duties (p={int_pval}).")
                    
                    # Display full model results in an expander
                    with st.expander("View Full Admin Interaction Model Results"):
                        st.text(model_int_admin.summary().as_text())
            
            # 5. Interaction with Field
            if 'field' in df.columns:
                st.write("#### Interaction between Sex and Field")
                
                # Creating a copy of the data for field interaction analysis
                df_int_field = df_adj.copy()
                
                # Find field dummy variables
                field_cols = [col for col in df_int_field.columns if col.startswith('field_')]
                
                if field_cols:
                    # Add interaction terms for each field
                    for field_col in field_cols:
                        df_int_field[f'sex_{field_col}'] = df_int_field['sex_binary'] * df_int_field[field_col]
                    
                    # Get other predictors
                    rank_cols = [col for col in df_int_field.columns if col.startswith('rank_')]
                    admin_cols = [col for col in df_int_field.columns if col.startswith('admin_')]
                    deg_cols = [col for col in df_int_field.columns if col.startswith('deg_')]
                    
                    # Create list of all predictors
                    predictors = ['sex_binary'] + field_cols
                    interaction_terms = [f'sex_{col}' for col in field_cols]
                    all_predictors = predictors + interaction_terms + rank_cols + admin_cols + deg_cols
                    
                    # Define X and y
                    X_int_field = df_int_field[all_predictors]
                    X_int_field = sm.add_constant(X_int_field)
                    y_int_field = df_int_field['salary']
                    
                    # Convert any boolean columns to integers
                    bool_cols = X_int_field.select_dtypes(include=['bool']).columns
                    if len(bool_cols) > 0:
                        X_int_field[bool_cols] = X_int_field[bool_cols].astype(int)
                    
                    # Fit field interaction model
                    model_int_field = sm.OLS(y_int_field, X_int_field).fit()
                    
                    # Extract interaction terms
                    interaction_terms = [term for term in model_int_field.params.index if term.startswith('sex_field_')]
                    
                    if interaction_terms:
                        # Create a DataFrame for interaction results
                        field_inter_data = []
                        for term in interaction_terms:
                            coef = model_int_field.params[term]
                            p_val = model_int_field.pvalues[term]
                            field = term.replace('sex_field_', '')
                            
                            field_inter_data.append({
                                'Interaction': f"Sex × {field}",
                                'Coefficient': f"{coef:.2f}",
                                'Std Error': f"{model_int_field.bse[term]:.2f}",
                                't-value': f"{model_int_field.tvalues[term]:.4f}",
                                'P-value': f"{p_val}",
                                'Significant': "Yes" if p_val < 0.05 else "No"
                            })
                        
                        field_inter_df = pd.DataFrame(field_inter_data)
                        st.dataframe(field_inter_df)
                        
                        # Interpretation for significant interactions
                        sig_inters = [d for d in field_inter_data if d['Significant'] == "Yes"]
                        if sig_inters:
                            st.write("**Significant interactions found:**")
                            for inter in sig_inters:
                                parts = inter['Interaction'].split('×')
                                field = parts[1].strip()
                                coef = float(inter['Coefficient'])
                                p_val = float(inter['P-value'].replace(',', '.') if isinstance(inter['P-value'], str) else inter['P-value'])
                                
                                if coef > 0:
                                    st.write(f"- For {field} field, men earn ${coef:.2f} more than would be expected from the main effects alone (p={p_val}).")
                                else:
                                    st.write(f"- For {field} field, men earn ${abs(coef):.2f} less than would be expected from the main effects alone (p={p_val}).")
                        else:
                            st.write("**No significant interactions found between sex and field.**")
                        
                        # Display full model results in an expander
                        with st.expander("View Full Field Interaction Model Results"):
                            st.text(model_int_field.summary().as_text())
            
            # 6. Residual Analysis
            st.write("### Residual Analysis")

            # Define a function to analyze residuals
            def analyze_residuals(residuals):
                """
                Analyze residuals by plotting a histogram and performing the Shapiro-Wilk normality test.
                
                Parameters:
                residuals (array-like): The residuals to analyze.
                """
                # First column for histogram
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of residuals
                    fig_res1, ax_res1 = plt.subplots(figsize=(6, 4))
                    ax_res1.hist(residuals, bins=30, edgecolor='black')
                    ax_res1.set_title('Histogram of Residuals')
                    ax_res1.set_xlabel('Residuals')
                    ax_res1.set_ylabel('Frequency')
                    ax_res1.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_res1)
                
                with col2:
                    # Q-Q plot of residuals
                    fig_res2, ax_res2 = plt.subplots(figsize=(6, 4))
                    sm.qqplot(residuals, line='45', ax=ax_res2)
                    ax_res2.set_title('Q-Q Plot of Residuals')
                    ax_res2.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_res2)
                
                # Performing the Shapiro-Wilk normality test
                stat, p_value_residuals = stats.shapiro(residuals)
                
                # Create a DataFrame for normality test results
                normality_results = pd.DataFrame({
                    'Statistic': ['Shapiro-Wilk Statistic', 'P-value', 'Normality Assumption'],
                    'Value': [
                        f"{stat:.4f}",
                        f"{p_value_residuals}",
                        "Satisfied" if p_value_residuals > 0.05 else "Violated"
                    ]
                })
                
                st.write("**Shapiro-Wilk Test for Normality of Residuals:**")
                st.dataframe(normality_results)
                
                # Interpretation
                if p_value_residuals < 0.05:
                    st.write("**Interpretation**: The residuals do not follow a normal distribution (p < 0.05). This may indicate that the model assumptions are violated, potentially affecting the reliability of p-values and confidence intervals.")
                else:
                    st.write("**Interpretation**: The residuals follow a normal distribution (p > 0.05). This suggests that the model assumptions regarding error distribution are satisfied.")
                
                return stat, p_value_residuals

            # Get residuals from the interaction admin model instead of the adjusted model
            residuals = model_int_admin.resid

            # Analyze residuals
            analyze_residuals(residuals)
                        
            # 7. Robust Standard Errors
            st.write("### Robust Standard Errors")

            try:
                # Generate robust covariance results for the interaction admin model instead of adjusted model
                robust_results = model_int_admin.get_robustcov_results(cov_type='HC1')
                
                # Check if 'sex_binary' is in the parameters before accessing
                if 'sex_binary' in robust_results.params:
                    # Extract sex coefficient and p-value with robust standard errors
                    robust_sex_coef = robust_results.params['sex_binary']
                    robust_sex_pval = robust_results.pvalues['sex_binary']
                    robust_sex_bse = robust_results.bse['sex_binary']
                    
                    # Get original coefficient for comparison (from interaction admin model)
                    original_sex_coef = model_int_admin.params['sex_binary']
                    original_sex_pval = model_int_admin.pvalues['sex_binary']
                    
                    # Create a comparison table
                    robust_comparison = pd.DataFrame({
                        'Method': ['Robust Standard Errors'],
                        'Sex Coefficient': [
                            f"{robust_sex_coef:.2f}"
                        ],
                        'Standard Error': [
                            f"{robust_sex_bse:.2f}"
                        ],
                        'P-value': [
                            f"{robust_sex_pval}"
                        ],
                        'Significant': [
                            "Yes" if robust_sex_pval < 0.05 else "No"
                        ]
                    })
                    
                    st.write("**Robust Standard Errors for Sex Effect:**")
                    st.dataframe(robust_comparison)
                    
                    # Interpretation
                    if robust_sex_pval < 0.05:
                        if robust_sex_coef > 0:
                            st.write(f"**Interpretation with Robust Standard Errors**: Even after accounting for heteroskedasticity, men earn ${robust_sex_coef:.2f} more than women on average (p={robust_sex_pval}).")
                        else:
                            st.write(f"**Interpretation with Robust Standard Errors**: Even after accounting for heteroskedasticity, women earn ${abs(robust_sex_coef):.2f} more than men on average (p={robust_sex_pval}).")
                    else:
                        st.write(f"**Interpretation with Robust Standard Errors**: After accounting for heteroskedasticity, there is no significant difference in salary between men and women (p={robust_sex_pval}).")
                else:
                    # Try to handle the case where parameters are numpy arrays instead of pandas Series
                    param_names = model_int_admin.params.index
                    if 'sex_binary' in param_names:
                        idx = param_names.get_loc('sex_binary')
                        robust_sex_coef = robust_results.params[idx]
                        robust_sex_pval = robust_results.pvalues[idx]
                        robust_sex_bse = robust_results.bse[idx]
                        
                        original_sex_coef = model_int_admin.params[idx]
                        original_sex_pval = model_int_admin.pvalues[idx]
                        
                        # Create a comparison table
                        robust_comparison = pd.DataFrame({
                            'Method': ['Robust Standard Errors'],
                            'Sex Coefficient': [
                                f"{robust_sex_coef:.2f}"
                            ],
                            'Standard Error': [
                                f"{robust_sex_bse:.2f}"
                            ],
                            'P-value': [
                                f"{robust_sex_pval}"
                            ],
                            'Significant': [
                                "Yes" if robust_sex_pval < 0.05 else "No"
                            ]
                        })
                        
                        st.write("**Robust Standard Errors for Sex Effect:**")
                        st.dataframe(robust_comparison)
                        
                        # Interpretation
                        if robust_sex_pval < 0.05:
                            if robust_sex_coef > 0:
                                st.write(f"**Interpretation with Robust Standard Errors**: Even after accounting for heteroskedasticity, men earn ${robust_sex_coef:.2f} more than women on average (p={robust_sex_pval}).")
                            else:
                                st.write(f"**Interpretation with Robust Standard Errors**: Even after accounting for heteroskedasticity, women earn ${abs(robust_sex_coef):.2f} more than men on average (p={robust_sex_pval}).")
                        else:
                            st.write(f"**Interpretation with Robust Standard Errors**: After accounting for heteroskedasticity, there is no significant difference in salary between men and women (p={robust_sex_pval}).")
                    else:
                        st.warning("Could not find 'sex_binary' parameter in the model results.")
                
                # Display full robust model results in an expander
                with st.expander("View Full Robust Model Results"):
                    st.text(robust_results.summary().as_text())
            except Exception as e:
                st.warning(f"Error calculating robust standard errors: {str(e)}")
                st.write("Robust standard errors could not be calculated. This might be due to issues with the model or data.")

    except Exception as e:
        st.error(f"Error in statistical tests: {str(e)}")
        st.exception(e)

def summary(df):
    """Generate a summary of findings for Question 1"""
    st.subheader("Question 1 Summary: Sex Bias in Recent year (1995)")
    
    if df is None or df.empty:
        st.error("No data available for summary.")
        return
        
    try:
        # Summary statistics
        st.write("### Key Findings")
        
        # Create binary sex variable
        df['sex_binary'] = df['sex'].map({'F': 0, 'M': 1})
        
        # Calculate basic salary statistics by sex
        salary_by_sex = df.groupby('sex')['salary'].agg(['mean', 'median', 'std', 'count'])
        
        # Format for display
        salary_by_sex = salary_by_sex.round(2)
        salary_by_sex.columns = ['Mean Salary', 'Median Salary', 'Std Deviation', 'Count']
        
        # Calculate raw difference
        male_mean = salary_by_sex.loc['M', 'Mean Salary'] if 'M' in salary_by_sex.index else 0
        female_mean = salary_by_sex.loc['F', 'Mean Salary'] if 'F' in salary_by_sex.index else 0
        raw_diff = male_mean - female_mean
        
        # T-test 
        male_salaries = df[df['sex'] == 'M']['salary']
        female_salaries = df[df['sex'] == 'F']['salary']
        t_stat, p_val = ttest_ind(male_salaries, female_salaries, equal_var=False)
        
        # Simple regression
        X_simple = sm.add_constant(df['sex_binary'])
        y = df['salary']
        simple_model = sm.OLS(y, X_simple).fit()
        sex_coef = simple_model.params['sex_binary']
        sex_pval = simple_model.pvalues['sex_binary']
        
        # Multiple regression if control variables are available
        categorical_cols = []
        if 'rank' in df.columns:
            categorical_cols.append('rank')
        if 'admin' in df.columns:
            categorical_cols.append('admin')
        if 'deg' in df.columns:
            categorical_cols.append('deg')
        if 'field' in df.columns:
            categorical_cols.append('field')
            
        adj_sex_coef = None
        adj_sex_pval = None
        
        if categorical_cols:
            try:
                # Create a copy of data for adjusted analysis
                df_adj = df.copy()
                
                # Ensure sex_binary is created
                df_adj['sex_binary'] = df_adj['sex'].map({'F': 0, 'M': 1})
                
                # Creating dummy variables for categorical predictors
                df_adj = pd.get_dummies(df_adj, columns=categorical_cols, drop_first=True)
                
                # Define columns to exclude
                exclude_cols = ['salary', 'case', 'id', 'sex', 'year', 'yrdeg', 'startyr']
                
                # Define independent variables (exclude unnecessary columns)
                X_adj = df_adj.drop(columns=exclude_cols)
                
                # Add constant term
                X_adj = sm.add_constant(X_adj)
                
                # Convert boolean columns to integers if needed
                bool_cols = X_adj.select_dtypes(include=['bool']).columns
                if len(bool_cols) > 0:
                    X_adj[bool_cols] = X_adj[bool_cols].astype(int)
                
                # Define dependent variable
                y_adj = df_adj['salary']
                
                # Fit multiple regression model
                model_adj = sm.OLS(y_adj, X_adj).fit()
                
                # Extract key results
                adj_sex_coef = model_adj.params['sex_binary']
                adj_sex_pval = model_adj.pvalues['sex_binary']
            except Exception as e:
                st.warning(f"Unable to run adjusted model: {str(e)}")
        
        # Create summary table
        summary_data = [
            {'Analysis': 'Raw Salary Difference', 'Finding': f"${raw_diff:.2f} {'higher for men' if raw_diff > 0 else 'higher for women'}", 'P-value': f"{p_val}", 'Significant': "Yes" if p_val < 0.05 else "No"},
            {'Analysis': 'Simple Regression', 'Finding': f"${sex_coef:.2f} {'higher for men' if sex_coef > 0 else 'higher for women'}", 'P-value': f"{sex_pval}", 'Significant': "Yes" if sex_pval < 0.05 else "No"}
        ]
        
        if adj_sex_coef is not None and adj_sex_pval is not None:
            summary_data.append(
                {'Analysis': 'Multiple Regression with Controls', 'Finding': f"${adj_sex_coef:.2f} {'higher for men' if adj_sex_coef > 0 else 'higher for women'}", 'P-value': f"{adj_sex_pval}", 'Significant': "Yes" if adj_sex_pval < 0.05 else "No"}
            )
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # Narrative summary
        st.write("### Detailed Findings")
        
        # Raw difference
        if p_val < 0.05:
            if raw_diff > 0:
                st.write(f"- **Raw Salary Difference**: Men earn ${raw_diff:.2f} more than women on average, which is statistically significant (p={p_val}).")
            else:
                st.write(f"- **Raw Salary Difference**: Women earn ${abs(raw_diff):.2f} more than men on average, which is statistically significant (p={p_val}).")
        else:
            st.write(f"- **Raw Salary Difference**: The raw difference in salary between men (${male_mean:.2f}) and women (${female_mean:.2f}) is not statistically significant (p={p_val}).")
        
        # Simple regression
        if sex_pval < 0.05:
            if sex_coef > 0:
                st.write(f"- **Simple Regression**: Sex is a significant predictor of salary. Men earn ${sex_coef:.2f} more than women on average (p={sex_pval}).")
            else:
                st.write(f"- **Simple Regression**: Sex is a significant predictor of salary. Women earn ${abs(sex_coef):.2f} more than men on average (p={sex_pval}).")
        else:
            st.write(f"- **Simple Regression**: Sex is not a significant predictor of salary (p={sex_pval}).")
        
        # Multiple regression
        if adj_sex_coef is not None and adj_sex_pval is not None:
            if adj_sex_pval < 0.05:
                if adj_sex_coef > 0:
                    st.write(f"- **Multiple Regression**: After controlling for other factors (e.g., rank, field, degree, administrative duties), men earn ${adj_sex_coef:.2f} more than women on average (p={adj_sex_pval}).")
                else:
                    st.write(f"- **Multiple Regression**: After controlling for other factors (e.g., rank, field, degree, administrative duties), women earn ${abs(adj_sex_coef):.2f} more than men on average (p={adj_sex_pval}).")
            else:
                st.write(f"- **Multiple Regression**: After controlling for other factors, there is no significant difference in salary between men and women (p={adj_sex_pval}).")
            
            # Compare raw vs. adjusted effect
            if (sex_pval < 0.05) != (adj_sex_pval < 0.05):
                if sex_pval < 0.05 and adj_sex_pval >= 0.05:
                    st.write("- **Important Note**: The sex difference that was significant in the simple model becomes non-significant after controlling for other factors. This suggests that the salary gap may be explained by other variables like rank, field, or administrative duties rather than direct sex discrimination.")
                elif sex_pval >= 0.05 and adj_sex_pval < 0.05:
                    st.write("- **Important Note**: A significant sex difference emerges after controlling for other factors, even though the raw difference is not significant. This suggests a potential suppression effect where other variables were masking the sex-based salary difference.")
            elif sex_pval < 0.05 and adj_sex_pval < 0.05:
                if abs(adj_sex_coef) < abs(sex_coef):
                    st.write(f"- **Important Note**: The sex difference remains significant but reduces after controlling for other factors. This suggests that part, but not all, of the salary gap can be explained by factors other than sex.")
                elif abs(adj_sex_coef) > abs(sex_coef):
                    st.write(f"- **Important Note**: The sex difference remains significant but increases from ${abs(sex_coef):.2f}  to ${abs(adj_sex_coef):.2f} after controlling for other factors. This suggests that the full extent of the sex gap is only revealed after accounting for other variables.")
        
        # Interaction effects summary (if available and significant)
        # try:
        #     if 'model_int_admin' in locals() and 'sex_admin' in model_int_admin.params.index:
        #         int_coef = model_int_admin.params['sex_admin']
        #         int_pval = model_int_admin.pvalues['sex_admin']
                
        #         if int_pval < 0.05:
        #             if int_coef > 0:
        #                 st.write(f"- **Interaction Effects**: There is a significant interaction between sex and administrative duties. Male administrators earn ${int_coef:.2f} more than would be expected from the main effects alone (p={int_pval:.4f}).")
        #             else:
        #                 st.write(f"- **Interaction Effects**: There is a significant interaction between sex and administrative duties. Male administrators earn ${abs(int_coef):.2f} less than would be expected from the main effects alone (p={int_pval:.4f}).")
        # except:
        #     pass  # Don't include interaction effects if not available
        
        # Generate additional findings based on what was significant in the analysis
        if 'rank' in df.columns:
            # Check rank distribution
            rank_counts = pd.crosstab(df['rank'], df['sex'])
            rank_props = pd.crosstab(df['rank'], df['sex'], normalize='index') * 100
            
            # Look for significant imbalances
            sig_rank_diffs = []
            for rank in rank_props.index:
                if 'M' in rank_props.columns and 'F' in rank_props.columns:
                    if abs(rank_props.loc[rank, 'M'] - rank_props.loc[rank, 'F']) > 20:  # 20% difference threshold
                        sig_rank_diffs.append((rank, rank_props.loc[rank, 'M'], rank_props.loc[rank, 'F']))
            
            if sig_rank_diffs:
                st.write("- **Rank Distribution**: There are notable differences in gender distribution across ranks.")
                for rank, male_pct, female_pct in sig_rank_diffs:
                    if male_pct > female_pct:
                        st.write(f"  * {rank} professors are {male_pct:.1f}% male and {female_pct:.1f}% female.")
                    else:
                        st.write(f"  * {rank} professors are {female_pct:.1f}% female and {male_pct:.1f}% male.")
                
                if adj_sex_pval is not None:
                    if adj_sex_pval >= 0.05 and sex_pval < 0.05:
                        st.write("  * These rank differences may help explain the raw salary gap between men and women.")
        
        # Overall conclusion
        st.write("### Overall Conclusion")
        
        if (p_val < 0.05 or sex_pval < 0.05) and (adj_sex_pval is None or adj_sex_pval < 0.05):
            st.write("Based on the analysis, there is evidence of sex bias in recent year(1995), even after controlling for relevant factors.")
        elif (p_val < 0.05 or sex_pval < 0.05) and adj_sex_pval is not None and adj_sex_pval >= 0.05:
            st.write("Based on the analysis, the apparent sex bias in raw faculty salaries can be largely explained by other factors such as rank, field, or administrative duties rather than direct sex discrimination.")
        else:
            st.write("Based on the analysis, there is no strong evidence of sex bias in current faculty salaries, either before or after controlling for other relevant factors.")
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.exception(e)
        st.write("Unable to generate comprehensive summary due to data issues.")

def run_analysis(uploaded_file):
    """Main function to run the Q1 current salaries analysis"""
    # Create tabs for different analysis components
    tabs = st.tabs(["Exploratory Analysis", "Statistical Tests", "Final Summary"])
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        with tabs[0]:
            exploratory_analysis(df)
        
        with tabs[1]:
            statistical_tests(df)
        
        with tabs[2]:
            summary(df)
    else:
        st.error("Unable to process data for Question 1. Please check file format and ensure it contains salary and sex data.")
