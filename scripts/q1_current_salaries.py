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
                f"{model_simple.pvalues['const']}", 
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
        control_vars = []
        if 'rank' in df.columns:
            control_vars.append('rank')
        if 'admin' in df.columns:
            control_vars.append('admin')
        if 'deg' in df.columns:
            control_vars.append('deg')
        if 'field' in df.columns:
            control_vars.append('field')
            
        if control_vars:
            # Create a copy of the data for adjusted analysis
            df_adj = df.copy()
            
            # Convert all columns to appropriate types before dummies
            for col in control_vars:
                df_adj[col] = df_adj[col].astype(str)
                
            # Creating dummy variables for categorical predictors
            df_adj = pd.get_dummies(df_adj, columns=control_vars, drop_first=True)
            
            # Get list of all columns except those we want to exclude
            exclude_cols = ['salary', 'case', 'id', 'sex', 'year', 'yrdeg', 'startyr']
            X_cols = [col for col in df_adj.columns if col not in exclude_cols and col != 'sex_binary']
            
            # Add sex_binary to the predictors
            X_cols.insert(0, 'sex_binary')
            
            # Make sure all columns in X_cols are numeric
            for col in X_cols:
                if df_adj[col].dtype == 'object':
                    try:
                        df_adj[col] = pd.to_numeric(df_adj[col])
                    except:
                        # If conversion fails, drop the column
                        X_cols.remove(col)
                        st.warning(f"Dropped column {col} because it couldn't be converted to numeric")
            
            # Define independent variables - only include numeric columns
            X_adj = df_adj[X_cols].select_dtypes(include=['number'])
            X_adj = sm.add_constant(X_adj)
            
            # Define dependent variable
            y_adj = df_adj['salary']
            
            # Fit multiple regression model
            model_adj = sm.OLS(y_adj, X_adj).fit()
            
            # Extract key results
            adj_sex_coef = model_adj.params['sex_binary']
            adj_sex_pval = model_adj.pvalues['sex_binary']
            adj_r_squared = model_adj.rsquared
            
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
            
            # # Check for multicollinearity using Variance Inflation Factors (VIF)
            # st.write("### Multicollinearity Check (VIF)")
            
            # # Calculate VIF for each predictor
            # vif_data = pd.DataFrame()
            # vif_data["Variable"] = X_adj.columns
            # vif_data["VIF"] = [variance_inflation_factor(X_adj.values, i) for i in range(X_adj.shape[1])]
            
            # # Sort by VIF value
            # vif_data = vif_data.sort_values('VIF', ascending=False)
            
            # # Add interpretation column
            # vif_data['Concern Level'] = pd.cut(
            #     vif_data['VIF'],
            #     bins=[0, 5, 10, float('inf')],
            #     labels=['Low', 'Moderate', 'High']
            # )
            
            # st.dataframe(vif_data)
            
            # st.write("""
            # **VIF Interpretation:**
            # - VIF < 5: Low concern of multicollinearity
            # - 5 ≤ VIF < 10: Moderate concern of multicollinearity
            # - VIF ≥ 10: High concern of multicollinearity
            # """)
            
            # 4. Interaction Effects
            st.write("### Interaction Effects")
            
            if 'admin' in df.columns:
                st.write("#### Interaction between Sex and Administrative Duties")
                
                # Creating a copy of the data for interaction analysis
                df_int = df_adj.copy()
                
                # Adding interaction term (sex × admin)
                admin_cols = [col for col in df_int.columns if col.startswith('admin_')]
                
                if admin_cols:
                    admin_col = admin_cols[0]  # Use the first admin dummy variable
                    df_int['sex_admin'] = df_int['sex_binary'] * df_int[admin_col]
                    
                    # Define independent variables for the model
                    X_cols_int = X_cols.copy()
                    X_cols_int.append('sex_admin')
                    
                    # Define X and y, ensure all types are numeric
                    X_int = df_int[X_cols_int].select_dtypes(include=['number'])
                    X_int = sm.add_constant(X_int)
                    y_int = df_int['salary']
                    
                    # Fit interaction model
                    model_int_admin = sm.OLS(y_int, X_int).fit()
                    
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
                df_field = df_adj.copy()
                
                # Find field dummy variables
                field_cols = [col for col in df_field.columns if col.startswith('field_')]
                
                if field_cols:
                    # Add interaction terms for each field
                    for field_col in field_cols:
                        df_field[f'sex_{field_col}'] = df_field['sex_binary'] * df_field[field_col]
                    
                    # Define independent variables including field interactions
                    X_cols_field = X_cols.copy()
                    X_cols_field.extend([f'sex_{col}' for col in field_cols])
                    
                    # Define X and y, ensuring all are numeric
                    X_field = df_field[X_cols_field].select_dtypes(include=['number'])
                    X_field = sm.add_constant(X_field)
                    y_field = df_field['salary']
                    
                    # Fit field interaction model
                    model_int_field = sm.OLS(y_field, X_field).fit()
                    
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
                                p_val = float(inter['P-value'])
                                
                                if coef > 0:
                                    st.write(f"- For {field} field, men earn ${coef:.2f} more than would be expected from the main effects alone (p={p_val}).")
                                else:
                                    st.write(f"- For {field} field, men earn ${abs(coef):.2f} less than would be expected from the main effects alone (p={p_val}).")
                        else:
                            st.write("**No significant interactions found between sex and field.**")
                        
                        # Display full model results in an expander
                        with st.expander("View Full Field Interaction Model Results"):
                            st.text(model_int_field.summary().as_text())
            
            # # 6. Stratified Analysis
            # st.write("### Stratified Analysis")
            
            # # Regression by rank
            # if 'rank' in df.columns:
            #     st.write("#### Regression Analysis by Rank")
                
            #     rank_models = {}
            #     rank_results = []
                
            #     for rank in df['rank'].unique():
            #         # Filter data for this rank
            #         subset = df[df['rank'] == rank].copy()
            #         subset['sex_binary'] = subset['sex'].map({'F': 0, 'M': 1})
                    
            #         # Only proceed if we have enough data for both sexes
            #         if len(subset) > 5 and 'M' in subset['sex'].values and 'F' in subset['sex'].values:
            #             # Fit simple model
            #             X_rank = sm.add_constant(subset['sex_binary'])
            #             y_rank = subset['salary']
            #             model_rank = sm.OLS(y_rank, X_rank).fit()
                        
            #             # Store model
            #             rank_models[rank] = model_rank
                        
            #             # Extract key results
            #             coef = model_rank.params['sex_binary']
            #             p_val = model_rank.pvalues['sex_binary']
            #             r2 = model_rank.rsquared
            #             n = len(subset)
                        
            #             # Store results for display
            #             rank_results.append({
            #                 'Rank': rank,
            #                 'Sex Coefficient': f"{coef:.2f}",
            #                 'P-value': f"{p_val}",
            #                 'R-squared': f"{r2:.4f}",
            #                 'Sample Size': n,
            #                 'Significant': "Yes" if p_val < 0.05 else "No"
            #             })
                
            #     if rank_results:
            #         # Convert to DataFrame and display
            #         rank_df = pd.DataFrame(rank_results)
            #         st.dataframe(rank_df)
                    
            #         # Interpretation
            #         sig_ranks = [r for r in rank_results if r['Significant'] == "Yes"]
            #         if sig_ranks:
            #             st.write("**Significant sex differences within ranks:**")
            #             for r in sig_ranks:
            #                 rank = r['Rank']
            #                 coef = float(r['Sex Coefficient'])
            #                 p_val = float(r['P-value'].replace(',', '.'))
                            
            #                 if coef > 0:
            #                     st.write(f"- For {rank} professors, men earn ${coef:.2f} more than women on average (p={p_val}).")
            #                 else:
            #                     st.write(f"- For {rank} professors, women earn ${abs(coef):.2f} more than men on average (p={p_val}).")
            #         else:
            #             st.write("**No significant sex differences found within any rank.**")
                    
            #         # Display each model in expanders
            #         for rank, model in rank_models.items():
            #             with st.expander(f"View Full Model Results for {rank} Professors"):
            #                 st.text(model.summary().as_text())
            #     else:
            #         st.warning("Not enough data to run regression by rank.")
            
            # # Regression by field
            # if 'field' in df.columns:
            #     st.write("#### Regression Analysis by Field")
                
            #     field_models = {}
            #     field_results = []
                
            #     for field in df['field'].unique():
            #         # Filter data for this field
            #         subset = df[df['field'] == field].copy()
            #         subset['sex_binary'] = subset['sex'].map({'F': 0, 'M': 1})
                    
            #         # Only proceed if we have enough data for both sexes
            #         if len(subset) > 5 and 'M' in subset['sex'].values and 'F' in subset['sex'].values:
            #             # Fit simple model
            #             X_field = sm.add_constant(subset['sex_binary'])
            #             y_field = subset['salary']
            #             model_field = sm.OLS(y_field, X_field).fit()
                        
            #             # Store model
            #             field_models[field] = model_field
                        
            #             # Extract key results
            #             coef = model_field.params['sex_binary']
            #             p_val = model_field.pvalues['sex_binary']
            #             r2 = model_field.rsquared
            #             n = len(subset)
                        
            #             # Store results for display
            #             field_results.append({
            #                 'Field': field,
            #                 'Sex Coefficient': f"{coef:.2f}",
            #                 'P-value': f"{p_val}",
            #                 'R-squared': f"{r2:.4f}",
            #                 'Sample Size': n,
            #                 'Significant': "Yes" if p_val < 0.05 else "No"
            #             })
                
            #     if field_results:
            #         # Convert to DataFrame and display
            #         field_df = pd.DataFrame(field_results)
            #         st.dataframe(field_df)
                    
            #         # Interpretation
            #         sig_fields = [f for f in field_results if f['Significant'] == "Yes"]
            #         if sig_fields:
            #             st.write("**Significant sex differences within fields:**")
            #             for f in sig_fields:
            #                 field = f['Field']
            #                 coef = float(f['Sex Coefficient'])
            #                 p_val = float(f['P-value'].replace(',', '.'))
                            
            #                 if coef > 0:
            #                     st.write(f"- In the {field} field, men earn ${coef:.2f} more than women on average (p={p_val}).")
            #                 else:
            #                     st.write(f"- In the {field} field, women earn ${abs(coef):.2f} more than men on average (p={p_val}).")
            #         else:
            #             st.write("**No significant sex differences found within any field.**")
                    
            #         # Display each model in expanders
            #         for field, model in field_models.items():
            #             with st.expander(f"View Full Model Results for {field} Field"):
            #                 st.text(model.summary().as_text())
            #     else:
            #         st.warning("Not enough data to run regression by field.")
            
            # # Regression by administrative role
            # if 'admin' in df.columns:
            #     st.write("#### Regression Analysis by Administrative Role")
                
            #     admin_models = {}
            #     admin_results = []
                
            #     for admin_val in [0, 1]:
            #         # Create label for admin role
            #         admin_label = "With Admin Duties" if admin_val == 1 else "Without Admin Duties"
                    
            #         # Filter data for this admin status
            #         subset = df[df['admin'] == admin_val].copy()
            #         subset['sex_binary'] = subset['sex'].map({'F': 0, 'M': 1})
                    
            #         # Only proceed if we have enough data for both sexes
            #         if len(subset) > 5 and 'M' in subset['sex'].values and 'F' in subset['sex'].values:
            #             # Fit simple model
            #             X_admin = sm.add_constant(subset['sex_binary'])
            #             y_admin = subset['salary']
            #             model_admin = sm.OLS(y_admin, X_admin).fit()
                        
            #             # Store model
            #             admin_models[admin_label] = model_admin
                        
            #             # Extract key results
            #             coef = model_admin.params['sex_binary']
            #             p_val = model_admin.pvalues['sex_binary']
            #             r2 = model_admin.rsquared
            #             n = len(subset)
                        
            #             # Store results for display
            #             admin_results.append({
            #                 'Admin Role': admin_label,
            #                 'Sex Coefficient': f"{coef:.2f}",
            #                 'P-value': f"{p_val}",
            #                 'R-squared': f"{r2:.4f}",
            #                 'Sample Size': n,
            #                 'Significant': "Yes" if p_val < 0.05 else "No"
            #             })
                
            #     if admin_results:
            #         # Convert to DataFrame and display
            #         admin_df = pd.DataFrame(admin_results)
            #         st.dataframe(admin_df)
                    
            #         # Interpretation
            #         sig_admin = [a for a in admin_results if a['Significant'] == "Yes"]
            #         if sig_admin:
            #             st.write("**Significant sex differences by administrative role:**")
            #             for a in sig_admin:
            #                 role = a['Admin Role']
            #                 coef = float(a['Sex Coefficient'])
            #                 p_val = float(a['P-value'].replace(',', '.'))
                            
            #                 if coef > 0:
            #                     st.write(f"- For faculty {role.lower()}, men earn ${coef:.2f} more than women on average (p={p_val}).")
            #                 else:
            #                     st.write(f"- For faculty {role.lower()}, women earn ${abs(coef):.2f} more than men on average (p={p_val}).")
            #         else:
            #             st.write("**No significant sex differences found within administrative roles.**")
                    
            #         # Display each model in expanders
            #         for role, model in admin_models.items():
            #             with st.expander(f"View Full Model Results for Faculty {role}"):
            #                 st.text(model.summary().as_text())
            #     else:
            #         st.warning("Not enough data to run regression by administrative role.")
            
            # 7. Residual analysis
            st.write("### Residual Analysis")
        
            # Get residuals from the adjusted model
            residuals = model_adj.resid
            
            # Center the plots using columns
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:  # Center column
                # Histogram of residuals - smaller size
                fig_res1, ax_res1 = plt.subplots(figsize=(6, 4))
                ax_res1.hist(residuals, bins=30, edgecolor='black')
                ax_res1.set_title('Histogram of Residuals')
                ax_res1.set_xlabel('Residuals')
                ax_res1.set_ylabel('Frequency')
                ax_res1.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_res1)
            
            # Shapiro-Wilk normality test
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
            
            # Q-Q plot of residuals - smaller and centered
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:  # Center column
                fig_res2, ax_res2 = plt.subplots(figsize=(6, 4))
                sm.qqplot(residuals, line='45', ax=ax_res2)
                ax_res2.set_title('Q-Q Plot of Residuals')
                ax_res2.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_res2)
            
            st.write("""
            **Q-Q Plot Interpretation:**
            - Points following the diagonal line suggest normally distributed residuals
            - Deviations from the line indicate departures from normality
            - S-shaped patterns may indicate skewness
            - Heavy tails at the ends suggest excess kurtosis
            """)

            
            # 8. Robust Standard Errors
            st.write("### Robust Standard Errors")

            try:
                # Generate robust covariance results for the adjusted model
                robust_results = model_adj.get_robustcov_results(cov_type='HC1')
                
                # Check if robust_results.params is a numpy array or a pandas Series
                if hasattr(robust_results.params, 'index'):
                    # It's a pandas Series with an index
                    if 'sex_binary' in robust_results.params.index:
                        # Extract sex coefficient and p-value with robust standard errors
                        robust_sex_coef = robust_results.params['sex_binary']
                        robust_sex_pval = robust_results.pvalues['sex_binary']
                        
                        # Create a comparison table
                        robust_comparison = pd.DataFrame({
                            'Method': ['Regular OLS', 'Robust Standard Errors'],
                            'Sex Coefficient': [
                                f"{adj_sex_coef:.2f}",
                                f"{robust_sex_coef:.2f}"
                            ],
                            'Standard Error': [
                                f"{model_adj.bse['sex_binary']:.2f}",
                                f"{robust_results.bse['sex_binary']:.2f}"
                            ],
                            'P-value': [
                                f"{adj_sex_pval}",
                                f"{robust_sex_pval}"
                            ],
                            'Significant': [
                                "Yes" if adj_sex_pval < 0.05 else "No",
                                "Yes" if robust_sex_pval < 0.05 else "No"
                            ]
                        })
                        
                        st.write("**Comparison of Regular and Robust Standard Errors for Sex Effect:**")
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
                        st.warning("Sex variable not found in robust model parameters.")
                else:
                    # It's a numpy array
                    # Find the index of 'sex_binary' in the original model
                    if 'sex_binary' in model_adj.params.index:
                        sex_binary_idx = list(model_adj.params.index).index('sex_binary')
                        
                        # Get the robust coefficient and p-value using the index
                        robust_sex_coef = robust_results.params[sex_binary_idx]
                        robust_sex_pval = robust_results.pvalues[sex_binary_idx]
                        robust_sex_bse = robust_results.bse[sex_binary_idx]
                        
                        # Create a comparison table
                        robust_comparison = pd.DataFrame({
                            'Method': ['Regular OLS', 'Robust Standard Errors'],
                            'Sex Coefficient': [
                                f"{adj_sex_coef:.2f}",
                                f"{robust_sex_coef:.2f}"
                            ],
                            'Standard Error': [
                                f"{model_adj.bse['sex_binary']:.2f}",
                                f"{robust_sex_bse:.2f}"
                            ],
                            'P-value': [
                                f"{adj_sex_pval}",
                                f"{robust_sex_pval}"
                            ],
                            'Significant': [
                                "Yes" if adj_sex_pval < 0.05 else "No",
                                "Yes" if robust_sex_pval < 0.05 else "No"
                            ]
                        })
                        
                        st.write("**Comparison of Regular and Robust Standard Errors for Sex Effect:**")
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
                        st.warning("Sex variable not found in model parameters.")
                
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
    st.subheader("Question 1 Summary: Sex Bias in Recent year")
    
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
        control_vars = []
        if 'rank' in df.columns:
            control_vars.append('rank')
        if 'admin' in df.columns:
            control_vars.append('admin')
        if 'deg' in df.columns:
            control_vars.append('deg')
        if 'field' in df.columns:
            control_vars.append('field')
            
        adj_sex_coef = None
        adj_sex_pval = None
        
        if control_vars:
            try:
                # Create a copy of data for adjusted analysis
                df_adj = df.copy()
                
                # Convert all columns to appropriate types before dummies
                for col in control_vars:
                    df_adj[col] = df_adj[col].astype(str)
                
                # Creating dummy variables for categorical predictors
                df_adj = pd.get_dummies(df_adj, columns=control_vars, drop_first=True)
                
                # Get list of all columns except those we want to exclude
                exclude_cols = ['salary', 'case', 'id', 'sex', 'year', 'yrdeg', 'startyr']
                X_cols = [col for col in df_adj.columns if col not in exclude_cols and col != 'sex_binary']
                
                # Add sex_binary to the predictors
                X_cols.insert(0, 'sex_binary')
                
                # Make sure all columns in X_cols are numeric
                for col in X_cols:
                    if df_adj[col].dtype == 'object':
                        try:
                            df_adj[col] = pd.to_numeric(df_adj[col])
                        except:
                            # If conversion fails, drop the column
                            X_cols.remove(col)
                
                # Define independent variables - only include numeric columns
                X_adj = df_adj[X_cols].select_dtypes(include=['number'])
                X_adj = sm.add_constant(X_adj)
                
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
                    st.write(f"- **Important Note**: The sex difference remains significant but reduces from ${abs(sex_coef):.2f} to ${abs(adj_sex_coef):.2f} after controlling for other factors. This suggests that part, but not all, of the salary gap can be explained by factors other than sex.")
                elif abs(adj_sex_coef) > abs(sex_coef):
                    st.write(f"- **Important Note**: The sex difference remains significant but increases from ${abs(sex_coef):.2f} to ${abs(adj_sex_coef):.2f} after controlling for other factors. This suggests that the full extent of the sex gap is only revealed after accounting for other variables.")
        
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
                st.write("- **Rank Distribution**: There are notable differences in gender distribution across ranks:")
                for rank, male_pct, female_pct in sig_rank_diffs:
                    if male_pct > female_pct:
                        st.write(f"  * {rank} professors are {male_pct:.1f}% male and {female_pct:.1f}% female")
                    else:
                        st.write(f"  * {rank} professors are {female_pct:.1f}% female and {male_pct:.1f}% male")
                
                if adj_sex_pval is not None:
                    if adj_sex_pval >= 0.05 and sex_pval < 0.05:
                        st.write("  * These rank differences may help explain the raw salary gap between men and women")
        
        # Overall conclusion
        st.write("### Overall Conclusion")
        
        if (p_val < 0.05 or sex_pval < 0.05) and (adj_sex_pval is None or adj_sex_pval < 0.05):
            st.write("Based on the analysis, there is evidence of sex bias in current faculty salaries, even after controlling for relevant factors.")
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
