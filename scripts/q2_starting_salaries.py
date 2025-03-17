import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import cpi

# Define palette for consistent colors
custom_palette = {'M': 'skyblue', 'F': 'lightcoral'}

def load_data(uploaded_file):
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\\s+')
        
        # Following the exact methodology from the reference stats_q2.py
        # Basic preprocessing
        try:
            # Filter for only those rows where startyr equals year (the year they were hired)
            df = df[df['startyr'] == df['year']]
            
            # Convert 2-digit years to 4-digits (add 1900)
            df['startyr'] = df['startyr'].astype(int) + 1900
            df['yrdeg'] = df['yrdeg'].astype(int) + 1900
            df['year'] = df['year'].astype(int) + 1900
            
            # Calculate experience
            df['exp'] = df['startyr'] - df['yrdeg']
            df['exp_coded'] = df['exp'] >= 0
            
            # Ensure sex column is correct
            df['sex'] = df['sex'].astype(str)
            
            # Check if we have enough data
            if len(df) < 10:
                st.error("Too few records match the criteria where startyr == year. Please check your dataset.")
                return None
                
            # Calculate inflation-adjusted salary
            try:
                # Update the CPI data
                cpi.update()

                # Get unique start years and compute inflation factors for each
                unique_years = df['startyr'].unique()
                inflation_factors = {year: cpi.inflate(1, year, to=1995) for year in unique_years}

                # Adjust salaries using vectorized multiplication
                df['inf_salary'] = df['salary'] * df['startyr'].map(inflation_factors)

            except Exception as e:
                # st.warning(f"Unable to adjust for inflation: {str(e)}. Using raw salary values.")
                df['inf_salary'] = df['salary']
                
            # Center startyr (for regression to reduce collinearity)
            df['startyr_centered'] = df['startyr'] - df['startyr'].mean()
            
            return df
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            st.warning("Attempting alternative data processing approach...")
            
            # Alternative approach - minimal preprocessing
            # Convert 2-digit years to 4-digits if needed
            if 'year' in df.columns and df['year'].max() < 100:
                df['year'] = df['year'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
            if 'startyr' in df.columns and df['startyr'].max() < 100:
                df['startyr'] = df['startyr'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
            if 'yrdeg' in df.columns and df['yrdeg'].max() < 100:
                df['yrdeg'] = df['yrdeg'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
                
            # Create minimal variables
            df['inf_salary'] = df['salary'].copy()  # Fallback without inflation adjustment
            df['exp'] = 0
            if 'yrdeg' in df.columns and 'startyr' in df.columns:
                df['exp'] = df['startyr'] - df['yrdeg']
            df['exp_coded'] = df['exp'] >= 0
            df['startyr_centered'] = 0
            if 'startyr' in df.columns:
                df['startyr_centered'] = df['startyr'] - df['startyr'].mean()
                
            return df
            
    except Exception as e:
        st.error(f"Error processing file for Question 2: {str(e)}")
        st.info("Please ensure your data file contains the required columns: id, startyr, year, yrdeg, sex, salary")
        return None

def exploratory_analysis(df):
    st.subheader("Exploratory Data Analysis")
    
    # Split data by sex
    df_M = df[df['sex'] == 'M']
    df_F = df[df['sex'] == 'F']
    
    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.write("Male Faculty Summary:")
        st.dataframe(df_M[['salary', 'inf_salary', 'startyr', 'exp']].describe())
    
    with col2:
        st.write("Female Faculty Summary:")
        st.dataframe(df_F[['salary', 'inf_salary', 'startyr', 'exp']].describe())
    
    # Count display
    st.write(f"Count Male: {len(df_M)}, Count Female: {len(df_F)}")
    
    # Hiring year stats
    median_hiring_male = df_M['startyr'].median()
    mode_hiring_male = df_M['startyr'].mode()[0]
    median_hiring_female = df_F['startyr'].median()
    mode_hiring_female = df_F['startyr'].mode()[0]
    
    hiring_stats = pd.DataFrame({
        'Statistic': ['Median Year', 'Mode Year'],
        'Male': [median_hiring_male, mode_hiring_male],
        'Female': [median_hiring_female, mode_hiring_female]
    })
    st.write("Hiring Year Statistics:")
    st.dataframe(hiring_stats)
    
    # Visualizations
    st.write("### Salary Visualizations")
    
    # Create two columns for the first row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot of Start Year vs. Salary
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(df_M['startyr'], df_M['salary'], marker='o', color='skyblue', label='Male')
        ax1.scatter(df_F['startyr'], df_F['salary'], marker='o', color='lightcoral', label='Female')
        ax1.set_xlabel('Start Year')
        ax1.set_ylabel('Salary')
        ax1.set_title('Start Year vs. Salary by Sex')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
    
    with col2:
        # Box plot of starting salaries by sex
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        bp = ax2.boxplot([df_M['salary'], df_F['salary']], labels=['Male', 'Female'], patch_artist=True)
        for box, color in zip(bp['boxes'], ['skyblue', 'lightcoral']):
            box.set(facecolor=color, alpha=0.8)
        ax2.set_xlabel('Sex')
        ax2.set_ylabel('Starting Salary')
        ax2.set_title('Distribution of Starting Salaries by Sex')
        ax2.grid(True)
        st.pyplot(fig2)
    
    # Create two columns for the second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Calculate the average salary for males and females for each start year
        avg_salary_by_year_male = df_M.groupby('startyr')['salary'].mean()
        avg_salary_by_year_female = df_F.groupby('startyr')['salary'].mean()
        
        # Line plot of average salary trends by start year
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.plot(avg_salary_by_year_male.index, avg_salary_by_year_male.values, 
                 color='skyblue', linestyle='-', marker='o', label='Male')
        ax3.plot(avg_salary_by_year_female.index, avg_salary_by_year_female.values, 
                 color='lightcoral', linestyle='-', marker='o', label='Female')
        ax3.set_xlabel('Start Year')
        ax3.set_ylabel('Average Salary')
        ax3.set_title('Average Salary Trends by Start Year')
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)
    
    with col4:
        # Line plot of inflation-adjusted average salaries
        # avg_inf_male = df_M.groupby('startyr')['inf_salary'].mean()
        # avg_inf_female = df_F.groupby('startyr')['inf_salary'].mean()
        
        # fig4, ax4 = plt.subplots(figsize=(8, 6))
        # ax4.plot(avg_inf_male.index, avg_inf_male.values, 
        #          color='skyblue', linestyle='-', marker='o', label='Male')
        # ax4.plot(avg_inf_female.index, avg_inf_female.values, 
        #          color='lightcoral', linestyle='-', marker='o', label='Female')
        # ax4.set_xlabel('Start Year')
        # ax4.set_ylabel('Inflation-Adjusted Salary (1995 $)')
        # ax4.set_title('Inflation-Adjusted Salary Trends by Start Year')
        # ax4.legend()
        # ax4.grid(True)
        # st.pyplot(fig4)
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        bp = ax4.boxplot([df_M['inf_salary'], df_F['inf_salary']], labels=['Male', 'Female'], patch_artist=True)
        for box, color in zip(bp['boxes'], ['skyblue', 'lightcoral']):
            box.set(facecolor=color, alpha=0.8)
        ax4.set_xlabel('Sex')
        ax4.set_ylabel('Inflation-Adjusted Starting Salary (1995 $)')
        ax4.set_title('Distribution of Inflation-Adjusted Starting Salaries by Sex')
        ax4.grid(True)
        st.pyplot(fig4)
    
    # # Distribution of inflated salaries - make smaller and centered
    # col1, col2, col3 = st.columns([1, 2, 1])
    
    # with col2:  # Center column
    #     fig5, ax5 = plt.subplots(figsize=(6, 4))
    #     bp = ax5.boxplot([df_M['inf_salary'], df_F['inf_salary']], labels=['Male', 'Female'], patch_artist=True)
    #     for box, color in zip(bp['boxes'], ['skyblue', 'lightcoral']):
    #         box.set(facecolor=color, alpha=0.8)
    #     ax5.set_xlabel('Sex')
    #     ax5.set_ylabel('Inflation-Adjusted Starting Salary (1995 $)')
    #     ax5.set_title('Distribution of Inflation-Adjusted Starting Salaries by Sex')
    #     ax5.grid(True)
    #     st.pyplot(fig5)

def statistical_tests(df):
    st.subheader("Statistical Tests")
    
    # Split data by sex
    df_M = df[df['sex'] == 'M']
    df_F = df[df['sex'] == 'F']
    
    # Two-Sample T-Test Function
    def twoSampleTTest(col1, col2, alpha=0.05):
        result = stats.ttest_ind(col1, col2, equal_var=False)  # not assuming equal variance
        t_stat, p_val = result.statistic, result.pvalue
        dof = result.df
        critical_val = stats.t.ppf(1 - alpha/2, dof)
        CI = result.confidence_interval()
        if np.abs(t_stat) > critical_val:
            reject_h0 = True
        else:
            reject_h0 = False
        return t_stat, p_val, dof, critical_val, CI, reject_h0
    
    # Raw salary t-test
    st.write("### T-Test on Raw Starting Salaries")
    t_stat, p_val, dof, critical_val, CI, reject_h0 = twoSampleTTest(df_M['salary'], df_F['salary'])
    
    ttest_raw_results = pd.DataFrame({
        'Statistic': ['Sample Mean Difference', 'T-statistic', 'P-value', 'Degrees of Freedom', 
                      'Critical Value', 'CI Lower', 'CI Upper', 'Reject H0'],
        'Value': [
            f"${df_M['salary'].mean() - df_F['salary'].mean():.2f}",
            f"{t_stat:.4f}",
            f"{p_val:.4f}",
            f"{dof:.2f}",
            f"{critical_val:.4f}",
            f"${CI[0]:.2f}",
            f"${CI[1]:.2f}",
            "Yes" if reject_h0 else "No"
        ]
    })
    
    st.dataframe(ttest_raw_results)
    
    if reject_h0:
        if df_M['salary'].mean() > df_F['salary'].mean():
            st.write(f"**Interpretation**: There is a statistically significant difference in starting salaries. On average, men receive ${df_M['salary'].mean() - df_F['salary'].mean():.2f} more than women (p={p_val:.4f}).")
        else:
            st.write(f"**Interpretation**: There is a statistically significant difference in starting salaries. On average, women receive ${df_F['salary'].mean() - df_M['salary'].mean():.2f} more than men (p={p_val:.4f}).")
    else:
        st.write(f"**Interpretation**: There is no statistically significant difference in raw starting salaries between men and women (p={p_val:.4f}).")
    
    # Inflation-adjusted salary t-test
    st.write("### T-Test on Inflation-Adjusted Starting Salaries")
    t_stat, p_val, dof, critical_val, CI, reject_h0 = twoSampleTTest(df_M['inf_salary'], df_F['inf_salary'])
    
    ttest_inf_results = pd.DataFrame({
        'Statistic': ['Sample Mean Difference', 'T-statistic', 'P-value', 'Degrees of Freedom', 
                      'Critical Value', 'CI Lower', 'CI Upper', 'Reject H0'],
        'Value': [
            f"${df_M['inf_salary'].mean() - df_F['inf_salary'].mean():.2f}",
            f"{t_stat:.4f}",
            f"{p_val:.4f}",
            f"{dof:.2f}",
            f"{critical_val:.4f}",
            f"${CI[0]:.2f}",
            f"${CI[1]:.2f}",
            "Yes" if reject_h0 else "No"
        ]
    })
    
    st.dataframe(ttest_inf_results)
    
    if reject_h0:
        if df_M['inf_salary'].mean() > df_F['inf_salary'].mean():
            st.write(f"**Interpretation**: After adjusting for inflation, there is a statistically significant difference in starting salaries. On average, men receive ${df_M['inf_salary'].mean() - df_F['inf_salary'].mean():.2f} more than women in 1995 dollars (p={p_val:.4f}).")
        else:
            st.write(f"**Interpretation**: After adjusting for inflation, there is a statistically significant difference in starting salaries. On average, women receive ${df_F['inf_salary'].mean() - df_M['inf_salary'].mean():.2f} more than men in 1995 dollars (p={p_val:.4f}).")
    else:
        st.write(f"**Interpretation**: After adjusting for inflation, there is no statistically significant difference in starting salaries between men and women (p={p_val:.4f}).")

    # Center the startyr variable to reduce multicollinearity
    df['startyr_centered'] = df['startyr'] - df['startyr'].mean()
    
    # Regression Models
    st.write("### Regression Models")
    
    # Create dummy variables
    df_dummy = pd.get_dummies(df, columns=['sex', 'rank', 'field', 'deg'], drop_first=True)
    
    # Display regression results in a more structured format
    st.write("#### Model 1: Sex Only")
    try:
        model1 = smf.ols(formula="inf_salary ~ C(sex)", data=df).fit()
        
        # Extract and display key results
        coef = model1.params.get('C(sex)[T.M]', 0)  # Get the coefficient for male
        p_val = model1.pvalues.get('C(sex)[T.M]', 1)
        r_squared = model1.rsquared
        adj_r_squared = model1.rsquared_adj
        f_stat = model1.fvalue
        f_pvalue = model1.f_pvalue
        
        # Create a DataFrame for clean display of model statistics
        model1_stats = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F p-value', 'AIC', 'BIC', 'Observations'],
            'Value': [
                f"{r_squared:.4f}",
                f"{adj_r_squared:.4f}",
                f"{f_stat:.4f}",
                f"{f_pvalue:.4f}",
                f"{model1.aic:.4f}",
                f"{model1.bic:.4f}",
                f"{model1.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model1_stats)
        
        # Create a DataFrame for the coefficients table
        coef_table = pd.DataFrame({
            'Variable': ['Intercept', 'Sex (Male=1)'],
            'Coefficient': [
                f"{model1.params['Intercept']:.2f}", 
                f"{coef:.2f}"
            ],
            'Std Error': [
                f"{model1.bse['Intercept']:.2f}", 
                f"{model1.bse.get('C(sex)[T.M]', 0):.2f}"
            ],
            't-value': [
                f"{model1.tvalues['Intercept']:.4f}", 
                f"{model1.tvalues.get('C(sex)[T.M]', 0):.4f}"
            ],
            'P-value': [
                f"{model1.pvalues['Intercept']:.4f}", 
                f"{p_val:.4f}"
            ],
            'CI Lower 95%': [
                f"{model1.conf_int().loc['Intercept'][0]:.2f}",
                f"{model1.conf_int().loc['C(sex)[T.M]'][0] if 'C(sex)[T.M]' in model1.conf_int().index else 0:.2f}"
            ],
            'CI Upper 95%': [
                f"{model1.conf_int().loc['Intercept'][1]:.2f}",
                f"{model1.conf_int().loc['C(sex)[T.M]'][1] if 'C(sex)[T.M]' in model1.conf_int().index else 0:.2f}"
            ]
        })
        
        st.write("**Coefficients:**")
        st.dataframe(coef_table)
        
        # Summary of sex effect
        model1_results = pd.DataFrame({
            'Statistic': ['Sex Effect (Male)', 'P-value', 'R-squared'],
            'Value': [f"${coef:.2f}", f"{p_val:.4f}", f"{r_squared:.4f}"]
        })
        
        st.write("**Sex Effect Summary:**")
        st.dataframe(model1_results)
        
        # Interpretation
        if p_val < 0.05:
            if coef > 0:
                st.write(f"**Interpretation**: Men earn ${coef:.2f} more than women on average (p={p_val:.4f}).")
            else:
                st.write(f"**Interpretation**: Women earn ${abs(coef):.2f} more than men on average (p={p_val:.4f}).")
        else:
            st.write(f"**Interpretation**: No significant difference in salary between men and women (p={p_val:.4f}).")
        
        # Display full model results in an expander
        with st.expander("View Full Model 1 Results"):
            st.text(model1.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running simple regression model: {str(e)}")
    
    # Full model with controls
    st.write("#### Model 2: Controlling for Field, Rank, Degree, etc.")
    try:
        model2 = smf.ols(formula="inf_salary ~ C(sex) + C(field) + C(rank) + C(deg) + admin + startyr_centered + yrdeg + exp_coded", data=df).fit()
        
        # Extract key results
        coef = model2.params.get('C(sex)[T.M]', 0)  # Get the coefficient for male
        p_val = model2.pvalues.get('C(sex)[T.M]', 1)
        r_squared = model2.rsquared
        adj_r_squared = model2.rsquared_adj
        f_stat = model2.fvalue
        f_pvalue = model2.f_pvalue
        ci = model2.conf_int().loc['C(sex)[T.M]'] if 'C(sex)[T.M]' in model2.conf_int().index else [np.nan, np.nan]
        
        # Create a DataFrame for model statistics
        model2_stats = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F p-value', 'AIC', 'BIC', 'Observations'],
            'Value': [
                f"{r_squared:.4f}",
                f"{adj_r_squared:.4f}",
                f"{f_stat:.4f}",
                f"{f_pvalue:.4f}",
                f"{model2.aic:.4f}",
                f"{model2.bic:.4f}",
                f"{model2.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model2_stats)
        
        # Create a table for all coefficients
        coef_data = []
        for var in model2.params.index:
            var_name = var
            if var == 'Intercept':
                var_name = 'Intercept'
            elif var == 'C(sex)[T.M]':
                var_name = 'Sex (Male=1)'
            elif var == 'startyr_centered':
                var_name = 'Start Year (centered)'
            elif var == 'exp_coded':
                var_name = 'Experience'
            elif var == 'admin':
                var_name = 'Administrative Duties'
            elif var == 'yrdeg':
                var_name = 'Year of Degree'
            
            coef_data.append({
                'Variable': var_name,
                'Coefficient': f"{model2.params[var]:.2f}",
                'Std Error': f"{model2.bse[var]:.2f}",
                't-value': f"{model2.tvalues[var]:.4f}",
                'P-value': f"{model2.pvalues[var]:.4f}",
                'Significant': "Yes" if model2.pvalues[var] < 0.05 else "No"
            })
        
        coef_df = pd.DataFrame(coef_data)
        
        # Show only significant variables first
        sig_vars = coef_df[coef_df['Significant'] == 'Yes'].copy()
        if not sig_vars.empty:
            st.write("**Significant Variables:**")
            st.dataframe(sig_vars)
        
        # Then show all coefficients in an expander
        with st.expander("View All Coefficients"):
            st.dataframe(coef_df)
        
        # Create a DataFrame specifically for the sex effect
        model2_results = pd.DataFrame({
            'Statistic': ['Adjusted Sex Effect (Male)', 'P-value', 'R-squared', 'CI Lower', 'CI Upper'],
            'Value': [f"${coef:.2f}", f"{p_val:.4f}", f"{r_squared:.4f}", f"${ci[0]:.2f}", f"${ci[1]:.2f}"]
        })
        
        st.write("**Sex Effect Summary (Model 2):**")
        st.dataframe(model2_results)
        
        # Interpretation
        if p_val < 0.05:
            if coef > 0:
                st.write(f"**Interpretation**: After controlling for other factors, men earn ${coef:.2f} more than women on average (p={p_val:.4f}).")
            else:
                st.write(f"**Interpretation**: After controlling for other factors, women earn ${abs(coef):.2f} more than men on average (p={p_val:.4f}).")
        else:
            st.write(f"**Interpretation**: After controlling for other factors, no significant difference in salary between men and women (p={p_val:.4f}).")
        
        # Display full model results in an expander
        with st.expander("View Full Model 2 Results"):
            st.text(model2.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running full regression model: {str(e)}")
    
    # Interaction models
    st.write("#### Model 3: Sex × Rank Interaction")
    try:
        model3 = smf.ols(formula="inf_salary ~ C(sex) + C(field) + C(sex)*C(rank) + C(deg) + admin + startyr_centered", data=df).fit()
        
        # Model statistics
        model3_stats = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F p-value', 'AIC', 'BIC', 'Observations'],
            'Value': [
                f"{model3.rsquared:.4f}",
                f"{model3.rsquared_adj:.4f}",
                f"{model3.fvalue:.4f}",
                f"{model3.f_pvalue:.4f}",
                f"{model3.aic:.4f}",
                f"{model3.bic:.4f}",
                f"{model3.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model3_stats)
        
        # Extract interaction terms
        interaction_terms = [term for term in model3.params.index if 'C(sex)[T.M]:C(rank)' in term]
        
        if interaction_terms:
            # Create a DataFrame for interaction results
            inter_data = []
            for term in interaction_terms:
                coef = model3.params[term]
                p_val = model3.pvalues[term]
                rank = term.split('[T.')[2].split(']')[0]
                inter_data.append({
                    'Interaction': f"Sex × {rank}",
                    'Coefficient': f"${coef:.2f}",
                    'Std Error': f"{model3.bse[term]:.2f}",
                    't-value': f"{model3.tvalues[term]:.4f}",
                    'P-value': f"{p_val:.4f}",
                    'Significant': "Yes" if p_val < 0.05 else "No"
                })
            
            inter_df = pd.DataFrame(inter_data)
            st.write("**Interaction Terms:**")
            st.dataframe(inter_df)
            
            # Interpretation for significant interactions
            sig_inters = [d for d in inter_data if d['Significant'] == "Yes"]
            if sig_inters:
                st.write("**Significant interactions found:**")
                for inter in sig_inters:
                    parts = inter['Interaction'].split('×')
                    rank = parts[1].strip()
                    coef = float(inter['Coefficient'].replace('$', ''))
                    p_val = float(inter['P-value'])
                    
                    if coef > 0:
                        st.write(f"- For {rank} professors, men earn ${coef:.2f} more than would be expected from the main effects alone (p={p_val:.4f}).")
                    else:
                        st.write(f"- For {rank} professors, men earn ${abs(coef):.2f} less than would be expected from the main effects alone (p={p_val:.4f}).")
            else:
                st.write("**No significant interactions found between sex and rank.**")
                
            # Show a DataFrame with key interaction statistics
            interaction_summary = pd.DataFrame({
                'Metric': ['Number of Interactions Tested', 'Number of Significant Interactions', 'Most Significant Interaction', 'Lowest P-value'],
                'Value': [
                    f"{len(inter_data)}",
                    f"{len(sig_inters)}",
                    f"{sig_inters[0]['Interaction'] if sig_inters else 'None'}",
                    f"{min([float(d['P-value']) for d in inter_data]) if inter_data else 'N/A'}"
                ]
            })
            
            st.write("**Interaction Summary:**")
            st.dataframe(interaction_summary)
        else:
            st.write("No interaction terms found between sex and rank in the model.")
        
        # Display full model results in an expander
        with st.expander("View Full Model 3 Results"):
            st.text(model3.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running interaction model: {str(e)}")

def summary(df):
    st.subheader("Question 2 Summary: Sex Bias in Starting Salaries")
    
    # Split data by sex
    df_M = df[df['sex'] == 'M']
    df_F = df[df['sex'] == 'F']
    
    # Calculate basic differences
    raw_diff = df_M['salary'].mean() - df_F['salary'].mean()
    inf_diff = df_M['inf_salary'].mean() - df_F['inf_salary'].mean()
    
    # Run t-test
    t_stat, p_val = stats.ttest_ind(df_M['salary'], df_F['salary'], equal_var=False)
    
    # Run simple regression for summary
    try:
        simple_model = smf.ols(formula="inf_salary ~ C(sex)", data=df).fit()
        sex_coef = simple_model.params.get('C(sex)[T.M]', 0)
        sex_pval = simple_model.pvalues.get('C(sex)[T.M]', 1)
        
        # Run full model
        full_model = smf.ols(formula="inf_salary ~ C(sex) + C(field) + C(rank) + C(deg) + admin + startyr_centered + yrdeg + exp_coded", data=df).fit()
        adj_sex_coef = full_model.params.get('C(sex)[T.M]', 0)
        adj_sex_pval = full_model.pvalues.get('C(sex)[T.M]', 1)
        
        # Check if we have interaction model info
        interaction_model = smf.ols(formula="inf_salary ~ C(sex) + C(field) + C(sex)*C(rank) + C(deg) + admin + startyr_centered", data=df).fit()
        interaction_terms = [term for term in interaction_model.params.index if 'C(sex)[T.M]:C(rank)' in term]
        significant_interactions = [term for term in interaction_terms if interaction_model.pvalues[term] < 0.05]
        
        # Create summary table
        summary_data = [
            {'Analysis': 'Raw Salary Difference', 'Finding': f"${raw_diff:.2f} {'higher for men' if raw_diff > 0 else 'higher for women'}", 'P-value': f"{p_val:.4f}", 'Significant': "Yes" if p_val < 0.05 else "No"},
            {'Analysis': 'Adjusted for Inflation', 'Finding': f"${inf_diff:.2f} {'higher for men' if inf_diff > 0 else 'higher for women'}", 'P-value': f"{p_val:.4f}", 'Significant': "Yes" if p_val < 0.05 else "No"},
            {'Analysis': 'Simple Regression', 'Finding': f"${sex_coef:.2f} {'higher for men' if sex_coef > 0 else 'higher for women'}", 'P-value': f"{sex_pval:.4f}", 'Significant': "Yes" if sex_pval < 0.05 else "No"},
            {'Analysis': 'Full Model with Controls', 'Finding': f"${adj_sex_coef:.2f} {'higher for men' if adj_sex_coef > 0 else 'higher for women'}", 'P-value': f"{adj_sex_pval:.4f}", 'Significant': "Yes" if adj_sex_pval < 0.05 else "No"}
        ]
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # Overall findings
        st.write("### Key Findings")
        
        # Raw difference
        if p_val < 0.05:
            if raw_diff > 0:
                st.write(f"- In raw dollars, men earn ${raw_diff:.2f} more than women on average at the time of hiring (p={p_val:.4f}).")
            else:
                st.write(f"- In raw dollars, women earn ${abs(raw_diff):.2f} more than men on average at the time of hiring (p={p_val:.4f}).")
        else:
            st.write(f"- No statistically significant difference in raw starting salaries between men and women (p={p_val:.4f}).")
        
        # Adjusted difference
        if adj_sex_pval < 0.05:
            if adj_sex_coef > 0:
                st.write(f"- After controlling for field, rank, degree, and other factors, men earn ${adj_sex_coef:.2f} more than women on average (p={adj_sex_pval:.4f}).")
            else:
                st.write(f"- After controlling for field, rank, degree, and other factors, women earn ${abs(adj_sex_coef):.2f} more than men on average (p={adj_sex_pval:.4f}).")
        else:
            st.write(f"- After controlling for field, rank, degree, and other factors, there is no significant difference in starting salaries (p={adj_sex_pval:.4f}).")
        
        # Interaction effects
        if significant_interactions:
            st.write("- Salary differences vary by academic rank:")
            for term in significant_interactions:
                coef = interaction_model.params[term]
                p_val = interaction_model.pvalues[term]
                rank = term.split('[T.')[2].split(']')[0]
                
                if coef > 0:
                    st.write(f"  * For {rank} professors, men earn ${coef:.2f} more than would be expected (p={p_val:.4f})")
                else:
                    st.write(f"  * For {rank} professors, men earn ${abs(coef):.2f} less than would be expected (p={p_val:.4f})")
        else:
            st.write("- No significant interactions were found between sex and academic rank.")
        
        # Overall conclusion
        st.write("### Overall Conclusion")
        if p_val < 0.05 or sex_pval < 0.05 or adj_sex_pval < 0.05 or significant_interactions:
            st.write("Based on the analysis, there is evidence of sex bias in faculty starting salaries. The extent and direction of this bias varies across different academic positions and when controlling for other factors.")
        else:
            st.write("Based on the analysis, there is no strong evidence of sex bias in faculty starting salaries after accounting for other relevant factors.")
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.write("Unable to generate complete summary due to model fitting issues.")

def run_analysis(uploaded_file):
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
        st.error("Unable to process data for Question 2. Please check file format and ensure it contains starting salary data.")
