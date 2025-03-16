import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.cov_struct import Exchangeable
import statsmodels.genmod.families as families
from scipy import stats

# Define palette for consistent colors
custom_palette = {'M': 'skyblue', 'F': 'lightcoral'}

def load_data(uploaded_file):
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\\s+')
        
        # Convert year to 4-digit format if needed
        if df['year'].astype(str).str.len().max() == 2:
            df['year_full'] = df['year'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
        else:
            df['year_full'] = df['year']
        
        return df
    except Exception as e:
        st.error(f"Error processing file for Question 3: {str(e)}")
        return None

def create_wide_format(df):
    try:
        # Filter data for years 1990-1995
        df_filtered = df[(df['year_full'] >= 1990) & (df['year_full'] <= 1995)].copy()
        
        if df_filtered.empty:
            st.error("No data found for years 1990-1995. Please check your dataset.")
            return None, None
        
        # Get unique faculty info from the latest year (e.g., 1995)
        faculty_info = df_filtered[df_filtered['year_full'] == 1995][['id', 'sex', 'deg', 'field', 'rank', 'admin']].drop_duplicates(subset='id')
        
        # Create dictionary for salary per year
        salary_data = {}
        for year in range(1990, 1996):
            year_data = df_filtered[df_filtered['year_full'] == year][['id', 'salary']].copy()
            year_data.columns = ['id', f'salary_{year}']
            salary_data[year] = year_data
        
        # Merge salary data with faculty_info to form wide-format DataFrame
        salary_wide = faculty_info.copy()
        for year, data_year in salary_data.items():
            salary_wide = pd.merge(salary_wide, data_year, on='id', how='left')
        
        # Calculate salary increase, percentage increase, and annual growth rate (slope)
        salary_wide['salary_increase'] = salary_wide['salary_1995'] - salary_wide['salary_1990']
        salary_wide['salary_increase_pct'] = (salary_wide['salary_increase'] / salary_wide['salary_1990']) * 100
        salary_wide['salary_slope'] = salary_wide['salary_increase'] / 5  # simple annual growth rate
        
        # Remove rows missing 1990 or 1995 salary data
        salary_growth = salary_wide.dropna(subset=['salary_1990', 'salary_1995'])
        
        return df_filtered, salary_growth
    except Exception as e:
        st.error(f"Error creating wide format data: {str(e)}")
        return None, None

def exploratory_analysis(df_filtered, salary_growth):
    st.subheader("Exploratory Analysis")
    
    # Summary of salary increases by sex
    summary_stats = salary_growth.groupby('sex')[['salary_increase', 'salary_increase_pct', 'salary_slope']].describe()
    st.write("Summary of Salary Increases by Sex:")
    st.dataframe(summary_stats)
    
    # Create columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot for Salary Increase by Sex
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='sex', y='salary_increase', data=salary_growth, palette=custom_palette, ax=ax1)
        ax1.set_title('Salary Increase (1990-1995) by Sex')
        ax1.set_xlabel('Sex')
        ax1.set_ylabel('Salary Increase ($)')
        st.pyplot(fig1)
    
    with col2:
        # Box plot: Percentage Salary Increase
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='sex', y='salary_increase_pct', data=salary_growth, palette=custom_palette, ax=ax2)
        ax2.set_title('Percentage Salary Increase (1990-1995) by Sex')
        ax2.set_xlabel('Sex')
        ax2.set_ylabel('Percentage Increase (%)')
        st.pyplot(fig2)
    
    # Create another row for charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Line Plot: Average Salary Trends over years
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        years = list(range(1990, 1996))
        avg_male_salary = [df_filtered[(df_filtered['sex'] == 'M') & (df_filtered['year_full'] == year)]['salary'].mean() for year in years]
        avg_female_salary = [df_filtered[(df_filtered['sex'] == 'F') & (df_filtered['year_full'] == year)]['salary'].mean() for year in years]
        
        ax3.plot(years, avg_male_salary, color=custom_palette['M'], linestyle='-', marker='o', label='Male')
        ax3.plot(years, avg_female_salary, color=custom_palette['F'], linestyle='-', marker='o', label='Female')
        ax3.set_title('Average Salary Trends (1990-1995)')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Average Salary ($)')
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)
    
    with col4:
        # Histograms: Salary Slopes by Sex - Fixing the small size issue
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        
        # Use transparent bins with outlines for better visibility
        sns.histplot(salary_growth[salary_growth['sex'] == 'M']['salary_slope'],
                    bins=20, alpha=0.6, color=custom_palette['M'], 
                    edgecolor='darkblue', label='Male', ax=ax4)
        sns.histplot(salary_growth[salary_growth['sex'] == 'F']['salary_slope'],
                    bins=20, alpha=0.6, color=custom_palette['F'], 
                    edgecolor='darkred', label='Female', ax=ax4)
        
        ax4.set_title("Distribution of Salary Growth Rates by Sex")
        ax4.set_xlabel("Annual Salary Growth ($)")
        ax4.set_ylabel("Count")
        ax4.legend()
        
        st.pyplot(fig4)

def statistical_tests(df_filtered, salary_growth):
    st.subheader("Statistical Tests")
    
    # Two-Sample T-Test on Annual Growth Rate
    male_slopes = salary_growth[salary_growth['sex'] == 'M']['salary_slope']
    female_slopes = salary_growth[salary_growth['sex'] == 'F']['salary_slope']
    t_stat, p_value = stats.ttest_ind(male_slopes, female_slopes, equal_var=False)
    
    # Create a DataFrame for the t-test results
    ttest_results = pd.DataFrame({
        'Statistic': ['t-statistic', 'p-value', 'Mean male growth rate', 'Mean female growth rate', 'Difference in means'],
        'Value': [
            f"{t_stat:.4f}",
            f"{p_value:.4f}",
            f"${male_slopes.mean():.2f} per year",
            f"${female_slopes.mean():.2f} per year",
            f"${male_slopes.mean() - female_slopes.mean():.2f} per year"
        ]
    })
    
    st.write("### Two-Sample T-Test for Annual Salary Growth Rate")
    st.dataframe(ttest_results)
    
    # Add interpretation
    if p_value < 0.05:
        if male_slopes.mean() > female_slopes.mean():
            st.write(f"**Interpretation**: There is a statistically significant difference in salary growth rates. Men received ${male_slopes.mean() - female_slopes.mean():.2f} more per year in raises than women (p={p_value:.4f}).")
        else:
            st.write(f"**Interpretation**: There is a statistically significant difference in salary growth rates. Women received ${female_slopes.mean() - male_slopes.mean():.2f} more per year in raises than men (p={p_value:.4f}).")
    else:
        st.write(f"**Interpretation**: There is no statistically significant difference in salary growth rates between men and women (p={p_value:.4f}).")
    
    # Create OLS Models
    # 1. Create dummy variable for sex (M=0, F=1)
    salary_growth['sex_dummy'] = salary_growth['sex'].map({'M': 0, 'F': 1})
    
    # 2. Create additional variables for the models
    salary_growth['has_phd'] = salary_growth['deg'].apply(lambda x: 1 if x == 'PhD' else 0)
    salary_growth['is_arts'] = salary_growth['field'].apply(lambda x: 1 if x == 'Arts' else 0)
    salary_growth['is_prof_field'] = salary_growth['field'].apply(lambda x: 1 if x == 'Prof' else 0)
    salary_growth['is_full_prof'] = salary_growth['rank'].apply(lambda x: 1 if x == 'Full' else 0)
    salary_growth['is_assoc_prof'] = salary_growth['rank'].apply(lambda x: 1 if x == 'Assoc' else 0)
    
    # Structure regression results in a more organized way
    st.write("### Regression Models")
    
    # 1. Unadjusted Model
    st.write("#### Model 1: Sex Only (Unadjusted)")
    try:
        simple_model = sm.OLS.from_formula('salary_slope ~ sex_dummy', data=salary_growth)
        simple_results = simple_model.fit()
        
        # Extract key findings
        sex_coef = simple_results.params['sex_dummy']
        sex_pval = simple_results.pvalues['sex_dummy']
        r_squared = simple_results.rsquared
        
        # Create a DataFrame for clean display
        model1_results = pd.DataFrame({
            'Statistic': ['Effect of being female on salary growth', 'P-value', 'R-squared'],
            'Value': [f"${sex_coef:.2f} per year", f"{sex_pval:.4f}", f"{r_squared:.4f}"]
        })
        
        st.dataframe(model1_results)
        
        # Interpretation
        if sex_pval < 0.05:
            if sex_coef < 0:
                st.write(f"**Interpretation**: Without controlling for other factors, women received ${abs(sex_coef):.2f} less per year in raises than men (p={sex_pval:.4f}).")
            else:
                st.write(f"**Interpretation**: Without controlling for other factors, women received ${sex_coef:.2f} more per year in raises than men (p={sex_pval:.4f}).")
        else:
            st.write(f"**Interpretation**: Without controlling for other factors, there is no statistically significant difference in salary growth between men and women (p={sex_pval:.4f}).")
        
        # Display key model statistics in a structured table instead of raw summary output
        model_stats = pd.DataFrame({
            'Metric': ['Dependent Variable', 'R-squared', 'Adjusted R-squared', 'F-statistic', 'Prob (F-statistic)', 'Observations'],
            'Value': [
                'salary_slope',
                f"{simple_results.rsquared:.4f}",
                f"{simple_results.rsquared_adj:.4f}",
                f"{simple_results.fvalue:.4f}",
                f"{simple_results.f_pvalue:.4f}",
                f"{simple_results.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model_stats)
        
        coef_table = pd.DataFrame({
            'Variable': ['Intercept', 'sex_dummy (Female=1)'],
            'Coefficient': [f"{simple_results.params['Intercept']:.4f}", f"{simple_results.params['sex_dummy']:.4f}"],
            'Std Error': [f"{simple_results.bse['Intercept']:.4f}", f"{simple_results.bse['sex_dummy']:.4f}"],
            'T-value': [f"{simple_results.tvalues['Intercept']:.4f}", f"{simple_results.tvalues['sex_dummy']:.4f}"],
            'P-value': [f"{simple_results.pvalues['Intercept']:.4f}", f"{simple_results.pvalues['sex_dummy']:.4f}"],
            '95% CI Lower': [f"{simple_results.conf_int().iloc[0, 0]:.4f}", f"{simple_results.conf_int().iloc[1, 0]:.4f}"],
            '95% CI Upper': [f"{simple_results.conf_int().iloc[0, 1]:.4f}", f"{simple_results.conf_int().iloc[1, 1]:.4f}"]
        })
        
        st.write("**Coefficients:**")
        st.dataframe(coef_table)
        
        # Still provide the raw output in an expander for those who want it
        with st.expander("View Raw Model Output"):
            st.text(simple_results.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running simple model: {str(e)}")
    
    # 2. Full (Adjusted) Model
    st.write("#### Model 2: Full Model with Controls")
    try:
        full_model = sm.OLS(salary_growth['salary_slope'],
                          sm.add_constant(salary_growth[['sex_dummy', 'has_phd', 'is_arts', 'is_prof_field',
                                                         'is_full_prof', 'is_assoc_prof', 'admin', 'salary_1990']]))
        full_results = full_model.fit()
        
        # Extract key findings
        sex_coef = full_results.params['sex_dummy']
        sex_pval = full_results.pvalues['sex_dummy']
        r_squared = full_results.rsquared
        conf_int = full_results.conf_int()
        sex_ci_lower = conf_int.loc['sex_dummy', 0]
        sex_ci_upper = conf_int.loc['sex_dummy', 1]
        
        # Create a DataFrame for clean display
        model2_results = pd.DataFrame({
            'Statistic': ['Adjusted effect of being female', 'P-value', 'R-squared', 'CI Lower', 'CI Upper'],
            'Value': [f"${sex_coef:.2f} per year", f"{sex_pval:.4f}", f"{r_squared:.4f}", f"${sex_ci_lower:.2f}", f"${sex_ci_upper:.2f}"]
        })
        
        st.dataframe(model2_results)
        
        # Interpretation
        if abs(sex_coef) > 0 and (sex_ci_lower * sex_ci_upper > 0):
            if sex_coef < 0:
                st.write(f"**Interpretation**: After controlling for confounders, women received ${abs(sex_coef):.2f} less per year in raises than men with similar profiles (p={sex_pval:.4f}).")
            else:
                st.write(f"**Interpretation**: After controlling for confounders, women received ${sex_coef:.2f} more per year in raises than men with similar profiles (p={sex_pval:.4f}).")
        else:
            st.write(f"**Interpretation**: After controlling for confounders, there is no statistically significant evidence of sex bias in salary growth (p={sex_pval:.4f}).")
        
        # Display key model statistics in a structured table
        model_stats = pd.DataFrame({
            'Metric': ['Dependent Variable', 'R-squared', 'Adjusted R-squared', 'F-statistic', 'Prob (F-statistic)', 'Observations'],
            'Value': [
                'salary_slope',
                f"{full_results.rsquared:.4f}",
                f"{full_results.rsquared_adj:.4f}",
                f"{full_results.fvalue:.4f}",
                f"{full_results.f_pvalue:.4f}",
                f"{full_results.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model_stats)
        
        # Create a table with coefficients
        coef_data = []
        for var_name in full_results.params.index:
            coef_data.append({
                'Variable': var_name,
                'Coefficient': f"{full_results.params[var_name]:.4f}",
                'Std Error': f"{full_results.bse[var_name]:.4f}",
                'T-value': f"{full_results.tvalues[var_name]:.4f}",
                'P-value': f"{full_results.pvalues[var_name]:.4f}",
                'Significant': "Yes" if full_results.pvalues[var_name] < 0.05 else "No"
            })
        
        coef_table = pd.DataFrame(coef_data)
        
        st.write("**Coefficients:**")
        st.dataframe(coef_table)
        
        # Still provide the raw output in an expander
        with st.expander("View Raw Model Output"):
            st.text(full_results.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running full model: {str(e)}")
    
    # 3. Interaction Model
    st.write("#### Model 3: Interactions between Sex and Rank/Field")
    try:
        # Create interaction terms
        salary_growth['sex_full_prof'] = salary_growth['sex_dummy'] * salary_growth['is_full_prof']
        salary_growth['sex_assoc_prof'] = salary_growth['sex_dummy'] * salary_growth['is_assoc_prof']
        salary_growth['sex_arts'] = salary_growth['sex_dummy'] * salary_growth['is_arts']
        salary_growth['sex_prof_field'] = salary_growth['sex_dummy'] * salary_growth['is_prof_field']
        
        interaction_model = sm.OLS(salary_growth['salary_slope'],
                                 sm.add_constant(salary_growth[['sex_dummy', 'has_phd', 'is_arts', 'is_prof_field',
                                                               'is_full_prof', 'is_assoc_prof', 'admin', 'salary_1990',
                                                               'sex_full_prof', 'sex_assoc_prof', 'sex_arts', 'sex_prof_field']]))
        interaction_results = interaction_model.fit()
        
        # Extract interaction results
        interaction_vars = ['sex_full_prof', 'sex_assoc_prof', 'sex_arts', 'sex_prof_field']
        interaction_data = []
        
        for var in interaction_vars:
            coef = interaction_results.params.get(var, np.nan)
            p_val = interaction_results.pvalues.get(var, np.nan)
            
            if not np.isnan(coef) and not np.isnan(p_val):
                category = var.replace('sex_', '').replace('_', ' ').title()
                interaction_data.append({
                    'Interaction': f"Female × {category}",
                    'Coefficient': f"${coef:.2f}",
                    'P-value': f"{p_val:.4f}",
                    'Significant': "Yes" if p_val < 0.05 else "No"
                })
        
        if interaction_data:
            interaction_df = pd.DataFrame(interaction_data)
            st.dataframe(interaction_df)
            
            # Interpretation for significant interactions
            significant_interactions = [d for d in interaction_data if d['Significant'] == "Yes"]
            if significant_interactions:
                st.write("**Significant interactions found:**")
                for inter in significant_interactions:
                    parts = inter['Interaction'].split('×')
                    category = parts[1].strip()
                    coef = float(inter['Coefficient'].replace('$', ''))
                    p_val = float(inter['P-value'])
                    
                    if coef < 0:
                        st.write(f"- Among {category}, women received ${abs(coef):.2f} less per year than would be expected from the main effects alone (p={p_val:.4f}).")
                    else:
                        st.write(f"- Among {category}, women received ${coef:.2f} more per year than would be expected from the main effects alone (p={p_val:.4f}).")
            else:
                st.write("**No significant interactions were found** indicating variation of the sex effect across rank or field.")
        else:
            st.write("No interaction terms were found in the model.")
        
        # Display key model statistics in a structured table
        model_stats = pd.DataFrame({
            'Metric': ['Dependent Variable', 'R-squared', 'Adjusted R-squared', 'F-statistic', 'Prob (F-statistic)', 'Observations'],
            'Value': [
                'salary_slope',
                f"{interaction_results.rsquared:.4f}",
                f"{interaction_results.rsquared_adj:.4f}",
                f"{interaction_results.fvalue:.4f}",
                f"{interaction_results.f_pvalue:.4f}",
                f"{interaction_results.nobs}"
            ]
        })
        
        st.write("**Model Statistics:**")
        st.dataframe(model_stats)
        
        # Create a table with key coefficients only (not all)
        key_vars = ['const', 'sex_dummy'] + interaction_vars
        coef_data = []
        for var_name in key_vars:
            if var_name in interaction_results.params.index:
                var_label = var_name
                if var_name == 'const':
                    var_label = 'Intercept'
                elif var_name == 'sex_dummy':
                    var_label = 'Female (vs Male)'
                else:
                    var_label = var_name.replace('sex_', 'Female × ').replace('_', ' ').title()
                
                coef_data.append({
                    'Variable': var_label,
                    'Coefficient': f"{interaction_results.params[var_name]:.4f}",
                    'Std Error': f"{interaction_results.bse[var_name]:.4f}",
                    'T-value': f"{interaction_results.tvalues[var_name]:.4f}",
                    'P-value': f"{interaction_results.pvalues[var_name]:.4f}",
                    'Significant': "Yes" if interaction_results.pvalues[var_name] < 0.05 else "No"
                })
        
        if coef_data:
            coef_table = pd.DataFrame(coef_data)
            st.write("**Key Coefficients:**")
            st.dataframe(coef_table)
        
        # Still provide the raw output in an expander
        with st.expander("View Raw Model Output"):
            st.text(interaction_results.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running interaction model: {str(e)}")
    
    # 4. GEE Model (Always run, not optional)
    st.write("#### Model 4: Generalized Estimating Equations (GEE) Model")
    st.write("Analyzing repeated measurements of faculty salaries over years:")
    
    try:
        with st.spinner("Running GEE model... This may take a moment."):
            # Reshape the wide-format salary_growth data to long format for GEE
            salary_long = pd.melt(salary_growth,
                                id_vars=['id', 'sex', 'deg', 'field', 'rank', 'admin', 'sex_dummy',
                                        'has_phd', 'is_arts', 'is_prof_field', 'is_full_prof', 'is_assoc_prof',
                                        'salary_1990', 'sex_full_prof', 'sex_assoc_prof', 'sex_arts', 'sex_prof_field'],
                                value_vars=[f'salary_{year}' for year in range(1990, 1996)],
                                var_name='year', value_name='salary')
            
            salary_long['year'] = salary_long['year'].str.replace('salary_', '').astype(int)
            
            # Run the GEE model
            gee_model = smf.gee("salary ~ sex_dummy + has_phd + is_arts + is_prof_field + is_full_prof + is_assoc_prof + admin + salary_1990",
                                groups='id',
                                data=salary_long,
                                family=families.Gaussian(),
                                cov_struct=Exchangeable())
            gee_results = gee_model.fit()
            
            # Extract key findings
            gee_sex_coef = gee_results.params.get('sex_dummy', 0)
            gee_sex_pval = gee_results.pvalues.get('sex_dummy', 1)
            
            # Create a DataFrame for clean display
            gee_results_df = pd.DataFrame({
                'Statistic': ['GEE model effect of being female', 'P-value'],
                'Value': [f"${gee_sex_coef:.2f}", f"{gee_sex_pval:.4f}"]
            })
            
            st.dataframe(gee_results_df)
            
            # Interpretation
            if gee_sex_pval < 0.05:
                if gee_sex_coef < 0:
                    st.write(f"**Interpretation**: Taking into account repeated measurements, women had ${abs(gee_sex_coef):.2f} lower salaries than men with similar profiles (p={gee_sex_pval:.4f}).")
                else:
                    st.write(f"**Interpretation**: Taking into account repeated measurements, women had ${gee_sex_coef:.2f} higher salaries than men with similar profiles (p={gee_sex_pval:.4f}).")
            else:
                st.write(f"**Interpretation**: Taking into account repeated measurements, there is no statistically significant evidence of sex bias in salaries (p={gee_sex_pval:.4f}).")
            
            # Display full model results in an expander
            with st.expander("View Full GEE Model Results"):
                st.text(gee_results.summary().as_text())
    
    except Exception as e:
        st.error(f"Error running GEE model: {str(e)}")
        st.warning("The GEE model could not be computed, possibly due to data issues or computational limitations.")
    
    # Return models for summary tab
    return t_stat, p_value, full_results, interaction_results, male_slopes, female_slopes

def summary(t_stat, p_value, full_results, interaction_results, male_slopes, female_slopes):
    st.subheader("Question 3 Summary: Sex Bias in Salary Increases")
    
    # Extract sex coefficient and confidence interval from full model
    conf_int = full_results.conf_int()
    sex_coef = full_results.params['sex_dummy']
    sex_ci_lower = conf_int.loc['sex_dummy', 0]
    sex_ci_upper = conf_int.loc['sex_dummy', 1]
    
    # Create a summary table
    summary_data = [
        {'Analysis': 'Unadjusted t-test', 'Finding': f"${male_slopes.mean() - female_slopes.mean():.2f} per year difference", 'P-value': f"{p_value:.4f}", 'Significant': "Yes" if p_value < 0.05 else "No"},
        {'Analysis': 'Adjusted Model', 'Finding': f"${-sex_coef:.2f} per year difference after controls", 'P-value': f"{full_results.pvalues['sex_dummy']:.4f}", 'Significant': "Yes" if full_results.pvalues['sex_dummy'] < 0.05 else "No"}
    ]
    
    # Add interaction findings if significant
    significant_interactions = []
    for var in ['sex_full_prof', 'sex_assoc_prof', 'sex_arts', 'sex_prof_field']:
        p_val = interaction_results.pvalues.get(var, 1)
        if p_val < 0.05:
            coef = interaction_results.params[var]
            category = var.replace('sex_', '').replace('_', ' ').title()
            
            if coef < 0:
                finding = f"Women received ${abs(coef):.2f} less per year in {category}"
            else:
                finding = f"Women received ${coef:.2f} more per year in {category}"
                
            summary_data.append({
                'Analysis': f'Interaction: {category}',
                'Finding': finding,
                'P-value': f"{p_val:.4f}",
                'Significant': "Yes"
            })
            significant_interactions.append((var, coef, p_val))
    
    # Display the summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)
    
    st.write("### Key Findings")
    
    # 1. Unadjusted Analysis
    if p_value < 0.05:
        if male_slopes.mean() > female_slopes.mean():
            diff = male_slopes.mean() - female_slopes.mean()
            st.write(f"- **Unadjusted Analysis**: There is a statistically significant difference in raw salary growth (p={p_value:.4f}).")
            st.write(f"  * Men received an average of ${diff:.2f} more per year in raises than women.")
        else:
            diff = female_slopes.mean() - male_slopes.mean()
            st.write(f"- **Unadjusted Analysis**: There is a statistically significant difference in raw salary growth (p={p_value:.4f}).")
            st.write(f"  * Women received an average of ${diff:.2f} more per year in raises than men.")
    else:
        st.write(f"- **Unadjusted Analysis**: There is no statistically significant difference in raw salary growth (p={p_value:.4f}).")
    
    # 2. Adjusted Analysis
    if abs(sex_coef) > 0 and (sex_ci_lower * sex_ci_upper > 0):
        st.write("- **Adjusted Analysis**: After controlling for confounders, there is statistically significant evidence of sex bias.")
        if sex_coef < 0:
            st.write(f"  * Women received an average of ${abs(sex_coef):.2f} less per year than men with similar profiles.")
        else:
            st.write(f"  * Men received an average of ${abs(sex_coef):.2f} less per year than women with similar profiles.")
    else:
        st.write("- **Adjusted Analysis**: After controlling for confounders, there is no statistically significant evidence of sex bias.")
    
    # 3. Interaction Analysis
    if significant_interactions:
        st.write("- **Interaction Analysis**: Significant interactions found between sex and other factors:")
        for var, coef, p in significant_interactions:
            category = var.replace('sex_', '').replace('_', ' ').title()
            if coef < 0:
                st.write(f"  * Among {category}, women received ${abs(coef):.2f} less per year (p={p:.4f}).")
            else:
                st.write(f"  * Among {category}, women received ${coef:.2f} more per year (p={p:.4f}).")
    else:
        st.write("- **Interaction Analysis**: No significant interactions were found indicating variation of the sex effect across rank or field.")
    
    # Overall conclusion
    st.write("### Overall Conclusion")
    if p_value < 0.05 or (abs(sex_coef) > 0 and (sex_ci_lower * sex_ci_upper > 0)) or significant_interactions:
        st.write("Based on the analysis, there is evidence of sex bias in granting salary increases between 1990-1995. The specific patterns vary when controlling for different factors.")
    else:
        st.write("Based on the analysis, there is no strong evidence of sex bias in granting salary increases between 1990-1995 after controlling for relevant factors.")

def run_analysis(uploaded_file):
    # Create tabs for different analysis components
    tabs = st.tabs(["Exploratory Analysis", "Statistical Tests", "Final Summary"])
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Create wide format data
        df_filtered, salary_growth = create_wide_format(df)
        
        if df_filtered is not None and salary_growth is not None:
            with tabs[0]:
                exploratory_analysis(df_filtered, salary_growth)
                
            with tabs[1]:
                t_stat, p_value, full_results, interaction_results, male_slopes, female_slopes = statistical_tests(df_filtered, salary_growth)
                
            with tabs[2]:
                summary(t_stat, p_value, full_results, interaction_results, male_slopes, female_slopes)
        else:
            st.error("Unable to create wide format data for analysis. Please ensure your data includes faculty records for years 1990-1995.")
    else:
        st.error("Unable to process data for Question 3. Please check file format.")

if __name__ == "__main__":
    st.write("Q3 Salary Increases Analysis Module")
