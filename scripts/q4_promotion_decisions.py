import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from scipy import stats
from patsy import dmatrices

# Define palette for consistent colors
custom_palette = {'M': 'skyblue', 'F': 'lightcoral'}

def load_data(uploaded_file, exclude_after=1990):
    """Load and preprocess data for promotion analysis"""
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\\s+')
        
        # Ensure 'sex' column is treated as a string
        df['sex'] = df['sex'].astype(str)
        
        # Check for key columns
        required_cols = ['id', 'sex', 'rank']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        # Check if we have Associate and Full Professors
        unique_ranks = df['rank'].unique()
        if 'Assoc' not in unique_ranks or 'Full' not in unique_ranks:
            st.error("This analysis requires both Associate and Full Professor data.")
            st.write(f"Available ranks in dataset: {', '.join(unique_ranks)}")
            return None
            
        # Keep only Associate & Full Professors
        df = df[df['rank'].isin(['Assoc', 'Full'])]
        
        # Check if we still have data after filtering
        if df.empty:
            st.error("No Associate or Full Professor data found after filtering.")
            return None
        
        # Convert year columns to 4-digit if needed
        if 'year' in df.columns and df['year'].max() < 100:
            df['year'] = df['year'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
            
        if 'startyr' in df.columns and df['startyr'].max() < 100:
            df['startyr'] = df['startyr'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
            
        if 'yrdeg' in df.columns and df['yrdeg'].max() < 100:
            df['yrdeg'] = df['yrdeg'].apply(lambda x: 1900 + x if x >= 76 else 2000 + x)
        
        # Check/filter for startyr column
        if 'startyr' not in df.columns:
            st.warning("No 'startyr' column found. All faculty will be included in analysis.")
        else:
            # Exclude faculty who became Associate AFTER exclude_after year
            initial_count = len(df)
            df = df[df['startyr'] <= exclude_after]
            filtered_count = len(df)
            # if filtered_count < initial_count:
                # st.info(f"Excluded {initial_count - filtered_count} faculty who became Associate after {exclude_after}.")
        
        # Compute years spent as Associate Professor
        if 'year' in df.columns and 'startyr' in df.columns:
            df['years_as_assoc'] = df['year'] - df['startyr']
        else:
            st.warning("Cannot calculate years as Associate Professor. Missing year or startyr columns.")
            df['years_as_assoc'] = 0
        
        # Identify faculty IDs that appear as Full Professor
        full_professor_ids = df[df['rank'] == 'Full']['id'].unique()
        
        # Assign promotion status: 1 if the ID appears in Full Professors list
        df['promoted'] = df.apply(lambda row: 1 if row['id'] in full_professor_ids else 0, axis=1)
        
        # Compute additional variables if possible
        if 'year' in df.columns and 'yrdeg' in df.columns:
            df['age_at_promotion'] = df['year'] - df['yrdeg']  # Age at promotion
        else:
            df['age_at_promotion'] = 0
            
        if 'startyr' in df.columns and 'yrdeg' in df.columns:
            df['experience'] = df['startyr'] - df['yrdeg']  # Experience at hiring
        else:
            df['experience'] = 0
        
        # Final sanity check
        if df['promoted'].sum() == 0:
            st.warning("No promotions found in this dataset. Some analyses will be limited.")
            
        return df
    except Exception as e:
        st.error(f"Error processing file for Question 4: {str(e)}")
        st.exception(e)
        return None

def exploratory_analysis(df):
    """Perform exploratory analysis with visualizations"""
    # st.subheader("Exploratory Analysis")
    
    try:
        # Basic summary stats
        st.write("### Summary Statistics")
        
        # Calculate promotion counts and rates by gender
        if 'sex' in df.columns and 'promoted' in df.columns:
            # Create contingency table
            try:
                promotion_counts = pd.crosstab(df['sex'], df['promoted'], margins=True, margins_name='Total')
                st.write("Promotion Counts:")
                st.dataframe(promotion_counts)
            except Exception as e:
                st.error(f"Error creating promotion counts table: {str(e)}")
            
            # Calculate promotion rates
            try:
                promotion_rates = df.groupby('sex')['promoted'].mean() * 100
                promotion_rates = pd.DataFrame(promotion_rates).rename(columns={'promoted': 'Promotion Rate (%)'})
                st.write("Promotion Rates (%):")
                st.dataframe(promotion_rates)
            except Exception as e:
                st.error(f"Error calculating promotion rates: {str(e)}")
        else:
            st.error("Missing required columns for summary statistics.")
        
        # Create data visualizations
        st.write("### Visualizations")
        
        # Create two columns for the first row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # 1. Promotion rates by gender
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                promotion_rates.plot(kind='bar', color=[custom_palette.get(i, '#999999') for i in promotion_rates.index], ax=ax1)
                ax1.set_xlabel('Sex')
                ax1.set_ylabel('Promotion Rate (%)')
                ax1.set_title('Promotion Rate by Gender')
                ax1.set_ylim(0, 100)
                ax1.grid(axis='y')
                st.pyplot(fig1)
            except Exception as e:
                st.error(f"Error creating promotion rates chart: {str(e)}")
        
        with col2:
            try:
                # 2. Promotion counts
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.countplot(x='promoted', hue='sex', data=df, palette=custom_palette, ax=ax2)
                ax2.set_xlabel('Promotion Status (0 = Not Promoted, 1 = Promoted)')
                ax2.set_ylabel('Count')
                ax2.set_title('Promotion Counts by Gender')
                ax2.legend(title='Sex')
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error creating promotion counts chart: {str(e)}")
        
        # Only show time-to-promotion charts if we have promoted faculty
        promoted_df = df[df['promoted'] == 1]
        if not promoted_df.empty and 'years_as_assoc' in promoted_df.columns:
            # Create another row for charts
            col3, col4 = st.columns(2)
            
            # Only proceed if we have data for both genders
            male_time = promoted_df[promoted_df['sex'] == 'M']['years_as_assoc']
            female_time = promoted_df[promoted_df['sex'] == 'F']['years_as_assoc']
            
            if not male_time.empty and not female_time.empty:
                with col3:
                    try:
                        # 3. Promotion time boxplot
                        fig3, ax3 = plt.subplots(figsize=(8, 6))
                        sns.boxplot(x='sex', y='years_as_assoc', data=promoted_df, palette=custom_palette, ax=ax3)
                        ax3.set_xlabel('Sex')
                        ax3.set_ylabel('Years as Associate Professor')
                        ax3.set_title('Time to Promotion by Gender')
                        st.pyplot(fig3)
                    except Exception as e:
                        st.error(f"Error creating time to promotion boxplot: {str(e)}")
                
                with col4:
                    try:
                        # 4. Promotion time histogram
                        fig4, ax4 = plt.subplots(figsize=(8, 6))
                        sns.histplot(data=promoted_df, x='years_as_assoc', hue='sex', bins=min(15, int(len(promoted_df)/5)),
                                    kde=True, palette=custom_palette, ax=ax4)
                        ax4.set_xlabel('Years as Associate Professor')
                        ax4.set_ylabel('Count')
                        ax4.set_title('Distribution of Years to Promotion')
                        ax4.legend(title='Sex')
                        st.pyplot(fig4)
                    except Exception as e:
                        st.error(f"Error creating time to promotion histogram: {str(e)}")
            else:
                st.warning("Not enough data to compare time to promotion between genders.")
        else:
            st.warning("No promoted faculty found to analyze time to promotion.")
            
        # 5. Survival Analysis (Kaplan-Meier) - only if we have enough data
        if 'years_as_assoc' in df.columns and df['promoted'].sum() > 0:
            st.write("### Kaplan-Meier Survival Analysis")
            st.write("This shows the probability of remaining an Associate Professor over time:")
            
            try:
                # Prepare data for survival analysis
                df['years_as_assoc'] = pd.to_numeric(df['years_as_assoc'], errors='coerce')
                df['promoted'] = pd.to_numeric(df['promoted'], errors='coerce')
                
                # Drop missing values
                df_survival = df.dropna(subset=['years_as_assoc', 'promoted'])
                
                if df_survival['promoted'].sum() == 0:
                    st.warning("No faculty members were promoted in this dataset. Cannot perform survival analysis.")
                else:
                    # Split dataset by gender
                    male_group = df_survival[df_survival['sex'] == 'M']
                    female_group = df_survival[df_survival['sex'] == 'F']
                    
                    if male_group.empty or female_group.empty:
                        st.warning("Missing data for one or both gender groups. Cannot perform survival analysis.")
                    else:
                        # Check if we have promotions in both groups
                        if male_group['promoted'].sum() == 0 or female_group['promoted'].sum() == 0:
                            st.warning("No promotions for one gender group. Cannot compare survival curves.")
                        else:
                            # Fit Kaplan-Meier models
                            kmf_male = KaplanMeierFitter()
                            kmf_female = KaplanMeierFitter()
                            
                            kmf_male.fit(male_group['years_as_assoc'], event_observed=male_group['promoted'], label="Male")
                            kmf_female.fit(female_group['years_as_assoc'], event_observed=female_group['promoted'], label="Female")
                            
                            # Plot Survival Curve - smaller and centered
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col2:  # Center column
                                fig5, ax5 = plt.subplots(figsize=(7, 5))
                                kmf_male.plot(ax=ax5)
                                kmf_female.plot(ax=ax5)
                                ax5.set_xlabel('Years as Associate Professor')
                                ax5.set_ylabel('Probability of Remaining Associate')
                                ax5.set_title('Kaplan-Meier Survival Curve: Time to Promotion')
                                ax5.legend()
                                ax5.grid(True)
                                st.pyplot(fig5)
                            
                            # Add interpretation
                            st.write("""
                            **How to interpret this plot:**
                            - The y-axis shows the probability of remaining an Associate Professor
                            - The x-axis shows years spent as Associate Professor
                            - A steeper decline indicates faster promotion rates
                            - Separate lines for men and women allow comparison of promotion patterns by gender
                            """)
            except Exception as e:
                st.error(f"Error performing survival analysis: {str(e)}")
        else:
            st.info("Survival analysis requires years as Associate Professor data and at least some promotions.")
    except Exception as e:
        st.error(f"Error in exploratory analysis: {str(e)}")
        st.exception(e)
        
        # 5. Survival Analysis (Kaplan-Meier) - only if we have enough data
        if 'years_as_assoc' in df.columns and df['promoted'].sum() > 0:
            st.write("### Kaplan-Meier Survival Analysis")
            st.write("This shows the probability of remaining an Associate Professor over time:")
            
            try:
                # Prepare data for survival analysis
                df['years_as_assoc'] = pd.to_numeric(df['years_as_assoc'], errors='coerce')
                df['promoted'] = pd.to_numeric(df['promoted'], errors='coerce')
                
                # Drop missing values
                df_survival = df.dropna(subset=['years_as_assoc', 'promoted'])
                
                if df_survival['promoted'].sum() == 0:
                    st.warning("No faculty members were promoted in this dataset. Cannot perform survival analysis.")
                else:
                    # Split dataset by gender
                    male_group = df_survival[df_survival['sex'] == 'M']
                    female_group = df_survival[df_survival['sex'] == 'F']
                    
                    if male_group.empty or female_group.empty:
                        st.warning("Missing data for one or both gender groups. Cannot perform survival analysis.")
                    else:
                        # Check if we have promotions in both groups
                        if male_group['promoted'].sum() == 0 or female_group['promoted'].sum() == 0:
                            st.warning("No promotions for one gender group. Cannot compare survival curves.")
                        else:
                            # Fit Kaplan-Meier models
                            kmf_male = KaplanMeierFitter()
                            kmf_female = KaplanMeierFitter()
                            
                            kmf_male.fit(male_group['years_as_assoc'], event_observed=male_group['promoted'], label="Male")
                            kmf_female.fit(female_group['years_as_assoc'], event_observed=female_group['promoted'], label="Female")
                            
                            # Plot Survival Curve - smaller and centered
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col2:  # Center column
                                fig5, ax5 = plt.subplots(figsize=(7, 5))
                                kmf_male.plot(ax=ax5)
                                kmf_female.plot(ax=ax5)
                                ax5.set_xlabel('Years as Associate Professor')
                                ax5.set_ylabel('Probability of Remaining Associate')
                                ax5.set_title('Kaplan-Meier Survival Curve: Time to Promotion')
                                ax5.legend()
                                ax5.grid(True)
                                st.pyplot(fig5)
                            
                            # Add interpretation
                            st.write("""
                            **How to interpret this plot:**
                            - The y-axis shows the probability of remaining an Associate Professor
                            - The x-axis shows years spent as Associate Professor
                            - A steeper decline indicates faster promotion rates
                            - Separate lines for men and women allow comparison of promotion patterns by gender
                            """)
            except Exception as e:
                st.error(f"Error performing survival analysis: {str(e)}")
        else:
            st.info("Survival analysis requires years as Associate Professor data and at least some promotions.")
    except Exception as e:
        st.error(f"Error in exploratory analysis: {str(e)}")
        st.exception(e)

def statistical_tests(df):
    """Run statistical tests for promotion analysis"""
    st.subheader("Statistical Tests")
    
    # Check for required columns
    required_cols = ['sex', 'promoted']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for statistical tests: {', '.join(missing_cols)}")
        return
    
    # 1. Chi-Square Test for Promotion Rates
    st.write("### Chi-Square Test: Promotion Rates by Gender")
    
    try:
        # Create a contingency table
        promotion_table = pd.crosstab(df['sex'], df['promoted'])
        
        # Check if we have enough data
        if (promotion_table.shape[0] < 2) or (promotion_table.shape[1] < 2):
            st.error("Not enough categories for chi-square test.")
            return
            
        # Check for zero counts
        if (promotion_table.values == 0).any():
            st.warning("Some cells have zero counts. Chi-square results may be unreliable.")
        
        # Perform Chi-Square test
        chi2, p, dof, expected = chi2_contingency(promotion_table)
        
        # Create a DataFrame for clean display
        chi_results = pd.DataFrame({
            'Statistic': ['Chi-Square Value', 'P-value', 'Degrees of Freedom'],
            'Value': [f"{chi2:.4f}", f"{p}", f"{dof}"]
        })
        
        st.dataframe(chi_results)
        
        # Interpretation
        if p < 0.05:
            st.write("**Interpretation**: Promotion rates significantly differ by gender (p < 0.05).")
            
            # Add more details about the direction
            m_rate = promotion_table.loc['M', 1] / promotion_table.loc['M'].sum() * 100
            f_rate = promotion_table.loc['F', 1] / promotion_table.loc['F'].sum() * 100
            
            if m_rate > f_rate:
                st.write(f"Men have a higher promotion rate ({m_rate:.1f}%) than women ({f_rate:.1f}%).")
            else:
                st.write(f"Women have a higher promotion rate ({f_rate:.1f}%) than men ({m_rate:.1f}%).")
        else:
            st.write("**Interpretation**: No significant difference in promotion rates by gender (p > 0.05).")
        
        # Display contingency table
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Contingency Table (Observed Counts):**")
            st.dataframe(promotion_table)
        
        with col2:
            st.write("**Expected Counts (if no association):**")
            expected_df = pd.DataFrame(expected, index=promotion_table.index, columns=promotion_table.columns)
            st.dataframe(expected_df)
    
    except Exception as e:
        st.error(f"Error performing Chi-Square test: {str(e)}")
    
    # 2. Time to Promotion (t-test) - only if we have promoted faculty
    st.write("### T-Test: Time to Promotion by Gender")
    
    try:
        # Get promoted faculty
        promoted_df = df[df['promoted'] == 1]
        
        if promoted_df.empty or 'years_as_assoc' not in promoted_df.columns:
            st.warning("No promoted faculty or years_as_assoc data. Cannot analyze time to promotion.")
        else:
            # Get time to promotion by gender
            male_time = promoted_df[promoted_df['sex'] == 'M']['years_as_assoc']
            female_time = promoted_df[promoted_df['sex'] == 'F']['years_as_assoc']
            
            if male_time.empty or female_time.empty or len(male_time) < 2 or len(female_time) < 2:
                st.warning("Not enough promoted faculty in one or both gender groups for comparison.")
            else:
                # Run t-test
                t_stat, p_val = stats.ttest_ind(male_time, female_time, equal_var=False)
                
                # Get means
                male_mean = male_time.mean()
                female_mean = female_time.mean()
                
                # Create a DataFrame for results
                time_results = pd.DataFrame({
                    'Statistic': ['T-value', 'P-value', 'Mean time for men', 'Mean time for women', 'Difference in means'],
                    'Value': [
                        f"{t_stat:.4f}",
                        f"{p_val}",
                        f"{male_mean:.2f} years",
                        f"{female_mean:.2f} years",
                        f"{male_mean - female_mean:.2f} years"
                    ]
                })
                
                st.dataframe(time_results)
                
                # Interpretation
                if p_val < 0.05:
                    if male_mean < female_mean:
                        st.write(f"**Interpretation**: There is a statistically significant difference in time to promotion. Women take {female_mean - male_mean:.2f} years longer than men on average to be promoted from Associate to Full Professor (p={p_val}).")
                    else:
                        st.write(f"**Interpretation**: There is a statistically significant difference in time to promotion. Men take {male_mean - female_mean:.2f} years longer than women on average to be promoted from Associate to Full Professor (p={p_val}).")
                else:
                    st.write(f"**Interpretation**: There is no statistically significant difference in time to promotion between men ({male_mean:.2f} years) and women ({female_mean:.2f} years), p={p_val}.")
    
    except Exception as e:
        st.error(f"Error performing t-test on promotion time: {str(e)}")
    
    # 3. Logistic Regression Models
    st.write("### Logistic Regression Models")
    
    try:
        # Prepare data for regression
        df_model = df.copy()
        
        # Convert categorical variables to numeric
        df_model['sex_numeric'] = df_model['sex'].map({'M': 1, 'F': 0})
        
        # Convert to numeric & drop NaNs
        numeric_cols = ['years_as_assoc', 'promoted', 'age_at_promotion', 'experience']
        for col in numeric_cols:
            if col in df_model.columns:
                df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
        
        # Drop rows with missing key variables
        df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna(subset=['sex_numeric', 'promoted'])
        
        # Create categorical variables if needed
        if 'field' in df_model.columns:
            df_model['field_numeric'] = df_model['field'].astype('category').cat.codes
        if 'deg' in df_model.columns:
            df_model['deg_numeric'] = df_model['deg'].astype('category').cat.codes
            
        # Check if we have enough data
        if len(df_model) < 10:
            st.error("Not enough data for regression models after cleaning.")
            return
            
        # Simple model (Model 1: Gender Only)
        st.write("#### Model 1: Gender Only")
        try:
            X1 = df_model[['sex_numeric']]
            X1 = sm.add_constant(X1)
            y = df_model['promoted']
            
            model1 = sm.Logit(y, X1).fit(disp=0)
            
            # Extract key results
            coef = model1.params['sex_numeric']
            p_val = model1.pvalues['sex_numeric']
            odds_ratio = np.exp(coef)
            
            # Create a DataFrame for clean display
            model1_results = pd.DataFrame({
                'Statistic': ['Gender Coefficient', 'P-value', 'Odds Ratio'],
                'Value': [f"{coef:.4f}", f"{p_val}", f"{odds_ratio:.4f}"]
            })
            
            st.dataframe(model1_results)
            
            # Interpretation
            if p_val < 0.05:
                if coef > 0:
                    st.write(f"**Interpretation**: Men are {odds_ratio:.2f} times more likely to be promoted than women (p={p_val}).")
                else:
                    st.write(f"**Interpretation**: Women are {1/odds_ratio:.2f} times more likely to be promoted than men (p={p_val}).")
            else:
                st.write(f"**Interpretation**: No significant difference in promotion likelihood between men and women (p={p_val}).")
            
            # Model information in structured format
            model_info = pd.DataFrame({
                'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                'Value': [
                    'promoted (0/1)',
                    f"{model1.prsquared:.4f}",
                    f"{model1.llf:.4f}",
                    f"{model1.llr_pvalue}",
                    f"{model1.aic:.4f}",
                    f"{model1.bic:.4f}",
                    f"{model1.nobs}"
                ]
            })
            
            st.write("**Model Statistics:**")
            st.dataframe(model_info)
            
            # Coefficient table
            coef_table = pd.DataFrame({
                'Variable': ['Intercept', 'sex_numeric (Male=1)'],
                'Coefficient': [f"{model1.params['const']:.4f}", f"{model1.params['sex_numeric']:.4f}"],
                'Std Error': [f"{model1.bse['const']:.4f}", f"{model1.bse['sex_numeric']:.4f}"],
                'Z-value': [f"{model1.tvalues['const']:.4f}", f"{model1.tvalues['sex_numeric']:.4f}"],
                'P-value': [f"{model1.pvalues['const']}", f"{model1.pvalues['sex_numeric']}"],
                'Odds Ratio': [f"{np.exp(model1.params['const']):.4f}", f"{np.exp(model1.params['sex_numeric']):.4f}"]
            })
            
            st.write("**Coefficients:**")
            st.dataframe(coef_table)
            
            # Raw output in expander
            with st.expander("View Raw Model 1 Output"):
                st.text(model1.summary().as_text())
        
        except Exception as e:
            st.error(f"Error running simple logistic model: {str(e)}")
            model1 = None
            
        # Model 2: Add years as associate if available
        if 'years_as_assoc' in df_model.columns:
            st.write("#### Model 2: Gender + Years as Associate")
            try:
                X2 = df_model[['sex_numeric', 'years_as_assoc']].dropna()
                if len(X2) < 10:
                    st.warning("Not enough data for Model 2 after removing missing values.")
                else:
                    X2 = sm.add_constant(X2)
                    y2 = df_model.loc[X2.index, 'promoted']
                    
                    model2 = sm.Logit(y2, X2).fit(disp=0)
                    
                    # Extract key results in a table format
                    model2_results = pd.DataFrame({
                        'Variable': ['Gender (Male=1)', 'Years as Associate'],
                        'Coefficient': [f"{model2.params['sex_numeric']:.4f}", f"{model2.params['years_as_assoc']:.4f}"],
                        'P-value': [f"{model2.pvalues['sex_numeric']}", f"{model2.pvalues['years_as_assoc']}"],
                        'Odds Ratio': [f"{np.exp(model2.params['sex_numeric']):.4f}", f"{np.exp(model2.params['years_as_assoc']):.4f}"]
                    })
                    
                    st.dataframe(model2_results)
                    
                    # Interpretation
                    interpretations = []
                    
                    if model2.pvalues['sex_numeric'] < 0.05:
                        if model2.params['sex_numeric'] > 0:
                            interpretations.append(f"After controlling for time spent as Associate, men are {np.exp(model2.params['sex_numeric']):.2f} times more likely to be promoted than women (p={model2.pvalues['sex_numeric']}).")
                        else:
                            interpretations.append(f"After controlling for time spent as Associate, women are {1/np.exp(model2.params['sex_numeric']):.2f} times more likely to be promoted than men (p={model2.pvalues['sex_numeric']}).")
                    else:
                        interpretations.append(f"After controlling for time spent as Associate, there is no significant difference in promotion likelihood between men and women (p={model2.pvalues['sex_numeric']}).")
                    
                    if model2.pvalues['years_as_assoc'] < 0.05:
                        years_odds = np.exp(model2.params['years_as_assoc'])
                        if model2.params['years_as_assoc'] > 0:
                            interpretations.append(f"Each additional year spent as Associate Professor increases promotion likelihood by {(years_odds-1)*100:.1f}% (p={model2.pvalues['years_as_assoc']}).")
                        else:
                            interpretations.append(f"Each additional year spent as Associate Professor decreases promotion likelihood by {(1-years_odds)*100:.1f}% (p={model2.pvalues['years_as_assoc']}).")
                    
                    st.write("**Interpretations:**")
                    for interp in interpretations:
                        st.write(f"- {interp}")
                    
                    # Model information
                    model_info = pd.DataFrame({
                        'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                        'Value': [
                            'promoted (0/1)',
                            f"{model2.prsquared:.4f}",
                            f"{model2.llf:.4f}",
                            f"{model2.llr_pvalue}",
                            f"{model2.aic:.4f}",
                            f"{model2.bic:.4f}",
                            f"{model2.nobs}"
                        ]
                    })
                    
                    st.write("**Model Statistics:**")
                    st.dataframe(model_info)
                    
                    # Coefficient table with more details
                    coef_table = pd.DataFrame({
                        'Variable': ['Intercept', 'Sex (Male=1)', 'Years as Associate'],
                        'Coefficient': [
                            f"{model2.params['const']:.4f}", 
                            f"{model2.params['sex_numeric']:.4f}", 
                            f"{model2.params['years_as_assoc']:.4f}"
                        ],
                        'Std Error': [
                            f"{model2.bse['const']:.4f}", 
                            f"{model2.bse['sex_numeric']:.4f}", 
                            f"{model2.bse['years_as_assoc']:.4f}"
                        ],
                        'Z-value': [
                            f"{model2.tvalues['const']:.4f}", 
                            f"{model2.tvalues['sex_numeric']:.4f}", 
                            f"{model2.tvalues['years_as_assoc']:.4f}"
                        ],
                        'P-value': [
                            f"{model2.pvalues['const']}", 
                            f"{model2.pvalues['sex_numeric']}", 
                            f"{model2.pvalues['years_as_assoc']}"
                        ],
                        'Odds Ratio': [
                            f"{np.exp(model2.params['const']):.4f}", 
                            f"{np.exp(model2.params['sex_numeric']):.4f}", 
                            f"{np.exp(model2.params['years_as_assoc']):.4f}"
                        ]
                    })
                    
                    st.write("**Coefficients:**")
                    st.dataframe(coef_table)
                    
                    # Raw output in expander
                    with st.expander("View Raw Model 2 Output"):
                        st.text(model2.summary().as_text())
            except Exception as e:
                st.error(f"Error running Model 2: {str(e)}")
                model2 = None
        
        # Interaction Model: Sex × Administrative Role (from q4_stats(1).py)
        st.write("#### Model 3: Sex × Admin Interaction")
        try:
            # Ensure we have admin column
            if 'admin' not in df_model.columns:
                st.warning("Administrative duties column ('admin') not found. Cannot run interaction model.")
            else:
                # Convert admin to numeric if needed
                df_model['admin'] = pd.to_numeric(df_model['admin'], errors='coerce')
                
                # Create interaction term
                df_model['sex_admin'] = df_model['sex_numeric'] * df_model['admin']
                
                # Define features for interaction model
                model3_features = ['sex_numeric', 'admin', 'sex_admin']
                X3 = df_model[model3_features].dropna()
                
                if len(X3) < 10:
                    st.warning("Not enough data for Model 3 after removing missing values.")
                else:
                    X3 = sm.add_constant(X3)
                    y3 = df_model.loc[X3.index, 'promoted']
                    
                    model3 = sm.Logit(y3, X3).fit(disp=0)
                    
                    # Extract key results
                    # Create a DataFrame for coefficients
                    model3_results = pd.DataFrame({
                        'Variable': ['Gender (Male=1)', 'Admin Duties', 'Gender × Admin Interaction'],
                        'Coefficient': [
                            f"{model3.params['sex_numeric']:.4f}", 
                            f"{model3.params['admin']:.4f}",
                            f"{model3.params['sex_admin']:.4f}"
                        ],
                        'P-value': [
                            f"{model3.pvalues['sex_numeric']}", 
                            f"{model3.pvalues['admin']}",
                            f"{model3.pvalues['sex_admin']}"
                        ],
                        'Odds Ratio': [
                            f"{np.exp(model3.params['sex_numeric']):.4f}", 
                            f"{np.exp(model3.params['admin']):.4f}",
                            f"{np.exp(model3.params['sex_admin']):.4f}"
                        ]
                    })
                    
                    st.dataframe(model3_results)
                    
                    # Interpretation
                    st.write("**Interpretations:**")
                    
                    # Gender effect
                    if model3.pvalues['sex_numeric'] < 0.05:
                        if model3.params['sex_numeric'] > 0:
                            st.write(f"- For faculty without administrative duties, men are {np.exp(model3.params['sex_numeric']):.2f} times more likely to be promoted than women (p={model3.pvalues['sex_numeric']}).")
                        else:
                            st.write(f"- For faculty without administrative duties, women are {1/np.exp(model3.params['sex_numeric']):.2f} times more likely to be promoted than men (p={model3.pvalues['sex_numeric']}).")
                    else:
                        st.write(f"- For faculty without administrative duties, there is no significant difference in promotion likelihood between men and women (p={model3.pvalues['sex_numeric']}).")
                    
                    # Admin effect
                    if model3.pvalues['admin'] < 0.05:
                        if model3.params['admin'] > 0:
                            st.write(f"- For women, having administrative duties increases promotion likelihood by {(np.exp(model3.params['admin'])-1)*100:.1f}% (p={model3.pvalues['admin']}).")
                        else:
                            st.write(f"- For women, having administrative duties decreases promotion likelihood by {(1-np.exp(model3.params['admin']))*100:.1f}% (p={model3.pvalues['admin']}).")
                    else:
                        st.write(f"- For women, administrative duties do not significantly affect promotion likelihood (p={model3.pvalues['admin']}).")
                    
                    # Interaction effect
                    if model3.pvalues['sex_admin'] < 0.05:
                        if model3.params['sex_admin'] > 0:
                            st.write(f"- The effect of administrative duties is significantly stronger for men than for women (p={model3.pvalues['sex_admin']}).")
                        else:
                            st.write(f"- The effect of administrative duties is significantly stronger for women than for men (p={model3.pvalues['sex_admin']}).")
                        
                        # Calculate combined effect for men
                        combined_effect = model3.params['admin'] + model3.params['sex_admin']
                        if combined_effect > 0:
                            st.write(f"- For men, having administrative duties increases promotion likelihood by {(np.exp(combined_effect)-1)*100:.1f}%.")
                        else:
                            st.write(f"- For men, having administrative duties decreases promotion likelihood by {(1-np.exp(combined_effect))*100:.1f}%.")
                    else:
                        st.write(f"- The effect of administrative duties does not significantly differ between men and women (p={model3.pvalues['sex_admin']}).")
                    
                    # Model information
                    model_info = pd.DataFrame({
                        'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                        'Value': [
                            'promoted (0/1)',
                            f"{model3.prsquared:.4f}",
                            f"{model3.llf:.4f}",
                            f"{model3.llr_pvalue}",
                            f"{model3.aic:.4f}",
                            f"{model3.bic:.4f}",
                            f"{model3.nobs}"
                        ]
                    })
                    
                    st.write("**Model Statistics:**")
                    st.dataframe(model_info)
                    
                    # Coefficients table
                    coef_table = pd.DataFrame({
                        'Variable': ['Intercept', 'Gender (Male=1)', 'Admin Duties', 'Gender × Admin'],
                        'Coefficient': [
                            f"{model3.params['const']:.4f}",
                            f"{model3.params['sex_numeric']:.4f}", 
                            f"{model3.params['admin']:.4f}",
                            f"{model3.params['sex_admin']:.4f}"
                        ],
                        'Std Error': [
                            f"{model3.bse['const']:.4f}",
                            f"{model3.bse['sex_numeric']:.4f}", 
                            f"{model3.bse['admin']:.4f}",
                            f"{model3.bse['sex_admin']:.4f}"
                        ],
                        'Z-value': [
                            f"{model3.tvalues['const']:.4f}",
                            f"{model3.tvalues['sex_numeric']:.4f}", 
                            f"{model3.tvalues['admin']:.4f}",
                            f"{model3.tvalues['sex_admin']:.4f}"
                        ],
                        'P-value': [
                            f"{model3.pvalues['const']}",
                            f"{model3.pvalues['sex_numeric']}", 
                            f"{model3.pvalues['admin']}",
                            f"{model3.pvalues['sex_admin']}"
                        ],
                        'Odds Ratio': [
                            f"{np.exp(model3.params['const']):.4f}",
                            f"{np.exp(model3.params['sex_numeric']):.4f}", 
                            f"{np.exp(model3.params['admin']):.4f}",
                            f"{np.exp(model3.params['sex_admin']):.4f}"
                        ]
                    })
                    
                    st.write("**Coefficients:**")
                    st.dataframe(coef_table)
                    
                    # Raw output in expander
                    with st.expander("View Raw Model 3 Output"):
                        st.text(model3.summary().as_text())
                        # Extract and display key tables from the raw output
                        st.write("**Extracted Tables from Raw Output:**")
                        
                        # Table of coefficients
                        st.write("*Coefficients Table:*")
                        coef_df = pd.DataFrame({
                            'coef': model3.params,
                            'std err': model3.bse,
                            'z': model3.tvalues,
                            'P>|z|': model3.pvalues,
                            'Odds Ratio': np.exp(model3.params),
                            '[0.025': model3.conf_int()[0],
                            '0.975]': model3.conf_int()[1]
                        })
                        st.dataframe(coef_df)
                        
                        # Marginal effects
                        st.write("*Marginal Effects:*")
                        marginal_effects = pd.DataFrame({
                            'Variable': ['Gender (Male=1)', 'Admin Duties', 'Gender × Admin'],
                            'dy/dx': [
                                f"{model3.params['sex_numeric']:.4f}", 
                                f"{model3.params['admin']:.4f}",
                                f"{model3.params['sex_admin']:.4f}"
                            ],
                            'Effect': [
                                f"{(np.exp(model3.params['sex_numeric'])-1)*100:.1f}% change in odds",
                                f"{(np.exp(model3.params['admin'])-1)*100:.1f}% change in odds",
                                f"{(np.exp(model3.params['sex_admin'])-1)*100:.1f}% change in odds"
                            ]
                        })
                        st.dataframe(marginal_effects)
                        
                        # Model fit statistics
                        st.write("*Model Fit Statistics:*")
                        fit_stats = pd.DataFrame({
                            'Metric': ['Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC'],
                            'Value': [
                                f"{model3.prsquared:.4f}",
                                f"{model3.llf:.4f}",
                                f"{model3.llr_pvalue}",
                                f"{model3.aic:.4f}",
                                f"{model3.bic:.4f}"
                            ]
                        })
                        st.dataframe(fit_stats)
        except Exception as e:
            st.error(f"Error running Model 3: {str(e)}")
            
        # Field Interaction Model
        st.write("#### Model 4: Sex × Field Interaction")
        try:
            # Ensure we have field column
            if 'field_numeric' not in df_model.columns:
                st.warning("Academic field column ('field') not found. Cannot run field interaction model.")
            else:
                # Create interaction term
                df_model['sex_field'] = df_model['sex_numeric'] * df_model['field_numeric']
                
                # Define features for interaction model
                model4_features = ['sex_numeric', 'field_numeric', 'sex_field']
                X4 = df_model[model4_features].dropna()
                
                if len(X4) < 10:
                    st.warning("Not enough data for Model 4 after removing missing values.")
                else:
                    X4 = sm.add_constant(X4)
                    y4 = df_model.loc[X4.index, 'promoted']
                    
                    model4 = sm.Logit(y4, X4).fit(disp=0)
                    
                    # Create a DataFrame for coefficients
                    model4_results = pd.DataFrame({
                        'Variable': ['Gender (Male=1)', 'Academic Field', 'Gender × Field Interaction'],
                        'Coefficient': [
                            f"{model4.params['sex_numeric']:.4f}", 
                            f"{model4.params['field_numeric']:.4f}",
                            f"{model4.params['sex_field']:.4f}"
                        ],
                        'P-value': [
                            f"{model4.pvalues['sex_numeric']}", 
                            f"{model4.pvalues['field_numeric']}",
                            f"{model4.pvalues['sex_field']}"
                        ],
                        'Odds Ratio': [
                            f"{np.exp(model4.params['sex_numeric']):.4f}", 
                            f"{np.exp(model4.params['field_numeric']):.4f}",
                            f"{np.exp(model4.params['sex_field']):.4f}"
                        ]
                    })
                    
                    st.dataframe(model4_results)
                    
                    # Interpretation
                    st.write("**Interpretations:**")
                    
                    # Gender effect
                    if model4.pvalues['sex_numeric'] < 0.05:
                        if model4.params['sex_numeric'] > 0:
                            st.write(f"- For the reference field (field=0), men are {np.exp(model4.params['sex_numeric']):.2f} times more likely to be promoted than women (p={model4.pvalues['sex_numeric']}).")
                        else:
                            st.write(f"- For the reference field (field=0), women are {1/np.exp(model4.params['sex_numeric']):.2f} times more likely to be promoted than men (p={model4.pvalues['sex_numeric']}).")
                    else:
                        st.write(f"- For the reference field (field=0), there is no significant difference in promotion likelihood between men and women (p={model4.pvalues['sex_numeric']}).")
                    
                    # Field effect
                    if model4.pvalues['field_numeric'] < 0.05:
                        if model4.params['field_numeric'] > 0:
                            st.write(f"- For women, moving up one field category increases promotion likelihood by {(np.exp(model4.params['field_numeric'])-1)*100:.1f}% (p={model4.pvalues['field_numeric']}).")
                        else:
                            st.write(f"- For women, moving up one field category decreases promotion likelihood by {(1-np.exp(model4.params['field_numeric']))*100:.1f}% (p={model4.pvalues['field_numeric']}).")
                    else:
                        st.write(f"- For women, academic field does not significantly affect promotion likelihood (p={model4.pvalues['field_numeric']}).")
                    
                    # Interaction effect
                    if model4.pvalues['sex_field'] < 0.05:
                        if model4.params['sex_field'] > 0:
                            st.write(f"- The effect of academic field is significantly stronger for men than for women (p={model4.pvalues['sex_field']}).")
                        else:
                            st.write(f"- The effect of academic field is significantly stronger for women than for men (p={model4.pvalues['sex_field']}).")
                        
                        # Calculate combined effect for men
                        combined_effect = model4.params['field_numeric'] + model4.params['sex_field']
                        if combined_effect > 0:
                            st.write(f"- For men, moving up one field category increases promotion likelihood by {(np.exp(combined_effect)-1)*100:.1f}%.")
                        else:
                            st.write(f"- For men, moving up one field category decreases promotion likelihood by {(1-np.exp(combined_effect))*100:.1f}%.")
                    else:
                        st.write(f"- The effect of academic field does not significantly differ between men and women (p={model4.pvalues['sex_field']}).")
                    
                    # Model information
                    model_info = pd.DataFrame({
                        'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                        'Value': [
                            'promoted (0/1)',
                            f"{model4.prsquared:.4f}",
                            f"{model4.llf:.4f}",
                            f"{model4.llr_pvalue}",
                            f"{model4.aic:.4f}",
                            f"{model4.bic:.4f}",
                            f"{model4.nobs}"
                        ]
                    })
                    
                    st.write("**Model Statistics:**")
                    st.dataframe(model_info)
                    
                    # Coefficients table
                    coef_table = pd.DataFrame({
                        'Variable': ['Intercept', 'Gender (Male=1)', 'Academic Field', 'Gender × Field'],
                        'Coefficient': [
                            f"{model4.params['const']:.4f}",
                            f"{model4.params['sex_numeric']:.4f}", 
                            f"{model4.params['field_numeric']:.4f}",
                            f"{model4.params['sex_field']:.4f}"
                        ],
                        'Std Error': [
                            f"{model4.bse['const']:.4f}",
                            f"{model4.bse['sex_numeric']:.4f}", 
                            f"{model4.bse['field_numeric']:.4f}",
                            f"{model4.bse['sex_field']:.4f}"
                        ],
                        'Z-value': [
                            f"{model4.tvalues['const']:.4f}",
                            f"{model4.tvalues['sex_numeric']:.4f}", 
                            f"{model4.tvalues['field_numeric']:.4f}",
                            f"{model4.tvalues['sex_field']:.4f}"
                        ],
                        'P-value': [
                            f"{model4.pvalues['const']}",
                            f"{model4.pvalues['sex_numeric']}", 
                            f"{model4.pvalues['field_numeric']}",
                            f"{model4.pvalues['sex_field']}"
                        ],
                        'Odds Ratio': [
                            f"{np.exp(model4.params['const']):.4f}",
                            f"{np.exp(model4.params['sex_numeric']):.4f}", 
                            f"{np.exp(model4.params['field_numeric']):.4f}",
                            f"{np.exp(model4.params['sex_field']):.4f}"
                        ]
                    })
                    
                    st.write("**Coefficients:**")
                    st.dataframe(coef_table)
                    
                    # Raw output in expander
                    with st.expander("View Raw Model 4 Output"):
                        st.text(model4.summary().as_text())
                        # Extract and display key tables from the raw output
                        st.write("**Extracted Tables from Raw Output:**")
                        
                        # Table of coefficients
                        st.write("*Coefficients Table:*")
                        coef_df = pd.DataFrame({
                            'coef': model4.params,
                            'std err': model4.bse,
                            'z': model4.tvalues,
                            'P>|z|': model4.pvalues,
                            'Odds Ratio': np.exp(model4.params),
                            '[0.025': model4.conf_int()[0],
                            '0.975]': model4.conf_int()[1]
                        })
                        st.dataframe(coef_df)
                        
                        # Marginal effects
                        st.write("*Marginal Effects:*")
                        marginal_effects = pd.DataFrame({
                            'Variable': ['Gender (Male=1)', 'Academic Field', 'Gender × Field'],
                            'dy/dx': [
                                f"{model4.params['sex_numeric']:.4f}", 
                                f"{model4.params['field_numeric']:.4f}",
                                f"{model4.params['sex_field']:.4f}"
                            ],
                            'Effect': [
                                f"{(np.exp(model4.params['sex_numeric'])-1)*100:.1f}% change in odds",
                                f"{(np.exp(model4.params['field_numeric'])-1)*100:.1f}% change in odds",
                                f"{(np.exp(model4.params['sex_field'])-1)*100:.1f}% change in odds"
                            ]
                        })
                        st.dataframe(marginal_effects)
                        
                        # Model fit statistics
                        st.write("*Model Fit Statistics:*")
                        fit_stats = pd.DataFrame({
                            'Metric': ['Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC'],
                            'Value': [
                                f"{model4.prsquared:.4f}",
                                f"{model4.llf:.4f}",
                                f"{model4.llr_pvalue}",
                                f"{model4.aic:.4f}",
                                f"{model4.bic:.4f}"
                            ]
                        })
                        st.dataframe(fit_stats)
        except Exception as e:
            st.error(f"Error running Model 4: {str(e)}")
    except Exception as e:
        st.error(f"Error in logistic regression analysis: {str(e)}")
        st.exception(e)

def summary(df):
    """Generate a summary of findings for Question 4"""
    st.subheader("Question 4 Summary: Sex Bias in Promotion Decisions")
    
    if df is None or df.empty:
        st.error("No data available for summary.")
        return
        
    try:
        # Basic promotion statistics
        try:
            promotion_table = pd.crosstab(df['sex'], df['promoted'])
            promotion_rates = df.groupby('sex')['promoted'].mean() * 100
            
            # Chi-square test
            chi2, p, dof, expected = chi2_contingency(promotion_table)
            has_chi2 = True
        except Exception as e:
            st.warning(f"Could not calculate promotion statistics: {str(e)}")
            has_chi2 = False
        
        # Try to get time-to-promotion stats
        try:
            promoted_df = df[df['promoted'] == 1]
            if not promoted_df.empty and 'years_as_assoc' in promoted_df.columns:
                male_time = promoted_df[promoted_df['sex'] == 'M']['years_as_assoc']
                female_time = promoted_df[promoted_df['sex'] == 'F']['years_as_assoc']
                
                if len(male_time) >= 2 and len(female_time) >= 2:
                    t_stat, p_val_time = stats.ttest_ind(male_time, female_time, equal_var=False)
                    avg_male_time = male_time.mean()
                    avg_female_time = female_time.mean()
                    has_time_data = True
                else:
                    has_time_data = False
            else:
                has_time_data = False
        except Exception as e:
            has_time_data = False
        
        # Try to run simplified models
        try:
            df_model = df.copy()
            df_model['sex_numeric'] = df_model['sex'].map({'M': 1, 'F': 0})
            df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna(subset=['sex_numeric', 'promoted'])
            
            # Basic gender model
            if len(df_model) >= 10:
                X = sm.add_constant(df_model[['sex_numeric']])
                y = df_model['promoted']
                
                model = sm.Logit(y, X).fit(disp=0)
                gender_coef = model.params['sex_numeric']
                gender_pval = model.pvalues['sex_numeric']
                odds_ratio = np.exp(gender_coef)
                has_model = True
            else:
                has_model = False
                
            # Try interaction models if data available
            if 'admin' in df_model.columns:
                df_model['admin'] = pd.to_numeric(df_model['admin'], errors='coerce')
                df_model['sex_admin'] = df_model['sex_numeric'] * df_model['admin']
                has_admin_interaction = True
            else:
                has_admin_interaction = False
                
            if 'field_numeric' in df_model.columns:
                df_model['sex_field'] = df_model['sex_numeric'] * df_model['field_numeric']
                has_field_interaction = True
            else:
                has_field_interaction = False
                
        except Exception as e:
            has_model = False
            has_admin_interaction = False
            has_field_interaction = False
            
        # Create a comprehensive summary table
        summary_data = []
        
        if has_chi2:
            m_rate = promotion_rates['M'] if 'M' in promotion_rates.index else 0
            f_rate = promotion_rates['F'] if 'F' in promotion_rates.index else 0
            
            summary_data.append({
                'Analysis': 'Promotion Rate (Chi-Square)', 
                'Finding': f"Male: {m_rate:.1f}%, Female: {f_rate:.1f}%", 
                'P-value': f"{p}",
                'Significant': "Yes" if p < 0.05 else "No"
            })
        
        if has_model:
            effect = "more likely" if gender_coef > 0 else "less likely"
            factor = odds_ratio if gender_coef > 0 else 1/odds_ratio
            finding = f"Men are {factor:.2f}× {effect} to be promoted than women"
            
            summary_data.append({
                'Analysis': 'Promotion Likelihood (Logistic Regression)', 
                'Finding': finding, 
                'P-value': f"{gender_pval}",
                'Significant': "Yes" if gender_pval < 0.05 else "No"
            })
            
            # Add interaction model findings if available
            if has_admin_interaction:
                try:
                    # Create admin interaction model for summary
                    X_admin = sm.add_constant(df_model[['sex_numeric', 'admin', 'sex_admin']].dropna())
                    y_admin = df_model.loc[X_admin.index, 'promoted']
                    admin_model = sm.Logit(y_admin, X_admin).fit(disp=0)
                    
                    # Check if interaction is significant
                    if admin_model.pvalues['sex_admin'] < 0.05:
                        if admin_model.params['sex_admin'] > 0:
                            admin_finding = f"Admin duties have stronger positive effect for men (p={admin_model.pvalues['sex_admin']})"
                        else:
                            admin_finding = f"Admin duties have stronger positive effect for women (p={admin_model.pvalues['sex_admin']})"
                        
                        summary_data.append({
                            'Analysis': 'Sex × Admin Interaction', 
                            'Finding': admin_finding, 
                            'P-value': f"{admin_model.pvalues['sex_admin']}",
                            'Significant': "Yes"
                        })
                except Exception as e:
                    pass
                
            if has_field_interaction:
                try:
                    # Create field interaction model for summary
                    X_field = sm.add_constant(df_model[['sex_numeric', 'field_numeric', 'sex_field']].dropna())
                    y_field = df_model.loc[X_field.index, 'promoted']
                    field_model = sm.Logit(y_field, X_field).fit(disp=0)
                    
                    # Check if interaction is significant
                    if field_model.pvalues['sex_field'] < 0.05:
                        if field_model.params['sex_field'] > 0:
                            field_finding = f"Field effect is stronger for men (p={field_model.pvalues['sex_field']})"
                        else:
                            field_finding = f"Field effect is stronger for women (p={field_model.pvalues['sex_field']})"
                        
                        summary_data.append({
                            'Analysis': 'Sex × Field Interaction', 
                            'Finding': field_finding, 
                            'P-value': f"{field_model.pvalues['sex_field']}",
                            'Significant': "Yes"
                        })
                except Exception as e:
                    pass
        
        if has_time_data:
            diff = avg_male_time - avg_female_time
            faster = "Men" if diff < 0 else "Women"
            slower = "Women" if diff < 0 else "Men"
            abs_diff = abs(diff)
            
            summary_data.append({
                'Analysis': 'Time to Promotion (T-test)', 
                'Finding': f"{faster} promoted {abs_diff:.2f} years faster than {slower}", 
                'P-value': f"{p_val_time}",
                'Significant': "Yes" if p_val_time < 0.05 else "No"
            })
        
        # Display the summary table
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data))
        else:
            st.warning("No statistical findings available to summarize.")
        
        # Key findings narrative
        st.write("### Key Findings")
        
        # Promotion rates
        if has_chi2:
            m_rate = promotion_rates['M'] if 'M' in promotion_rates.index else 0
            f_rate = promotion_rates['F'] if 'F' in promotion_rates.index else 0
            
            if p < 0.05:
                if m_rate > f_rate:
                    st.write(f"- **Promotion Rates**: Men have significantly higher promotion rates ({m_rate:.1f}%) than women ({f_rate:.1f}%), p={p}.")
                else:
                    st.write(f"- **Promotion Rates**: Women have significantly higher promotion rates ({f_rate:.1f}%) than men ({m_rate:.1f}%), p={p}.")
            else:
                st.write(f"- **Promotion Rates**: No statistically significant difference in promotion rates between men ({m_rate:.1f}%) and women ({f_rate:.1f}%), p={p}.")
        
        # Logistic regression results
        if has_model:
            if gender_pval < 0.05:
                if gender_coef > 0:
                    st.write(f"- **Promotion Likelihood**: Men are {odds_ratio:.2f} times more likely to be promoted than women (p={gender_pval}).")
                else:
                    st.write(f"- **Promotion Likelihood**: Women are {1/odds_ratio:.2f} times more likely to be promoted than men (p={gender_pval}).")
            else:
                st.write(f"- **Promotion Likelihood**: No significant difference in promotion likelihood between men and women (p={gender_pval}).")
        
        # Time to promotion
        if has_time_data:
            if p_val_time < 0.05:
                if avg_male_time < avg_female_time:
                    st.write(f"- **Time to Promotion**: Women take {avg_female_time - avg_male_time:.2f} years longer than men to be promoted (p={p_val_time}).")
                else:
                    st.write(f"- **Time to Promotion**: Men take {avg_male_time - avg_female_time:.2f} years longer than women to be promoted (p={p_val_time}).")
            else:
                st.write(f"- **Time to Promotion**: No significant difference in time to promotion between men ({avg_male_time:.2f} years) and women ({avg_female_time:.2f} years), p={p_val_time}.")
        
        # Interaction effects
        admin_model = None
        field_model = None
        
        if has_model and has_admin_interaction:
            try:
                # Check admin interaction model
                X_admin = sm.add_constant(df_model[['sex_numeric', 'admin', 'sex_admin']].dropna())
                y_admin = df_model.loc[X_admin.index, 'promoted']
                admin_model = sm.Logit(y_admin, X_admin).fit(disp=0)
                
                if admin_model.pvalues['sex_admin'] < 0.05:
                    if admin_model.params['sex_admin'] > 0:
                        st.write(f"- **Administrative Duties**: The effect of administrative duties on promotion is significantly stronger for men than women (p={admin_model.pvalues['sex_admin']}).")
                    else:
                        st.write(f"- **Administrative Duties**: The effect of administrative duties on promotion is significantly stronger for women than men (p={admin_model.pvalues['sex_admin']}).")
            except Exception as e:
                pass
                
        if has_model and has_field_interaction:
            try:
                # Check field interaction model
                X_field = sm.add_constant(df_model[['sex_numeric', 'field_numeric', 'sex_field']].dropna())
                y_field = df_model.loc[X_field.index, 'promoted']
                field_model = sm.Logit(y_field, X_field).fit(disp=0)
                
                if field_model.pvalues['sex_field'] < 0.05:
                    if field_model.params['sex_field'] > 0:
                        st.write(f"- **Academic Field**: The effect of academic field on promotion varies by gender, with a stronger effect for men (p={field_model.pvalues['sex_field']}).")
                    else:
                        st.write(f"- **Academic Field**: The effect of academic field on promotion varies by gender, with a stronger effect for women (p={field_model.pvalues['sex_field']}).")
            except Exception as e:
                pass
        
        # Overall conclusion
        st.write("### Overall Conclusion")
        
        # Check for significant findings in any model
        sig_admin = admin_model is not None and admin_model.pvalues['sex_admin'] < 0.05
        sig_field = field_model is not None and field_model.pvalues['sex_field'] < 0.05
        
        if (has_chi2 and p < 0.05) or (has_model and gender_pval < 0.05) or (has_time_data and p_val_time < 0.05) or sig_admin or sig_field:
            st.write("Based on the analysis, there is evidence of sex differences in promotion outcomes. The specific patterns are detailed above.")
        else:
            st.write("Based on the analysis, there is no strong evidence of sex bias in promotion decisions after controlling for relevant factors.")
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.exception(e)
        st.write("Unable to generate comprehensive summary due to data issues.")

def run_analysis(uploaded_file):
    """Main function to run the Q4 promotion analysis"""
    # Create tabs for different analysis components
    tabs = st.tabs(["Exploratory Analysis", "Statistical Tests", "Final Summary"])
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        with tabs[0]:
            try:
                exploratory_analysis(df)
            except Exception as e:
                st.error(f"Error in exploratory analysis: {str(e)}")
                st.exception(e)
        
        with tabs[1]:
            try:
                statistical_tests(df)
            except Exception as e:
                st.error(f"Error in statistical tests: {str(e)}")
                st.exception(e)
        
        with tabs[2]:
            try:
                summary(df)
            except Exception as e:
                st.error(f"Error in summary: {str(e)}")
                st.exception(e)
    else:
        st.error("Unable to process data for Question 4. Please check file format and ensure it contains Associate and Full Professor data.")
