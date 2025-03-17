import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from lifelines import KaplanMeierFitter
from scipy import stats

# Define palette for consistent colors
custom_palette = {'M': 'skyblue', 'F': 'lightcoral'}

def load_data(uploaded_file, exclude_after=1990):
    """
    Load dataset exactly as in the correct file:
    - Always use whitespace-delimited reading.
    - Convert 'sex' to string.
    - Keep only Associate and Full Professors.
    - Exclude faculty with startyr > exclude_after.
    - Compute years as associate and assign promotion status.
    """
    try:
        # Always use whitespace-delimited reading (as in stats_q4.py)
        df = pd.read_csv(uploaded_file, delim_whitespace=True)
        df['sex'] = df['sex'].astype(str)
        
        # st.write("**Before Filtering** - Unique values in 'sex':", df['sex'].unique())
        
        # Keep only Associate and Full Professors and exclude later start years
        df = df[df['rank'].isin(['Assoc', 'Full'])]
        df = df[df['startyr'] <= exclude_after]
        
        # st.write("**After Filtering** - Unique values in 'sex':", df['sex'].unique())
        # st.write(df['sex'].value_counts(dropna=False))
        
        # Compute years as associate and assign promotion status
        df['years_as_assoc'] = df['year'] - df['startyr']
        full_professor_ids = df[df['rank'] == 'Full']['id'].unique()
        df['promoted'] = df.apply(lambda row: 1 if row['id'] in full_professor_ids else 0, axis=1)
        
        # Compute additional variables if available
        if 'yrdeg' in df.columns:
            df['age_at_promotion'] = df['year'] - df['yrdeg']
            df['experience'] = df['startyr'] - df['yrdeg']
        else:
            df['age_at_promotion'] = 0
            df['experience'] = 0
        
        if df['promoted'].sum() == 0:
            st.warning("No promotions found in this dataset. Some analyses will be limited.")
            
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)
        return None

def exploratory_analysis(df):
    """Perform exploratory analysis and visualizations."""
    try:
        st.subheader("Exploratory Analysis")
        # Ensure gender labels are displayed as 'M' and 'F'
        if df['sex'].dtype != object:
            df['sex'] = df['sex'].map({1: 'M', 0: 'F'})
        
        try:
            promotion_counts = pd.crosstab(df['sex'], df['promoted'], margins=True, margins_name='Total')
            st.write("Promotion Counts:")
            st.dataframe(promotion_counts)
        except Exception as e:
            st.error(f"Error creating promotion counts table: {str(e)}")
        
        try:
            promotion_rates = df.groupby('sex')['promoted'].mean() * 100
            promotion_rates = pd.DataFrame(promotion_rates).rename(columns={'promoted': 'Promotion Rate (%)'})
            st.write("Promotion Rates (%):")
            st.dataframe(promotion_rates)
        except Exception as e:
            st.error(f"Error calculating promotion rates: {str(e)}")
        
        st.write("### Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
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
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.countplot(x='promoted', hue='sex', data=df, palette=custom_palette, ax=ax2)
                ax2.set_xlabel('Promotion Status (0 = Not Promoted, 1 = Promoted)')
                ax2.set_ylabel('Count')
                ax2.set_title('Promotion Counts by Gender')
                ax2.legend(title='Sex')
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error creating promotion counts chart: {str(e)}")
        
        promoted_df = df[df['promoted'] == 1]
        if not promoted_df.empty and 'years_as_assoc' in promoted_df.columns:
            col3, col4 = st.columns(2)
            male_time = promoted_df[promoted_df['sex'] == 'M']['years_as_assoc']
            female_time = promoted_df[promoted_df['sex'] == 'F']['years_as_assoc']
            if not male_time.empty and not female_time.empty:
                with col3:
                    try:
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
                        fig4, ax4 = plt.subplots(figsize=(8, 6))
                        sns.histplot(data=promoted_df, x='years_as_assoc', hue='sex',
                                     bins=min(15, int(len(promoted_df)/5)), kde=True,
                                     palette=custom_palette, ax=ax4)
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
        
        if 'years_as_assoc' in df.columns and df['promoted'].sum() > 0:
            st.write("### Kaplan-Meier Survival Analysis")
            st.write("This shows the probability of remaining an Associate Professor over time:")
            try:
                df['years_as_assoc'] = pd.to_numeric(df['years_as_assoc'])
                df['promoted'] = pd.to_numeric(df['promoted'])
                df_survival = df.dropna(subset=['years_as_assoc', 'promoted'])
                if df_survival['promoted'].sum() == 0:
                    st.warning("No faculty members were promoted in this dataset. Cannot perform survival analysis.")
                else:
                    male_group = df_survival[df_survival['sex'] == 'M']
                    female_group = df_survival[df_survival['sex'] == 'F']
                    if male_group.empty or female_group.empty:
                        st.warning("Missing data for one or both gender groups. Cannot perform survival analysis.")
                    else:
                        if male_group['promoted'].sum() == 0 or female_group['promoted'].sum() == 0:
                            st.warning("No promotions for one gender group. Cannot compare survival curves.")
                        else:
                            kmf_male = KaplanMeierFitter()
                            kmf_female = KaplanMeierFitter()
                            kmf_male.fit(male_group['years_as_assoc'], event_observed=male_group['promoted'], label="Male")
                            kmf_female.fit(female_group['years_as_assoc'], event_observed=female_group['promoted'], label="Female")
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                fig5, ax5 = plt.subplots(figsize=(7, 5))
                                kmf_male.plot(ax=ax5)
                                kmf_female.plot(ax=ax5)
                                ax5.set_xlabel('Years as Associate Professor')
                                ax5.set_ylabel('Probability of Remaining Associate')
                                ax5.set_title('Kaplan-Meier Survival Curve: Time to Promotion')
                                ax5.legend()
                                ax5.grid(True)
                                st.pyplot(fig5)
                            st.write("""
**How to interpret this plot:**
- The y-axis shows the probability of remaining an Associate Professor.
- The x-axis shows years spent as Associate Professor.
- A steeper decline indicates faster promotion rates.
- Separate lines for men and women allow comparison of promotion patterns by gender.
""")
            except Exception as e:
                st.error(f"Error performing survival analysis: {str(e)}")
        else:
            st.info("Survival analysis requires years as Associate Professor data and at least some promotions.")
    except Exception as e:
        st.error(f"Error in exploratory analysis: {str(e)}")
        st.exception(e)

def statistical_tests(df):
    st.subheader("Statistical Tests")
    required_cols = ['sex', 'promoted']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for statistical tests: {', '.join(missing_cols)}")
        return

    ## Ensure proper gender labels
    if df['sex'].dtype != object:
        df['sex'] = df['sex'].map({1: 'M', 0: 'F'})

    ### Chi-Square Test: Promotion Rates by Gender
    st.write("### Chi-Square Test: Promotion Rates by Gender")
    try:
        promotion_table = pd.crosstab(df['sex'], df['promoted'])
        if (promotion_table.shape[0] < 2) or (promotion_table.shape[1] < 2):
            st.error("Not enough categories for chi-square test.")
            return
        if (promotion_table.values == 0).any():
            st.warning("Some cells have zero counts. Chi-square results may be unreliable.")
        chi2, p, dof, expected = chi2_contingency(promotion_table)
        chi_results = pd.DataFrame({
            'Statistic': ['Chi-Square Value', 'P-value', 'Degrees of Freedom'],
            'Value': [f"{chi2:.4f}", f"{p}", f"{dof}"]
        })
        st.dataframe(chi_results)
        if p < 0.05:
            st.write("**Interpretation**: Promotion rates significantly differ by gender (p < 0.05).")
            m_rate = promotion_table.loc['M', 1] / promotion_table.loc['M'].sum() * 100
            f_rate = promotion_table.loc['F', 1] / promotion_table.loc['F'].sum() * 100
            if m_rate > f_rate:
                st.write(f"Men have a higher promotion rate ({m_rate:.1f}%) than women ({f_rate:.1f}%).")
            else:
                st.write(f"Women have a higher promotion rate ({f_rate:.1f}%) than men ({m_rate:.1f}%).")
        else:
            st.write("**Interpretation**: No significant difference in promotion rates by gender (p > 0.05).")
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

    ### T-Test: Time to Promotion by Gender
    st.write("### T-Test: Time to Promotion by Gender")
    try:
        promoted_df = df[df['promoted'] == 1]
        if promoted_df.empty or 'years_as_assoc' not in promoted_df.columns:
            st.warning("No promoted faculty or years_as_assoc data. Cannot analyze time to promotion.")
        else:
            male_time = promoted_df[promoted_df['sex'] == 'M']['years_as_assoc']
            female_time = promoted_df[promoted_df['sex'] == 'F']['years_as_assoc']
            if male_time.empty or female_time.empty or len(male_time) < 2 or len(female_time) < 2:
                st.warning("Not enough promoted faculty in one or both gender groups for comparison.")
            else:
                t_stat, p_val = stats.ttest_ind(male_time, female_time, equal_var=False)
                male_mean = male_time.mean()
                female_mean = female_time.mean()
                time_results = pd.DataFrame({
                    'Statistic': ['T-value', 'P-value', 'Mean time for men', 'Mean time for women', 'Difference in means'],
                    'Value': [f"{t_stat:.4f}", f"{p_val}", f"{male_mean:.2f} years", f"{female_mean:.2f} years", f"{male_mean - female_mean:.2f} years"]
                })
                st.dataframe(time_results)
                if p_val < 0.05:
                    if male_mean < female_mean:
                        st.write(f"**Interpretation**: There is a statistically significant difference in time to promotion. Women take {female_mean - male_mean:.2f} years longer than men on average (p={p_val}).")
                    else:
                        st.write(f"**Interpretation**: There is a statistically significant difference in time to promotion. Men take {male_mean - female_mean:.2f} years longer than women on average (p={p_val}).")
                else:
                    st.write(f"**Interpretation**: There is no statistically significant difference in time to promotion between men ({male_mean:.2f} years) and women ({female_mean:.2f} years), p={p_val}.")
    except Exception as e:
        st.error(f"Error performing t-test on promotion time: {str(e)}")

    ### Prepare data for regression models
    st.write("### Logistic Regression Models")
    try:
        df_model = df.copy()
        df_model['sex_numeric'] = df_model['sex'].map({'M': 1, 'F': 0})
        for col in ['years_as_assoc', 'promoted', 'age_at_promotion', 'experience']:
            if col in df_model.columns:
                df_model[col] = pd.to_numeric(df_model[col])
        if len(df_model) < 10:
            st.error("Not enough data for regression models after cleaning.")
            return

        #### Model 1: Gender Only
        st.write("#### Model 1: Gender Only")
        try:
            X1 = df_model[['sex_numeric']]
            X1 = sm.add_constant(X1)
            y = df_model['promoted']
            model1 = sm.Logit(y, X1).fit(disp=0)
            coef = model1.params['sex_numeric']
            p_val = model1.pvalues['sex_numeric']
            odds_ratio = np.exp(coef)
            results_data = [
                {'Statistic': 'Gender Coefficient', 'Value': f"{coef:.4f}"},
                {'Statistic': 'P-value (Gender)', 'Value': f"{p_val}"},
                {'Statistic': 'Odds Ratio (Gender)', 'Value': f"{odds_ratio:.4f}"}
            ]
            model1_results = pd.DataFrame(results_data)
            st.dataframe(model1_results)
            if p_val < 0.05:
                if coef > 0:
                    st.write(f"**Interpretation**: Men are {odds_ratio:.2f} times more likely to be promoted than women (p={p_val}).")
                else:
                    st.write(f"**Interpretation**: Women are {1/odds_ratio:.2f} times more likely to be promoted than men (p={p_val}).")
            else:
                st.write(f"**Interpretation**: No significant difference in promotion likelihood between men and women (p={p_val}).")
            model1_stats = pd.DataFrame({
                'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                'Value': ['promoted (0/1)', f"{model1.prsquared:.4f}", f"{model1.llf:.4f}", f"{model1.llr_pvalue}", f"{model1.aic:.4f}", f"{model1.bic:.4f}", f"{model1.nobs}"]
            })
            st.write("**Model Statistics:**")
            st.dataframe(model1_stats)
            model1_coef = pd.DataFrame({
                'Variable': ['Intercept', 'sex_numeric (Male=1)'],
                'Coefficient': [f"{model1.params['const']:.4f}", f"{model1.params['sex_numeric']:.4f}"],
                'Std Error': [f"{model1.bse['const']:.4f}", f"{model1.bse['sex_numeric']:.4f}"],
                'Z-value': [f"{model1.tvalues['const']:.4f}", f"{model1.tvalues['sex_numeric']:.4f}"],
                'P-value': [f"{model1.pvalues['const']}", f"{model1.pvalues['sex_numeric']}"],
                'Odds Ratio': [f"{np.exp(model1.params['const']):.4f}", f"{np.exp(model1.params['sex_numeric']):.4f}"]
            })
            st.write("**Coefficients:**")
            st.dataframe(model1_coef)
            with st.expander("View Raw Model 1 Output"):
                st.text(model1.summary().as_text())
        except Exception as e:
            st.error(f"Error running Model 1: {str(e)}")
            model1 = None

        #### Model 2: Gender + Years as Associate
        st.write("#### Model 2: Gender + Years as Associate")
        try:
            X2 = df_model[['sex_numeric', 'years_as_assoc']]
            X2 = sm.add_constant(X2)
            y2 = df_model['promoted']
            model2 = sm.Logit(y2, X2).fit(disp=0)
            results_data = [
                {'Statistic': 'Gender Coefficient', 'Value': f"{model2.params['sex_numeric']:.4f}"},
                {'Statistic': 'P-value (Gender)', 'Value': f"{model2.pvalues['sex_numeric']}"},
                {'Statistic': 'Odds Ratio (Gender)', 'Value': f"{np.exp(model2.params['sex_numeric']):.4f}"},
                {'Statistic': 'Years as Associate Coefficient', 'Value': f"{model2.params['years_as_assoc']:.4f}"},
                {'Statistic': 'P-value (Years as Associate)', 'Value': f"{model2.pvalues['years_as_assoc']}"},
                {'Statistic': 'Odds Ratio (Years as Associate)', 'Value': f"{np.exp(model2.params['years_as_assoc']):.4f}"}
            ]
            model2_results = pd.DataFrame(results_data)
            st.dataframe(model2_results)
            if model2.pvalues['sex_numeric'] < 0.05:
                if model2.params['sex_numeric'] > 0:
                    st.write(f"**Interpretation**: After controlling for years as associate, men are {np.exp(model2.params['sex_numeric']):.2f} times more likely to be promoted than women (p={model2.pvalues['sex_numeric']}).")
                else:
                    st.write(f"**Interpretation**: After controlling for years as associate, women are {1/np.exp(model2.params['sex_numeric']):.2f} times more likely to be promoted than men (p={model2.pvalues['sex_numeric']}).")
            else:
                st.write(f"**Interpretation**: No significant difference in promotion likelihood by gender after controlling for years as associate (p={model2.pvalues['sex_numeric']}).")
            model2_stats = pd.DataFrame({
                'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                'Value': ['promoted (0/1)', f"{model2.prsquared:.4f}", f"{model2.llf:.4f}", f"{model2.llr_pvalue}", f"{model2.aic:.4f}", f"{model2.bic:.4f}", f"{model2.nobs}"]
            })
            st.write("**Model Statistics:**")
            st.dataframe(model2_stats)
            model2_coef = pd.DataFrame({
                'Variable': ['Intercept', 'sex_numeric (Male=1)', 'years_as_assoc'],
                'Coefficient': [f"{model2.params['const']:.4f}", f"{model2.params['sex_numeric']:.4f}", f"{model2.params['years_as_assoc']:.4f}"],
                'Std Error': [f"{model2.bse['const']:.4f}", f"{model2.bse['sex_numeric']:.4f}", f"{model2.bse['years_as_assoc']:.4f}"],
                'Z-value': [f"{model2.tvalues['const']:.4f}", f"{model2.tvalues['sex_numeric']:.4f}", f"{model2.tvalues['years_as_assoc']:.4f}"],
                'P-value': [f"{model2.pvalues['const']}", f"{model2.pvalues['sex_numeric']}", f"{model2.pvalues['years_as_assoc']}"],
                'Odds Ratio': [f"{np.exp(model2.params['const']):.4f}", f"{np.exp(model2.params['sex_numeric']):.4f}", f"{np.exp(model2.params['years_as_assoc']):.4f}"]
            })
            st.write("**Coefficients:**")
            st.dataframe(model2_coef)
            with st.expander("View Raw Model 2 Output"):
                st.text(model2.summary().as_text())
        except Exception as e:
            st.error(f"Error running Model 2: {str(e)}")
            model2 = None

        #### Model 3: Gender + Years as Associate + Age & Experience
        st.write("#### Model 3: Gender + Years as Associate + Age & Experience")
        try:
            X3 = df_model[['sex_numeric', 'years_as_assoc', 'age_at_promotion', 'experience']]
            X3 = sm.add_constant(X3)
            y3 = df_model['promoted']
            model3 = sm.Logit(y3, X3).fit(disp=0)
            results_data = [
                {'Statistic': 'Gender Coefficient', 'Value': f"{model3.params['sex_numeric']:.4f}"},
                {'Statistic': 'P-value (Gender)', 'Value': f"{model3.pvalues['sex_numeric']}"},
                {'Statistic': 'Odds Ratio (Gender)', 'Value': f"{np.exp(model3.params['sex_numeric']):.4f}"},
                {'Statistic': 'Years as Associate Coefficient', 'Value': f"{model3.params['years_as_assoc']:.4f}"},
                {'Statistic': 'P-value (Years as Associate)', 'Value': f"{model3.pvalues['years_as_assoc']}"},
                {'Statistic': 'Odds Ratio (Years as Associate)', 'Value': f"{np.exp(model3.params['years_as_assoc']):.4f}"},
                {'Statistic': 'Age at Promotion Coefficient', 'Value': f"{model3.params['age_at_promotion']:.4f}"},
                {'Statistic': 'P-value (Age at Promotion)', 'Value': f"{model3.pvalues['age_at_promotion']}"},
                {'Statistic': 'Odds Ratio (Age at Promotion)', 'Value': f"{np.exp(model3.params['age_at_promotion']):.4f}"},
                {'Statistic': 'Experience Coefficient', 'Value': f"{model3.params['experience']:.4f}"},
                {'Statistic': 'P-value (Experience)', 'Value': f"{model3.pvalues['experience']}"},
                {'Statistic': 'Odds Ratio (Experience)', 'Value': f"{np.exp(model3.params['experience']):.4f}"}
            ]
            model3_results = pd.DataFrame(results_data)
            st.dataframe(model3_results)
            if model3.pvalues['sex_numeric'] < 0.05:
                if model3.params['sex_numeric'] > 0:
                    st.write(f"**Interpretation**: Men are {np.exp(model3.params['sex_numeric']):.2f} times more likely to be promoted than women (p={model3.pvalues['sex_numeric']}).")
                else:
                    st.write(f"**Interpretation**: Women are {1/np.exp(model3.params['sex_numeric']):.2f} times more likely to be promoted than men (p={model3.pvalues['sex_numeric']}).")
            else:
                st.write(f"**Interpretation**: No significant difference in promotion likelihood between genders (p={model3.pvalues['sex_numeric']}).")
            model3_stats = pd.DataFrame({
                'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                'Value': ['promoted (0/1)', f"{model3.prsquared:.4f}", f"{model3.llf:.4f}", f"{model3.llr_pvalue}", f"{model3.aic:.4f}", f"{model3.bic:.4f}", f"{model3.nobs}"]
            })
            st.write("**Model Statistics:**")
            st.dataframe(model3_stats)
            model3_coef = pd.DataFrame({
                'Variable': ['Intercept', 'sex_numeric (Male=1)', 'years_as_assoc', 'age_at_promotion', 'experience'],
                'Coefficient': [f"{model3.params['const']:.4f}", f"{model3.params['sex_numeric']:.4f}",
                                f"{model3.params['years_as_assoc']:.4f}", f"{model3.params['age_at_promotion']:.4f}", f"{model3.params['experience']:.4f}"],
                'Std Error': [f"{model3.bse['const']:.4f}", f"{model3.bse['sex_numeric']:.4f}",
                              f"{model3.bse['years_as_assoc']:.4f}", f"{model3.bse['age_at_promotion']:.4f}", f"{model3.bse['experience']:.4f}"],
                'Z-value': [f"{model3.tvalues['const']:.4f}", f"{model3.tvalues['sex_numeric']:.4f}",
                            f"{model3.tvalues['years_as_assoc']:.4f}", f"{model3.tvalues['age_at_promotion']:.4f}", f"{model3.tvalues['experience']:.4f}"],
                'P-value': [f"{model3.pvalues['const']}", f"{model3.pvalues['sex_numeric']}",
                            f"{model3.pvalues['years_as_assoc']}", f"{model3.pvalues['age_at_promotion']}", f"{model3.pvalues['experience']}"],
                'Odds Ratio': [f"{np.exp(model3.params['const']):.4f}", f"{np.exp(model3.params['sex_numeric']):.4f}",
                               f"{np.exp(model3.params['years_as_assoc']):.4f}", f"{np.exp(model3.params['age_at_promotion']):.4f}", f"{np.exp(model3.params['experience']):.4f}"]
            })
            st.write("**Coefficients:**")
            st.dataframe(model3_coef)
            with st.expander("View Raw Model 3 Output"):
                st.text(model3.summary().as_text())
        except Exception as e:
            st.error(f"Error running Model 3: {str(e)}")
        
        #### Model 4: Full Model with All Variables
        st.write("#### Model 4: Full Model with All Variables")
        try:
            df_model_full = df_model.copy()
            if 'field' in df_model_full.columns:
                df_model_full['field_numeric'] = df_model_full['field'].astype('category').cat.codes
            if 'deg' in df_model_full.columns:
                df_model_full['deg_numeric'] = df_model_full['deg'].astype('category').cat.codes
            X4 = df_model_full[['sex_numeric', 'years_as_assoc', 'age_at_promotion', 'experience']]
            if 'field_numeric' in df_model_full.columns:
                X4['field'] = df_model_full['field_numeric']
            if 'deg_numeric' in df_model_full.columns:
                X4['deg'] = df_model_full['deg_numeric']
            if 'admin' in df_model_full.columns:
                X4['admin'] = pd.to_numeric(df_model_full['admin'])
            X4 = sm.add_constant(X4)
            y4 = df_model_full['promoted']
            model4 = sm.Logit(y4, X4).fit(disp=0)
            results_data = []
            for var in model4.params.index:
                results_data.append({
                    'Variable': var,
                    'Coefficient': f"{model4.params[var]:.4f}",
                    'Std Error': f"{model4.bse[var]:.4f}",
                    'Z-value': f"{model4.tvalues[var]:.4f}",
                    'P-value': f"{model4.pvalues[var]}",
                    'Odds Ratio': f"{np.exp(model4.params[var]):.4f}"
                })
            model4_results = pd.DataFrame(results_data)
            st.dataframe(model4_results)
            if model4.pvalues['sex_numeric'] < 0.05:
                if model4.params['sex_numeric'] > 0:
                    st.write(f"**Interpretation**: Men are {np.exp(model4.params['sex_numeric']):.2f} times more likely to be promoted than women (p={model4.pvalues['sex_numeric']}).")
                else:
                    st.write(f"**Interpretation**: Women are {1/np.exp(model4.params['sex_numeric']):.2f} times more likely to be promoted than men (p={model4.pvalues['sex_numeric']}).")
            else:
                st.write(f"**Interpretation**: No significant difference in promotion likelihood between genders (p={model4.pvalues['sex_numeric']}).")
            model4_stats = pd.DataFrame({
                'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                'Value': ['promoted (0/1)', f"{model4.prsquared:.4f}", f"{model4.llf:.4f}", f"{model4.llr_pvalue}", f"{model4.aic:.4f}", f"{model4.bic:.4f}", f"{model4.nobs}"]
            })
            st.write("**Model Statistics:**")
            st.dataframe(model4_stats)
            with st.expander("View Raw Model 4 Output"):
                st.text(model4.summary().as_text())
        except Exception as e:
            st.error(f"Error running Model 4: {str(e)}")
        
        #### Model 5: Interaction - Sex x Admin Role
        st.write("#### Model 5: Interaction - Sex x Admin Role")
        try:
            if 'admin' not in df_model.columns:
                st.warning("Administrative duties column ('admin') not found. Cannot run interaction model.")
            else:
                df_model['admin'] = pd.to_numeric(df_model['admin'])
                df_model['sex_admin'] = df_model['sex_numeric'] * df_model['admin']
                X5 = df_model[['sex_numeric', 'admin', 'sex_admin']]
                X5 = sm.add_constant(X5)
                y5 = df_model['promoted']
                model5 = sm.Logit(y5, X5).fit(disp=0)
                results_data = []
                for var in model5.params.index:
                    results_data.append({
                        'Variable': var,
                        'Coefficient': f"{model5.params[var]:.4f}",
                        'Std Error': f"{model5.bse[var]:.4f}",
                        'Z-value': f"{model5.tvalues[var]:.4f}",
                        'P-value': f"{model5.pvalues[var]}",
                        'Odds Ratio': f"{np.exp(model5.params[var]):.4f}"
                    })
                model5_results = pd.DataFrame(results_data)
                st.dataframe(model5_results)
                if model5.pvalues['sex_admin'] < 0.05:
                    if model5.params['sex_admin'] > 0:
                        st.write(f"**Interpretation**: The interaction indicates that the effect of admin duties is significantly stronger for men (p={model5.pvalues['sex_admin']}).")
                    else:
                        st.write(f"**Interpretation**: The interaction indicates that the effect of admin duties is significantly stronger for women (p={model5.pvalues['sex_admin']}).")
                else:
                    st.write(f"**Interpretation**: No significant interaction between sex and admin duties (p={model5.pvalues['sex_admin']}).")
                model5_stats = pd.DataFrame({
                    'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                    'Value': ['promoted (0/1)', f"{model5.prsquared:.4f}", f"{model5.llf:.4f}", f"{model5.llr_pvalue}", f"{model5.aic:.4f}", f"{model5.bic:.4f}", f"{model5.nobs}"]
                })
                st.write("**Model Statistics:**")
                st.dataframe(model5_stats)
                with st.expander("View Raw Model 5 Output"):
                    st.text(model5.summary().as_text())
        except Exception as e:
            st.error(f"Error running Model 5: {str(e)}")
        
        #### Model 6: Interaction - Sex x Academic Field
        st.write("#### Model 6: Interaction - Sex x Academic Field")
        try:
            if 'field' not in df_model.columns:
                st.warning("Academic field column ('field') not found. Cannot run field interaction model.")
            else:
                df_model['field_numeric'] = df_model['field'].astype('category').cat.codes
                df_model['sex_field'] = df_model['sex_numeric'] * df_model['field_numeric']
                X6 = df_model[['sex_numeric', 'field_numeric', 'sex_field']]
                X6 = sm.add_constant(X6)
                y6 = df_model['promoted']
                model6 = sm.Logit(y6, X6).fit(disp=0)
                results_data = []
                for var in model6.params.index:
                    results_data.append({
                        'Variable': var,
                        'Coefficient': f"{model6.params[var]:.4f}",
                        'Std Error': f"{model6.bse[var]:.4f}",
                        'Z-value': f"{model6.tvalues[var]:.4f}",
                        'P-value': f"{model6.pvalues[var]}",
                        'Odds Ratio': f"{np.exp(model6.params[var]):.4f}"
                    })
                model6_results = pd.DataFrame(results_data)
                st.dataframe(model6_results)
                if model6.pvalues['sex_field'] < 0.05:
                    if model6.params['sex_field'] > 0:
                        st.write(f"**Interpretation**: The interaction indicates that the effect of academic field is significantly stronger for men (p={model6.pvalues['sex_field']}).")
                    else:
                        st.write(f"**Interpretation**: The interaction indicates that the effect of academic field is significantly stronger for women (p={model6.pvalues['sex_field']}).")
                else:
                    st.write(f"**Interpretation**: No significant interaction between sex and academic field (p={model6.pvalues['sex_field']}).")
                model6_stats = pd.DataFrame({
                    'Metric': ['Dependent Variable', 'Pseudo R-squared', 'Log-Likelihood', 'LLR p-value', 'AIC', 'BIC', 'Observations'],
                    'Value': ['promoted (0/1)', f"{model6.prsquared:.4f}", f"{model6.llf:.4f}", f"{model6.llr_pvalue}", f"{model6.aic:.4f}", f"{model6.bic:.4f}", f"{model6.nobs}"]
                })
                st.write("**Model Statistics:**")
                st.dataframe(model6_stats)
                with st.expander("View Raw Model 6 Output"):
                    st.text(model6.summary().as_text())
        except Exception as e:
            st.error(f"Error running Model 6: {str(e)}")
    except Exception as e:
        st.error(f"Error in logistic regression analysis: {str(e)}")
        st.exception(e)
        
def summary(df):
    st.subheader("Final Summary: Sex Bias in Promotion Decisions")
    if df is None or df.empty:
        st.error("No data available for summary.")
        return
    try:
        if df['sex'].dtype != object:
            df['sex'] = df['sex'].map({1: 'M', 0: 'F'})
        
        try:
            promotion_table = pd.crosstab(df['sex'], df['promoted'])
            promotion_rates = df.groupby('sex')['promoted'].mean() * 100
            chi2, p, dof, expected = chi2_contingency(promotion_table)
            has_chi2 = True
        except Exception as e:
            st.warning(f"Could not calculate promotion statistics: {str(e)}")
            has_chi2 = False
        
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
        
        try:
            df_model = df.copy()
            df_model['sex_numeric'] = df_model['sex'].map({'M': 1, 'F': 0})
            for col in ['years_as_assoc', 'promoted', 'age_at_promotion', 'experience']:
                if col in df_model.columns:
                    df_model[col] = pd.to_numeric(df_model[col])
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
            if 'admin' in df_model.columns:
                df_model['admin'] = pd.to_numeric(df_model['admin'])
                df_model['sex_admin'] = df_model['sex_numeric'] * df_model['admin']
                has_admin_interaction = True
            else:
                has_admin_interaction = False
            if 'field' in df_model.columns:
                df_model['field_numeric'] = df_model['field'].astype('category').cat.codes
                df_model['sex_field'] = df_model['sex_numeric'] * df_model['field_numeric']
                has_field_interaction = True
            else:
                has_field_interaction = False
        except Exception as e:
            has_model = False
            has_admin_interaction = False
            has_field_interaction = False
            
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
            if has_admin_interaction:
                try:
                    X_admin = sm.add_constant(df_model[['sex_numeric', 'admin', 'sex_admin']])
                    y_admin = df_model['promoted']
                    admin_model = sm.Logit(y_admin, X_admin).fit(disp=0)
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
                    X_field = sm.add_constant(df_model[['sex_numeric', 'field_numeric', 'sex_field']])
                    y_field = df_model['promoted']
                    field_model = sm.Logit(y_field, X_field).fit(disp=0)
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
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data))
        else:
            st.warning("No statistical findings available to summarize.")
        
        st.write("### Key Findings")
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
        if has_model:
            if gender_pval < 0.05:
                if gender_coef > 0:
                    st.write(f"- **Promotion Likelihood**: Men are {odds_ratio:.2f} times more likely to be promoted than women (p={gender_pval}).")
                else:
                    st.write(f"- **Promotion Likelihood**: Women are {1/odds_ratio:.2f} times more likely to be promoted than men (p={gender_pval}).")
            else:
                st.write(f"- **Promotion Likelihood**: No significant difference in promotion likelihood between men and women (p={gender_pval}).")
        if has_time_data:
            if p_val_time < 0.05:
                if avg_male_time < avg_female_time:
                    st.write(f"- **Time to Promotion**: Women take {avg_female_time - avg_male_time:.2f} years longer than men to be promoted (p={p_val_time}).")
                else:
                    st.write(f"- **Time to Promotion**: Men take {avg_male_time - avg_female_time:.2f} years longer than women to be promoted (p={p_val_time}).")
            else:
                st.write(f"- **Time to Promotion**: No significant difference in time to promotion between men ({avg_male_time:.2f} years) and women ({avg_female_time:.2f} years), p={p_val_time}.")
        if has_model and has_admin_interaction:
            try:
                X_admin = sm.add_constant(df_model[['sex_numeric', 'admin', 'sex_admin']])
                y_admin = df_model['promoted']
                admin_model = sm.Logit(y_admin, X_admin).fit(disp=0)
                if admin_model.pvalues['sex_admin'] < 0.05:
                    if admin_model.params['sex_admin'] > 0:
                        st.write(f"- **Administrative Duties**: The effect of admin duties on promotion is significantly stronger for men (p={admin_model.pvalues['sex_admin']}).")
                    else:
                        st.write(f"- **Administrative Duties**: The effect of admin duties on promotion is significantly stronger for women (p={admin_model.pvalues['sex_admin']}).")
            except Exception as e:
                pass
        if has_model and has_field_interaction:
            try:
                X_field = sm.add_constant(df_model[['sex_numeric', 'field_numeric', 'sex_field']])
                y_field = df_model['promoted']
                field_model = sm.Logit(y_field, X_field).fit(disp=0)
                if field_model.pvalues['sex_field'] < 0.05:
                    if field_model.params['sex_field'] > 0:
                        st.write(f"- **Academic Field**: The effect of academic field on promotion varies by gender, with a stronger effect for men (p={field_model.pvalues['sex_field']}).")
                    else:
                        st.write(f"- **Academic Field**: The effect of academic field on promotion varies by gender, with a stronger effect for women (p={field_model.pvalues['sex_field']}).")
            except Exception as e:
                pass
        
        st.write("### Overall Conclusion")
        sig_admin = False
        sig_field = False
        try:
            if has_model and has_admin_interaction:
                X_admin = sm.add_constant(df_model[['sex_numeric', 'admin', 'sex_admin']])
                y_admin = df_model['promoted']
                admin_model = sm.Logit(y_admin, X_admin).fit(disp=0)
                sig_admin = admin_model.pvalues['sex_admin'] < 0.05
        except Exception as e:
            pass
        try:
            if has_model and has_field_interaction:
                X_field = sm.add_constant(df_model[['sex_numeric', 'field_numeric', 'sex_field']])
                y_field = df_model['promoted']
                field_model = sm.Logit(y_field, X_field).fit(disp=0)
                sig_field = field_model.pvalues['sex_field'] < 0.05
        except Exception as e:
            pass
        if (has_chi2 and p < 0.05) or (has_model and gender_pval < 0.05) or (has_time_data and p_val_time < 0.05) or sig_admin or sig_field:
            st.write("Based on the analysis, there is evidence of sex differences in promotion outcomes. The specific patterns are detailed above.")
        else:
            st.write("Based on the analysis, there is no strong evidence of sex bias in promotion decisions after controlling for relevant factors.")
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.exception(e)
        st.write("Unable to generate comprehensive summary due to data issues.")

def run_analysis(uploaded_file):
    """Main function to run the Q4 promotion analysis."""
    tabs = st.tabs(["Exploratory Analysis", "Statistical Tests", "Final Summary"])
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

if __name__ == '__main__':
    st.write("Q4 Promotion Decisions Analysis Module")
    uploaded_file = st.file_uploader("Upload data file", type=["csv", "txt"])
    if uploaded_file is not None:
        run_analysis(uploaded_file)
