# Statistical Test Analysis Dashboard

A Streamlit dashboard for analyzing potential gender bias in academic faculty salary and promotion data.

## Overview

This application provides statistical analysis and visualization tools to examine potential gender bias across four key areas:

1. **Recent Year Analysis** - Examines gender bias in the most current year available (1995)
2. **Starting Salaries** - Analyzes gender differences in faculty starting salaries
3. **Salary Increases** - Evaluates gender bias in salary increases between 1990-1995
4. **Promotion Decisions** - Assesses gender bias in promotion from Associate to Full Professor

## Features

- Interactive data analysis with real-time visualization
- Comprehensive statistical testing including t-tests, chi-square tests, and regression models
- Multiple visualization types (box plots, histograms, Kaplan-Meier survival curves)
- Detailed interpretation of findings and statistical significance
- Summary reporting of key findings for each analysis type

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/faculty-analysis.git
cd faculty-analysis

# Install required packages
pip install -r requirements.txt
```

## Required Packages

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- streamlit
- lifelines
- cpi

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

## Data Format

The application expects faculty data in a text file (CSV or space/tab-separated) with the following columns:

- **id**: Faculty identifier
- **year**: Year of observation
- **sex**: Gender (M/F)
- **deg**: Degree (e.g., PhD, Masters)
- **field**: Academic field (e.g., Arts, Sciences, Prof)
- **rank**: Academic rank (e.g., Full, Assoc, Assist)
- **admin**: Administrative duties (0/1)
- **salary**: Annual salary
- **startyr**: Start year
- **yrdeg**: Year degree was obtained

## Project Structure

- `app.py` - Main Streamlit application entry point
- `scripts/` - Module directory containing analysis scripts
  - `q1_current_salaries.py` - Analysis of gender bias in current salaries
  - `q2_starting_salaries.py` - Analysis of gender bias in starting salaries
  - `q3_salary_increases.py` - Analysis of gender bias in salary increases
  - `q4_promotion_decisions.py` - Analysis of gender bias in promotion decisions
- `requirements.txt` - Required Python packages
- `salary/` - Data for salary file used in the analysis

## Acknowledgements

Final Course Project for BIOST 557 A - Applied Statistics And Experimental Design (University of Washington) 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
