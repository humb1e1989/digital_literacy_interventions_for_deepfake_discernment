import config
import pandas as pd
import numpy as np
from statsmodels.iolib.summary2 import summary_col
from statsmodels.formula.api import ols
from logger import get_logger
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = get_logger(__name__)
# print all pandas columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Function to run and summarize regression
def run_regression(data, formula, group=None, check_multicollinearity=False):
    model = ols(formula, data=data).fit()
    # Get p-values and coefficients
    p_values = model.pvalues
    coefficients = model.params
    # Create a DataFrame with both p-values and coefficients
    significant_predictors = pd.DataFrame({'Coefficient': coefficients, 'P-value': p_values})
    # Filter for significant predictors (p < 0.05)
    significant_predictors = significant_predictors[significant_predictors['P-value'] < 0.05]
    # Replace with display names
    significant_predictors.index = significant_predictors.index.map(
        lambda x: config.variable_display_names.get(x, x)
    )

    logger.info("\nSignificant predictors:")
    for predictor, row in significant_predictors.iterrows():
        logger.info(f"{predictor}: Coefficient: {row['Coefficient']:.4f} (p = {row['P-value']:.4f})")

        # Optional multicollinearity analysis
        if check_multicollinearity:
            analyze_multicollinearity(model)

        return model

def analyze_multicollinearity(model):
    n_predictors = model.model.exog.shape[1] - 1
    if n_predictors <= 1:
        return

    try:
        X = model.model.exog[:, 1:]  # Exclude intercept
        feature_names = model.model.exog_names[1:]

        # Calculate VIF for each variable
        vif_data = []
        for i in range(X.shape[1]):
            vif_value = variance_inflation_factor(X, i)
            if np.isinf(vif_value) or vif_value > 1000:
                vif_value = np.nan
            vif_data.append({'Variable': feature_names[i], 'VIF': vif_value})

        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False, na_position='last')

        # Print VIF for all variables
        logger.info(f"\nMulticollinearity Analysis (VIF values):")
        for _, row in vif_df.iterrows():
            var_display = config.variable_display_names.get(row['Variable'], row['Variable'])
            if pd.isna(row['VIF']):
                logger.info(f"   {var_display}: VIF = N/A (calculation failed)")
            else:
                logger.info(f"   {var_display}: VIF = {row['VIF']:.2f}")

    except Exception as e:
        logger.warning(f"VIF analysis failed: {e}")

# Create interaction formulas
def add_interactions(formula):
    # Get the covariates (everything after the + signs)
    base_vars = formula.split(' + ')[1:]  # Skip the dependent variable and intervention
    # Add interaction terms
    interactions = [f'C(intervention_group):{var}' for var in base_vars]
    return formula + ' + ' + ' + '.join(interactions)


def describe_regression_data(df):
    # describe the data
    logger.info(f"Descriptives demographics main before filtering:")
    # logger.info(f"Counts gender binary (0=male; 1=female):\n{df['gender_binary'].value_counts()}")
    logger.info(f"Counts gender: \n{df['gender'].value_counts()}, \n{df['gender'].value_counts(normalize=True)}")
    df['age_group'] = pd.cut(df['age'], bins=[0, 34, 54, np.inf], labels=[0, 1, 2]).astype('float64')
    logger.info(f"Counts age:\n{df['age'].describe()}, \n{df['age_group'].value_counts()}, \n{df['age_group'].value_counts(normalize=True)}")
    logger.info(f"Counts ethnicity groups:\n{df['ethnicity'].value_counts()}, \n{df['ethnicity'].value_counts(normalize=True)}")
    logger.info(f"Counts education groups:\n{df['education'].value_counts()}, \n{df['education'].value_counts(normalize=True)}")
    logger.info(f"Counts religion groups:\n{df['religion'].value_counts()}, \n{df['religion'].value_counts(normalize=True)}")
    df['political_orientation'] = ((df['social_issues'] + df['economic_issues']) / 2).astype(
        'float64')
    logger.info(f"Counts political orientation groups:\n{df['political_orientation'].value_counts()}, \n{df['political_orientation'].value_counts(normalize=True)}")
    logger.info(f"Counts income groups:\n{df['income_us'].value_counts()}, \n{df['income_us'].value_counts(normalize=True)}")
    # logger.info(df[config.demographic_regression].describe())

def describe_recoded_data(df):
    # describe the data
    logger.info(f"Descriptives demographics main before filtering:")
    # logger.info(f"Counts gender binary (0=male; 1=female):\n{df['gender_binary'].value_counts()}")
    logger.info(f"Counts gender binary: \n{df['gender_binary'].value_counts()}")
    logger.info(f"Counts age:\n{df['age'].describe()}")
    logger.info(f"Counts ethnicity groups:\n{df['ethnicity_group'].value_counts()}")
    logger.info(f"Counts education groups:\n{df['education_group'].value_counts()}")
    logger.info(f"Counts religion groups:\n{df['religion_recoded'].value_counts()}")
    logger.info(f"Counts political orientation groups:\n{df['political_orientation'].value_counts()}")
    logger.info(f"Counts income groups:\n{df['income_group'].value_counts()}")
    logger.info(df[config.demographic_regression].describe())


def compute_cohens_d(df_merged, accuracy, model):
    # Get means for each group
    group_means = df_merged.groupby('intervention_group')[accuracy].mean()
    # Get pooled standard deviation from residuals
    pooled_sd = np.sqrt(np.sum(model.resid ** 2) / model.df_resid)
    # Calculate Cohen's d for each intervention vs control
    cohen_d = {}
    control_mean = group_means['control']
    for group in ['intervention_1', 'intervention_2', 'intervention_3', 'intervention_4', 'intervention_5']:
        mean_diff = group_means[group] - control_mean
        d = mean_diff / pooled_sd
        cohen_d[group] = d
        logger.debug(f"Cohen's d for {group} vs control: {d:.3f}")
    return cohen_d

def create_regression_table(models, model_names, dependent_var, variable_display_names,
                                file_name="regression_results.txt", significance_levels=(0.05, 0.01, 0.001)):
    # Create summary of models
    results_summary = summary_col(models, model_names=model_names, stars=False,
                                  float_format='%0.3f', info_dict={'N': lambda x: int(x.nobs)})

    # Convert to DataFrame
    results_df = results_summary.tables[0]

    # Function to add stars based on p-values
    def add_stars(coef, pvalue):
        if pvalue <= significance_levels[2]:
            return f"{coef}$^{{***}}$"
        elif pvalue <= significance_levels[1]:
            return f"{coef}$^{{**}}$"
        elif pvalue <= significance_levels[0]:
            return f"{coef}$^{{*}}$"
        else:
            return coef

    # Apply stars to coefficients based on p-values
    for i, model in enumerate(models):
        coef_col = results_df.columns[i]
        pvalues = model.pvalues
        for idx in results_df.index:
            if idx in pvalues.index:
                coef = results_df.loc[idx, coef_col]
                pvalue = pvalues[idx]
                results_df.loc[idx, coef_col] = add_stars(coef, pvalue)

    # Function to format interaction term
    def format_interaction(var_name):
        if ':' in var_name:
            var1, var2 = var_name.split(':')
            var1 = variable_display_names.get(var1, var1)
            var2 = variable_display_names.get(var2, var2)
            return f'{var1} × {var2}'
        return variable_display_names.get(var_name, var_name)

    new_index = [format_interaction(idx) for idx in results_df.index]
    results_df.index = new_index

    bold_model_names = [f"\\textsc{{{name}}}" for name in model_names]

    # Start creating LaTeX longtable
    latex_code = [
        "\\begin{longtable}{l" + "c" * len(model_names) + "}",

        "\\hline",
        " & " + " & ".join(bold_model_names) + " \\\\",
        "\\hline",
        "\\endfirsthead",
        "",
        "\\multicolumn{" + str(
            len(model_names) + 1) + "}{c}{\\tablename\\ \\thetable{} -- continued from previous page} \\\\",
        "\\hline",
        " & " + " & ".join(model_names) + " \\\\",
        "\\hline",
        "\\endhead",
        "",
        "\\hline \\multicolumn{" + str(len(model_names) + 1) + "}{r}{Continued on next page} \\\\",
        "\\endfoot",
        "",
        "\\hline",
        "\\caption{Regression results for " + dependent_var + ". Parentheses represent standard errors. Significance levels:  $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$} \\\\",
        "\\endlastfoot",

        ""
    ]

    # Add data rows
    for idx, row in results_df.iterrows():
        latex_code.append(idx + " & " + " & ".join(row.astype(str)) + " \\\\")

    # End the table
    latex_code.append("\\end{longtable}")

    # Join all lines
    latex_table = "\n".join(latex_code)

    # Save to file
    with open(file_name, "w") as f:
        f.write(latex_table)

    print(f"LaTeX longtable saved to {file_name}")

    return latex_table
