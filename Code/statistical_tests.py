from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from scipy import stats
from logger import get_logger
import pandas as pd
import numpy as np
from itertools import combinations
import config

logger = get_logger(__name__)
# print all pandas columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def describe_dependent_vars(df_participants, df_images):
    # average accuracy of participants per intervention group
    logger.info(f"Describe results for participants:")
    logger.info(f"Real accuracy\n{df_participants.groupby('intervention_group')['real_accuracy'].describe()}")
    logger.info(f"Fake accuracy\n{df_participants.groupby('intervention_group')['fake_accuracy'].describe()}")
    logger.info(f"Sharing rates\n{df_participants.groupby('intervention_group')['real_sharing_rate'].describe()}")
    logger.info(f"Sharing rates\n{df_participants.groupby('intervention_group')['fake_sharing_rate'].describe()}")
    # average accuracy of participants per image category
    logger.info(f"Describe results for images:")
    logger.info(f"\n{df_images.groupby('category')['accuracy'].describe()}")
    logger.info(f"\n{df_images.groupby('groundtruth')['accuracy'].describe()}")

def check_attrition_differences(df_detection, df_attributes, results_file):
    # check if there are differences in user attributes between participants who did the follow-up and those who did not
    def describe_demographics(df):
        # describe the data
        logger.info(f"Descriptives demographics main:")
        logger.info(f"Counts gender binary (0=male; 1=female):\n{df['gender_binary'].value_counts()}")
        logger.info(f"Counts age:\n{df['age'].describe()}; Age groups:\n{df['age_group'].value_counts()}")
        logger.info(f"Counts ethnicity groups:\n{df['ethnicity_group'].value_counts()}")
        logger.info(f"Counts education groups:\n{df['education_group'].value_counts()}")
        logger.info(f"Counts religion groups:\n{df['religion_recoded'].value_counts()}")
        logger.info(f"Counts political orientation groups:\n{df['political_orientation'].value_counts()}")
        logger.info(f"Counts income groups:\n{df['income_group'].value_counts()}")
        logger.info(f"Counts confidence:\n{df['slider_confidence_1'].value_counts()}")
        logger.info(df[config.demographic_regression].describe())

    # Get groups
    attrition_group = df_attributes[df_attributes['attrition'] == 1]
    follow_up_group = df_attributes[df_attributes['attrition'] == 0]

    # Run original descriptives
    logger.info(f"Descriptives for all participants:")
    describe_demographics(df_attributes)
    logger.info(f"Descriptives for attrition group:")
    describe_demographics(attrition_group)
    logger.info(f"Descriptives for follow-up group:")
    describe_demographics(follow_up_group)

    # Add statistical tests
    demographics = ['gender_binary', 'age', 'age_group', 'ethnicity_group',
                    'education_group', 'religion_recoded', 'political_orientation',
                    'income_group', 'slider_confidence_1']

    # Clear/create the file at the start
    with open(results_file, 'w') as f:
        f.write("Attrition Analysis Results\n\n")

    def write_results(message):
        # Write to both logger and file
        logger.info(message)
        with open(results_file, 'a') as f:
            f.write(message + '\n')

    # perform Mann-Whitney U test for each attribute
    write_results("\nMann-Whitney U Test Results:")
    write_results("-" * 50)
    for var in demographics:
        att = attrition_group[var].dropna()
        fol = follow_up_group[var].dropna()
        stat, p = mannwhitneyu(
            att,
            fol,
            alternative='two-sided'
        )
        significance = "*" if p < 0.05 else ""
        write_results(f"{var}: att mean={att.mean()}, fol mean={fol.mean()}, U={stat:.2f}, p={p:.4f}{significance}")

    # check for detection difference between attrition groups
    write_results("\nDetection Differences:")
    write_results("-" * 50)
    for var in ['real_accuracy', 'fake_accuracy']:
        fol = df_detection[df_detection['attrition'] == 0][var]
        att = df_detection[df_detection['attrition'] == 1][var]
        U, p = mannwhitneyu(att,
                            fol,
                            method="asymptotic")
        significance = "*" if p < 0.05 else ""
        write_results(f"{var}: att mean={att.mean()}, fol mean={fol.mean()}, U={stat:.2f}, p={p:.4f}{significance}")



def check_normality(df, variable):
    # Step 1: Group the data by intervention group
    grouped_data = df.groupby('intervention_group')[variable]

    # Step 2: Apply Shapiro-Wilk normality test to each group
    normality_results = {}
    for group, data in grouped_data:
        stat, p_value = shapiro(data.dropna())  # Drop NaNs for the test
        normality_results[group] = {"Statistic": stat, "p-value": p_value}

    # Step 3: Print the results
    for group, result in normality_results.items():
        logger.info(f"Group: {group} | Statistic: {result['Statistic']:.4f}, p-value: {result['p-value']:.4f}")

        # You can also interpret the results:
        if result['p-value'] < 0.05:
            logger.info(f"The {variable} data for {group} is NOT normally distributed (reject H0).")
        else:
            logger.info(f"The {variable} data for {group} is normally distributed (fail to reject H0).")


def get_significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def create_mann_whitney_matrix(df, accuracy_types, groups):
    results = {}
    for acc in accuracy_types:
        matrix = np.zeros((len(groups), len(groups)))
        p_values = np.zeros((len(groups), len(groups)))
        significance = np.empty((len(groups), len(groups)), dtype=object)

        for (i, group1), (j, group2) in combinations(enumerate(groups), 2):
            U, p = mannwhitneyu(df[df['intervention_group'] == group1][acc],
                                df[df['intervention_group'] == group2][acc],
                                method="asymptotic")
            matrix[i, j] = matrix[j, i] = U
            p_values[i, j] = p_values[j, i] = p
            mean1 = df[df['intervention_group'] == group1][acc].mean()
            mean2 = df[df['intervention_group'] == group2][acc].mean()
            stars = get_significance_stars(p)
            significance[i, j] = significance[j, i] = stars

            logger.info(f"{acc} - {group1} vs {group2}: Mean1 = {mean1}, Mean2 = {mean2}  U = {U}; p = {p} {stars}")

        results[acc] = {
            'U': pd.DataFrame(matrix, index=groups, columns=groups),
            'p': pd.DataFrame(p_values, index=groups, columns=groups),
            'significance': pd.DataFrame(significance, index=groups, columns=groups)
        }

    return results


def mann_whitney_u_tests(df, groups, variable='fake_accuracy'):
    results = {}
    for group in groups[1:]:
        U, p = mannwhitneyu(df[df['intervention_group'] == "control"][variable],
                            df[df['intervention_group'] == group][variable],
                            method="asymptotic")
        # bonferroni correction
        p_adj = min(p*5, 1.000)
        mean1 = df[df['intervention_group'] == "control"][variable].mean()
        mean2 = df[df['intervention_group'] == group][variable].mean()
        stars = get_significance_stars(p_adj)
        # significance[i, j] = significance[j, i] = stars

        logger.info(f"(Bonf) {variable} - control vs {group}: Mean1 = {mean1}, Mean2 = {mean2}  U = {U}; p = {p_adj} {stars}")
        results[group] = [U, p_adj]
    return results


def print_significant_combinations(results, accuracy_types, groups):
    for acc in accuracy_types:
        logger.info(f"\nSignificant combinations for {acc}:")
        for (group1, group2) in combinations(groups, 2):
            stars = results[acc]['significance'].loc[group1, group2]
            if stars:
                p_value = results[acc]['p'].loc[group1, group2]
                logger.info(f"{group1} vs {group2}: {stars} (p = {p_value:.4f})")

def mannwhitneyu_test_images(df):
    real_group = df[df['category'] == 'real']["accuracy"]
    fake_group = df[df['category'] == 'fake']["accuracy"]
    viral_group = df[df['category'] == 'viral']["accuracy"]

    U_real_fake, p = mannwhitneyu(real_group, fake_group, method="asymptotic")
    logger.info(f"U_real_fake: {U_real_fake}; p= {p}")
    U_real_viral, p = mannwhitneyu(real_group, viral_group, method="asymptotic")
    logger.info(f"U_real_viral: {U_real_viral}; p= {p}")
    U_fake_viral, p = mannwhitneyu(fake_group, viral_group, method="asymptotic")
    logger.info(f"U_fake_viral: {U_fake_viral}; p= {p}")

    not_real_group = df[df['category'] != 'real']["accuracy"]
    U_not_real_real, p = mannwhitneyu(not_real_group, real_group, method="asymptotic")
    logger.info(f"U_not_real_real: {U_not_real_real}; p= {p}")

def wilcoxon_test(df1, df2):
    df1_selected = df1[['id', 'intervention_group',
                        'real_accuracy', 'fake_accuracy']]
    df2_selected = df2[['id', 'intervention_group',
                        'real_accuracy', 'fake_accuracy']]

    # Merge the dataframes
    merged_df = pd.merge(
        df1_selected,
        df2_selected,
        on=['id', 'intervention_group'],  # merge on both ID and group
        suffixes=('_t1', '_t2')
    )


    for group in config.groups:
        group_data = merged_df[merged_df['intervention_group'] == group]

        logger.info(f"\nWilcoxon Results for group: {group}")
        for acc_type in config.accuracy_types:
            t1_col = f"{acc_type}_t1"
            t2_col = f"{acc_type}_t2"
            w_stat = wilcoxon(group_data[t1_col], group_data[t2_col])
            statistic = w_stat.statistic
            p_value = w_stat.pvalue
            stars = get_significance_stars(p_value)
            logger.info(f"{acc_type}: {w_stat} {stars}")

            # Optional: Add descriptive statistics
            mean_t1 = group_data[t1_col].mean()
            mean_t2 = group_data[t2_col].mean()
            logger.info(f"Means - Time 1: {mean_t1:.2f}, Time 2: {mean_t2:.2f}")
            logger.info(f"Standard Deviations - Time 1: {group_data[t1_col].std():.2f}, Time 2: {group_data[t2_col].std():.2f}")
            logger.info(f"Standard Errors - Time 1: {group_data[t1_col].sem():.2f}, Time 2: {group_data[t2_col].sem():.2f}")

def equivalence_test(df, groups, alpha=0.05, variable='real_accuracy'):
    # test for equivalence between control group and each intervention group for real accuracy at time point t1 using TOST method
    results = {}
    low_eqbound = -0.05  # Lower equivalence bound
    high_eqbound = 0.05  # Upper equivalence bound
    m = 5
    alpha_prime = alpha / m if m > 0 else alpha  # per-comparison alpha
    ci_level = 1 - 2 * alpha_prime

    for group in groups[1:]:
        control_data = df[df['intervention_group'] == 'control'][variable]
        intervention_data = df[df['intervention_group'] == group][variable]

        # Perform TOST
        diff = np.mean(control_data) - np.mean(intervention_data)

        # Standard error of difference
        se_diff = np.sqrt(np.var(control_data, ddof=1) / len(control_data) +
                          np.var(intervention_data, ddof=1) / len(intervention_data))

        # Two one-sided t-tests
        t1 = (diff - low_eqbound) / se_diff  # Test: diff > low_bound
        t2 = (high_eqbound - diff) / se_diff  # Test: diff < high_bound

        df_len = len(control_data) + len(intervention_data) - 2

        # p-values for one-sided tests
        p1 = 1 - stats.t.cdf(t1, df_len)  # p-value for diff > low_bound
        p2 = 1 - stats.t.cdf(t2, df_len)  # p-value for diff < high_bound

        # Overall p-value is the maximum of the two
        p_value = max(p1, p2)
        # Bonferroni-adjust across the m interventions
        p_value_adj = min(p_value * m, 1.0)

        # Confidence interval for the difference
        # t_critical = stats.t.ppf(1 - alpha / 2, df_len)
        # ci_lower = diff - t_critical * se_diff
        # ci_upper = diff + t_critical * se_diff

        # Simultaneous two-sided CI at level (1 - 2*alpha')
        t_critical = stats.t.ppf(1 - alpha_prime, df_len)
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff

        # Equivalent if both tests are significant
        # is_equivalent = p_value < 0.05
        is_equivalent = (p1 < alpha_prime) and (p2 < alpha_prime)
        if is_equivalent:
            status = "equivalent"
        elif (ci_lower > high_eqbound) or (ci_upper < low_eqbound):
            status = "significantly different"
        else:
            status = "undecided"

        results[group] = {
            'tost_p_value_raw': float(p_value),
            'tost_p_value_adj': float(p_value_adj),
            'ci_level_percent': ci_level * 100,
            'mean_difference': diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': (1 - alpha) * 100,
            'status': status,
            'equivalence_bounds': (low_eqbound, high_eqbound)
        }
        logger.info(f"Equivalence test (Bonf) for {group}: '{status} (p = {p_value_adj:.4f}, mean difference = {diff:.4f}, CI[{ci_level*100:.1f}%] = [{ci_lower:.4f}, {ci_upper:.4f}])")
    return results

