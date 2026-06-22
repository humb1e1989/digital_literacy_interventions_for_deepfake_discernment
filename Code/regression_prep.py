import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from logger import get_logger

logger = get_logger(__name__)

def recode_demographics(df):
    # Recode variables with correct data types
    df['gender_binary'] = df['gender'].map({'Male': 0, 'Female': 1}).astype('float64')

    def recode_education(edu):
        if edu == '7-12 (up to High School)':
            return 0
        elif edu == '13-16 (college/undergraduate university/certificate training)':
            return 1
        elif edu == 'More than 17 years (doctorate degree, medical degree, etc.)':
            return 2
        else:
            return np.nan

    df['education_group'] = df['education'].apply(recode_education).astype('float64')

    def recode_income(income):
        if income == 'Prefer not to say':
            return np.nan
        elif 'Less than' in income or any(f'${i:,}' in income for i in range(10000, 30000, 10000)):
            return 0  # Low income
        elif any(f'${i:,}' in income for i in range(30000, 80000, 10000)):
            return 1  # Middle income
        else:
            return 2  # High income

    df['income_group'] = df['income_us'].apply(recode_income).astype('float64')

    def recode_political_orientation(political_orientation):
        if political_orientation <= 1.5:
            return 0
        elif political_orientation == 2:
            return 1
        elif political_orientation >= 2.5:
            return 2
        else:
            return np.nan

    df['political_orientation'] = ((df['social_issues'] + df['economic_issues']) / 2).astype(
        'float64')


    # Recode Age
    df['age_group'] = pd.cut(df['age'], bins=[0, 34, 54, np.inf], labels=[0, 1, 2]).astype('float64')
    df['age'] = df['age'].astype('float64')

    # df['ethnicity_group'] = df['ethnicity'].apply(lambda x: 0 if x == 'White' else 1).astype('float64')
    def encode_ethnicity_numeric(ethnicity):
        if ethnicity == 'White':
            return 0
        elif ethnicity == 'Black or African American':
            return 1
        elif ethnicity == 'Hispanic, Latino or Spanish origin':
            return 2
        elif ethnicity == 'Asian':
            return 3
        else:
            return 4  # 'Other' category

    df['ethnicity_group'] = df['ethnicity'].apply(encode_ethnicity_numeric).astype('float64')

    def recode_religion(religion):
        if religion == 'A religious person':
            return 0
        elif religion == 'Not a religious person':
            return 1
        else:
            return np.nan  # This will create a NaN value for "Prefer not to answer"

    df['religion_recoded'] = df['religion'].apply(recode_religion).astype('float64')

    # Convert categorical variables
    df['intervention_group'] = pd.Categorical(df['intervention_group'])

    # Ensure social_status_q and overall_accuracy are float64
    df['social_status_q'] = df['social_status_q'].astype('float64')
    return df

def recode_social_media(df):
    # Existing encodings
    # df['uses_social_media'] = np.where(df['platform_presence'].notnull(), 1, 0)
    # df['shares_content'] = np.where(df['content_sharing'].notnull(), 1, 0)

    def count_platforms(platforms_string):
        if pd.isnull(platforms_string):
            return 0
        return len(str(platforms_string).split(','))

    df['platform_count'] = df['platform_presence'].apply(count_platforms).astype('float64')
    df['sharing_count'] = df['content_sharing'].apply(count_platforms).astype('float64')
    df['time_social_media'] = df['time_social_media'].astype('float64')
    return df

def recode_media_literacy(df):
    df['experience_deepfakes'] = df['experience_deepfakes_1'].astype('float64')
    df['experience_detecting'] = df['experience_detecting_1'].astype('float64')
    df['search_engine'] = df['search_engine_1'].astype('float64')
    df['reverse_image_search'] = df['reverse_image_search_1'].astype('float64')
    df['genai'] = df['genai_1'].astype('float64')
    return df

def recode_crt(df):
    # count how many correct
    df['crt'] = df[['crt_1_correct', 'crt_2_correct', 'crt_3_correct']].sum(axis=1).astype('float64')
    return df

def recode_variables(df):
    df = recode_demographics(df)
    df = recode_social_media(df)
    df = recode_media_literacy(df)
    df = recode_crt(df)
    df['slider_confidence_1'] = df['slider_confidence_1'].astype('float64')
    # logger.info("Checking variable types:")
    # check_variable_types(df_merged, demographic_vars + ['intervention_group', 'overall_accuracy'])
    check_variable_types(df,['social_status_q'])
    # Check for multicollinearity
    # check_multicollinearity(df_merged, demographic_vars + ['intervention_group'])
    return df

# Check data types and unique values
def check_variable_types(df, variables):
    for var in variables:
        logger.info(f"\nVariable: {var}")
        logger.info(f"Data type: {df[var].dtype}")
        logger.info(f"Unique values: {df[var].unique()}")
        logger.info(f"Number of unique values: {df[var].nunique()}")
        logger.info(f"Number of missing values: {df[var].isnull().sum()}")

def check_multicollinearity(df, variables):
    # Create a dataframe with only the variables of interest
    df_subset = df[variables].copy()

    # Convert categorical variables to dummy variables
    categorical_columns = df_subset.select_dtypes(include=['category', 'object']).columns
    df_dummy = pd.get_dummies(df_subset, columns=categorical_columns, drop_first=True)

    # Convert boolean columns to integer
    bool_columns = df_dummy.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_dummy[col] = df_dummy[col].astype(int)

    # Ensure all columns are numeric
    for col in df_dummy.columns:
        df_dummy[col] = pd.to_numeric(df_dummy[col], errors='coerce')

    # Drop rows with NaN values
    df_dummy = df_dummy.dropna()

    # Add a constant term to the dataframe (required for statsmodels VIF calculation)
    df_dummy = sm.add_constant(df_dummy)

    # Calculate VIF for each variable
    vif = pd.DataFrame()
    vif["Variable"] = df_dummy.columns

    try:
        vif["VIF"] = [variance_inflation_factor(df_dummy.values, i) for i in range(df_dummy.shape[1])]
        logger.info("Variance Inflation Factors:")
        logger.info(vif)
    except Exception as e:
        logger.info(f"Error calculating VIF: {e}")
        logger.info("Dataframe info:")
        logger.info(df_dummy.info())
        logger.info("\nDataframe head:")
        logger.info(df_dummy.head())
        return

    # Calculate correlation matrix
    correlation_matrix = df_dummy.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Independent Variables')
    plt.show()

