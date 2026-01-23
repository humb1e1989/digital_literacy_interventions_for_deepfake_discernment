import pandas as pd
import numpy as np
from logger import get_logger
import config

logger = get_logger(__name__)

# Load the CSV file
def load_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, compression='zip')
    # Reset the index to ensure it's clean and sequential
    df = df.reset_index(drop=True)
    return df

def clean_columns(df):
    # remove time columns and unused columns
    click_columns = [col for col in df.columns if col.endswith(tuple(["Last Click", "First Click", "Click Count"]))]
    columns_to_drop = ['StartDate', 'EndDate', 'Progress',
           'Finished', 'RecordedDate', 'ExternalReference',
            'UserLanguage', 'Consent', 'Explanation']
    columns_to_drop.extend(click_columns)
    columns_to_drop.extend(config.emotion_task_columns)

    df = df.drop(columns=columns_to_drop, errors='ignore')
    # lower case column names
    df.columns = df.columns.str.lower()
    # rename column
    df = df.rename(columns={'experience deepfakes_1': 'experience_deepfakes_1'})
    df = df.rename(columns={'exaplanation_timer_page submit': 'explanation_timer_page submit'})

    return df.reset_index(drop=True)

def clean_timer_columns(df):
    # clean timer columns to the same format and decimal places (in seconds, 1 decimal place)
    df = df.rename(columns={'duration (in seconds)': 'survey_duration'})
    df['survey_duration'] = pd.to_numeric(df['survey_duration']).astype(float)

    page_submit_columns = [col for col in df.columns if col.endswith('page submit')]
    if page_submit_columns:
        df[page_submit_columns] = df[page_submit_columns].astype(float).round(1)

    intervention_page_submit_columns = [col for col in df.columns if col.endswith('page_submit')]
    if intervention_page_submit_columns:
        df[intervention_page_submit_columns] = df[intervention_page_submit_columns].astype(float)
        # convert from milliseconds to seconds
        df[intervention_page_submit_columns] = df[intervention_page_submit_columns].apply(lambda x: x/1000)
        df[intervention_page_submit_columns] = df[intervention_page_submit_columns].astype(float).round(1)

    intervention_game_time_columns = [col for col in df.columns if col.endswith('gametime')]
    if intervention_game_time_columns:
        df[intervention_game_time_columns] = df[intervention_game_time_columns].astype(float)
        # convert from milliseconds to seconds
        df[intervention_game_time_columns] = df[intervention_game_time_columns].apply(lambda x: x / 1000)
        df[intervention_game_time_columns] = df[intervention_game_time_columns].astype(float).round(1)

    df = df.rename(columns=lambda col: col.replace('page submit', 'page_submit') if col.endswith('page submit') else col)
    return df.reset_index(drop=True)

def clean_column_data_type(df):
    # Map Likert scale values for political ideology
    ideology_mapping = {
        'Very liberal': 1,
        'Liberal': 2,
        'Somewhat liberal': 3,
        'Neutral': 4,
        'Somewhat conservative': 5,
        'Conservative': 6,
        'Very conservative': 7
    }

    # Apply mapping to ideology columns if they exist and contain string values
    for col in ['social_issues', 'economic_issues']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map(ideology_mapping)

    numeric_columns = ['slider_confidence_1', 'time_social_media', 'experience_deepfakes_1', 'experience_detecting_1',
                       'search_engine_1', 'reverse_image_search_1', 'genai_1', 'crt_1', 'crt_2', 'crt_3',
                       'age', 'social_issues', 'economic_issues', 'social_status_q', 'mobile', 'totalscore',
                       'game_tip_1_score', 'game_tip_2_score', 'game_tip_3_score', 'game_tip_4_score', 'game_tip_5_score',
                       'crt_1_correct', 'crt_2_correct', 'crt_3_correct', 'attention_1_failed', 'attention_2_failed',
                       'honesty_1_failed', 'honesty_2_failed']
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
    # convert to numeric int
    df[existing_numeric_columns] = df[existing_numeric_columns].apply(pd.to_numeric, errors='coerce')
    # filter users with age < 18
    if 'age' in df.columns:
        # Set age to NaN if less than 18
        df['age'] = df['age'].apply(lambda x: x if x >= 18.0 else np.nan)
    return df.reset_index(drop=True)

def filter_time_outliers(df):
    # count the number of participants per intervention group
    logger.debug(f"Number of participants per intervention group: {df['intervention_group'].value_counts()}")

    # disregard every row where the survey duration is outside the 95% interval for the respective intervention group
    filtered = df[df['survey_duration'] < df.groupby('intervention_group')['survey_duration'].transform(
        lambda x: x.quantile(0.95))]
    # log how many were filtered from each group
    logger.debug(
        f"Number of participants per intervention group after filtering time outliers: {filtered['intervention_group'].value_counts()}")
    # log difference between before and after filtering
    logger.debug(
        f"Number of participants filtered out per intervention group time outliers: {df['intervention_group'].value_counts() - filtered['intervention_group'].value_counts()}")

    return filtered.reset_index(drop=True)

def filter_attention_checks(df, group):
    print("before filtering", df.shape[0])
    # check if somebody failed both attention checks
    logger.info(f"Number of participants that failed first attention check: {df['attention_1_failed'].sum()}")
    logger.info(f"Number of participants that failed second attention check: {df['attention_2_failed'].sum()}")
    failed_both = df[(df['attention_1_failed'] == 1) & (df['attention_2_failed'] == 1)]
    logger.info(f"Number of participants that failed both attention checks: {failed_both.shape[0]}")
    if failed_both.shape[0] > 0:
        failed_both.to_csv(f"../../Data/processed/pilot2/{group}_filtered_attention_checks.csv", index=False)
        df = df.drop(failed_both.index).reset_index(drop=True)
        logger.info(f"Dropped participants that failed both attention")
    elif df['attention_1_failed'].sum() > 0:
        failed_first = df[df['attention_1_failed'] == 1]
        failed_first.to_csv(f"../../Data/processed/pilot2/{group}_filtered_attention_checks.csv", index=False)
        df = df.drop(df[df['attention_1_failed'] == 1].index).reset_index(drop=True)
        logger.info(f"Dropped participants that failed first attention check")

    # check if somebody failed both honesty checks
    logger.info(f"Number of participants that failed first honesty check: {df['honesty_1_failed'].sum()}")
    logger.info(f"Number of participants that failed second honesty check: {df['honesty_2_failed'].sum()}")
    failed_both = df[(df['honesty_1_failed'] == 1) & (df['honesty_2_failed'] == 1)]
    logger.info(f"Number of participants that failed both honesty checks: {failed_both.shape[0]}")
    if df['honesty_1_failed'].sum() > 0 or df['honesty_2_failed'].sum() > 0:
        failed_either = df[((df['honesty_1_failed'] == 1) | (df['honesty_2_failed'] == 1))]
    if failed_both.shape[0] > 0:
        failed_both.to_csv(f"../../Data/processed/pilot2/{group}_filtered_honesty_checks.csv", index=False)
        df = df.drop(failed_both.index).reset_index(drop=True) # change to failed_either if filtering needs to change
        logger.info(f"Dropped participants that failed either honesty check")
    logger.debug(f"Number of participants after filtering attention and honesty checks: {df.shape[0]}")
    return df.reset_index(drop=True)

def filter_same_answers(df, columns):
    # check if somebody answered the detection the same for all images
    subset_df = df[columns]
    # Check if all non-NaN values in each row are the same across the subset columns
    same_values_rows = subset_df.apply(lambda row: row.nunique(dropna=True) == 1, axis=1)
    # Get rows where the condition holds true
    matching_rows = df[same_values_rows]
    # print ids of participants that answered the same for all images
    logger.debug(f"Participants that answered the same for all images: {matching_rows['id'].tolist()}")
    # Drop the rows where the condition holds true
    df = df.drop(matching_rows.index).reset_index(drop=True)
    logger.debug(f"Dropped participants that answered the same for all images: {matching_rows.shape[0]}")
    return df.reset_index(drop=True)

def set_nan_based_on_validation(df, image_ratings_columns, image_sharing_columns, image_validation_columns, min_real_nans=3, min_fake_nans=6):
    # Create a dictionary to map validation columns to their corresponding rating and sharing columns
    column_mapping = {}
    for val_col in image_validation_columns:
        prefix = val_col.split('_')[0]  # Get the number prefix
        rating_col = next((col for col in image_ratings_columns if col.startswith(prefix)), None)
        sharing_col = next((col for col in image_sharing_columns if col.startswith(prefix)), None)
        if rating_col and sharing_col:
            column_mapping[val_col] = (rating_col, sharing_col)

    # Function to check if a value contains 'Yes' (case-insensitive)
    def contains_yes(value):
        return 'yes' in str(value).lower()

    # Initialize dictionaries to track NaNs by ID and by image type
    id_nan_counts = {}
    real_image_nan_counts = {}
    fake_image_nan_counts = {}

    image_validation_counts = {}

    # Identify which rating columns are for real images (first 5) vs fake images (last 10)
    real_image_cols = image_ratings_columns[:5]
    fake_image_cols = image_ratings_columns[5:15]

    # Iterate through the mapping and set NaN where validation is 'Yes'
    for val_col, (rating_col, sharing_col) in column_mapping.items():
        mask = df[val_col].apply(contains_yes)

        validation_count = mask.sum()
        image_validation_counts[rating_col] = validation_count

        # Determine if this is a real or fake image rating
        is_real_image = rating_col in real_image_cols

        # For each affected row, increment the appropriate counters
        for id_value in df.loc[mask, 'id']:
            id_nan_counts[id_value] = id_nan_counts.get(id_value, 0) + 1
            # Track by image type
            if is_real_image:
                real_image_nan_counts[id_value] = real_image_nan_counts.get(id_value, 0) + 1
            else:
                fake_image_nan_counts[id_value] = fake_image_nan_counts.get(id_value, 0) + 1
        # Set NaN values
        df.loc[mask, rating_col] = np.nan
        df.loc[mask, sharing_col] = np.nan

    logger.debug(f"Number of real image nan counts: {sum(real_image_nan_counts.values())}")
    logger.debug(f"Number of fake image nan counts: {sum(fake_image_nan_counts.values())}")

    logger.debug("\nValidation counts per image:")
    for col, count in sorted(image_validation_counts.items()):
        logger.debug(f"  {col}: {count} participants validated")

    # Filter out rows with too many NaNs
    ids_to_remove = [
        id_value for id_value in id_nan_counts
        if real_image_nan_counts.get(id_value, 0) >= min_real_nans
           and fake_image_nan_counts.get(id_value, 0) >= min_fake_nans
    ]
    # Remove filtered IDs
    if ids_to_remove:
        df = df[~df['id'].isin(ids_to_remove)].copy()
        logger.debug(f"Dropped participants with insufficient ratings: {len(ids_to_remove)}")
    logger.debug(f"Remaining participants: {df.shape[0]}")
    return df.reset_index(drop=True)


def prep_data(file_path, group):
    # Load the data
    df = load_data(file_path)
    df = df.drop_duplicates(subset=['id'], keep='first')
    logger.debug(f"Loaded {file_path} data with {df.shape[0]} rows and {df.shape[1]} columns")
    mapping = dict(zip(df['id'], df['intervention_group']))
    # Clean the columns
    df = clean_columns(df)
    logger.debug(f"Cleaned main data with {df.shape[0]} rows and {df.shape[1]} columns")

    # Clean the timer columns
    df = clean_timer_columns(df)
    logger.info(f"Cleaned timer columns")

    # Clean the data types
    df = clean_column_data_type(df)

    # Filter outliers
    df = filter_time_outliers(df)
    logger.info(f"Filtered outliers")

    # Filter attention checks
    df = filter_attention_checks(df, group)

    validation_columns = [c.lower() for c in config.image_validation_columns]
    image_ratings_columns = [c.lower() for c in config.image_ratings_columns]
    image_sharing_columns = [c.lower() for c in config.image_sharing_columns]
    # Filter same answers
    if 'emotion' not in df['intervention_group'].unique():
        df = filter_same_answers(df, image_ratings_columns)
        df.to_csv(f"../../Data/processed/pilot2/{group}_before_validation_filter.csv", index=False)
        # Set NaN based on validation
        df = set_nan_based_on_validation(df, image_ratings_columns, image_sharing_columns, validation_columns)

    # log remaining participants per intervention group
    logger.info(f"Remaining participants per intervention group: {df['intervention_group'].value_counts()}")

    return df, mapping

def prep_follow_up(file_path, group):
    df = load_data(file_path)
    logger.info(f"Loaded follow-up data with {df.shape[0]} rows and {df.shape[1]} columns")
    df = clean_columns(df)
    df = clean_timer_columns(df)
    # Create a new column to indicate if attention check 1 was failed
    df['attention_1_failed'] = df['attention check 1'].apply(
        lambda x: 0 if x == "Recycling,Composting" else 1
    )

    # Create a new column to indicate if attention check 1 was failed
    df['attention_2_failed'] = df['attention check 2'].apply(
        lambda x: 0 if x == "I prefer audiobooks" else 1
    )
    df = clean_column_data_type(df)
    df = filter_attention_checks(df, group)
    validation_columns = [c.lower() for c in config.image_validation_columns]
    image_ratings_columns = [c.lower() for c in config.image_ratings_columns]
    image_sharing_columns = [c.lower() for c in config.image_sharing_columns]
    # Filter same answers
    # df = filter_same_answers(df, image_ratings_columns)
    # df = set_nan_based_on_validation(df, image_ratings_columns, image_sharing_columns, validation_columns)
    if group == 'group1_followup' or group == 'group2_followup':
        df = filter_same_answers(df, image_ratings_columns)
        df.to_csv(f"../../Data/processed/pilot2/{group}_before_validation_filter.csv", index=False)
        # Set NaN based on validation
        df = set_nan_based_on_validation(df, image_ratings_columns, image_sharing_columns, validation_columns)
    else:
        df = filter_same_answers(df, image_ratings_columns)
        df = set_nan_based_on_validation(df, image_ratings_columns, image_sharing_columns, validation_columns)
    return df