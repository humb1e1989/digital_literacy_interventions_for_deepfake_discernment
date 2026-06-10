import config as conf # Import your configuration file depending on data
import data_preparation
import compute_metrics
import statistical_tests
import pandas as pd

import regression_prep
from logger import get_logger


logger = get_logger(__name__)

# run get_prolic_ids.py first!

def main():
    # Load and clean the data
    df1, mapping1 = data_preparation.prep_data(conf.file_path_group1, group='group1')
    df2, mapping2 = data_preparation.prep_data(conf.file_path_group2, group='group2')
    logger.debug(f"Filtered data lengths: Group1 main:{len(df1['id'].tolist())}, Group2 main: {len(df2['id'].tolist())}")

    # load and clean follow up data
    df_follow_up1 = data_preparation.prep_follow_up(conf.file_path_followup1, group='group1_followup')
    df_follow_up2 = data_preparation.prep_follow_up(conf.file_path_followup2, group='group2_followup')
    logger.debug(f"Filtered data lengths: Group1 followup: {len(df_follow_up1['id'].tolist())}, Group2 followup: {len(df_follow_up2['id'].tolist())}")

    # add intervention group based on main data id
    df_follow_up1['intervention_group'] = df_follow_up1['id'].map(mapping1)
    df_follow_up1['attrition'] = 0
    df_follow_up2['intervention_group'] = df_follow_up2['id'].map(mapping2)
    df_follow_up2['attrition'] = 0
    logger.info(f"Remaining participants per intervention group after follow up: {df_follow_up1['intervention_group'].value_counts()}, {df_follow_up2['intervention_group'].value_counts()}")

    # Create attrition column = 1 if ID is not in df follow up, 0 if it is
    df1['attrition'] = (~df1['id'].isin(df_follow_up1['id'])).astype(int)
    df2['attrition'] = (~df2['id'].isin(df_follow_up2['id'])).astype(int)

    # Save cleaned DataFrame to a new zip file
    df1.to_csv(conf.clean_file_path_group1, index=False, compression='zip')
    logger.info(f"Cleaned data saved to {conf.clean_file_path_group1}")
    df2.to_csv(conf.clean_file_path_group2, index=False, compression='zip')
    logger.info(f"Cleaned data saved to {conf.clean_file_path_group2}")

    # Save cleaned follow up DataFrame to a new zip file
    df_follow_up1.to_csv(conf.clean_file_path_followup1, index=False, compression='zip')
    logger.info(f"Cleaned data saved to {conf.clean_file_path_followup1}")
    df_follow_up2.to_csv(conf.clean_file_path_followup2, index=False, compression='zip')
    logger.info(f"Cleaned data saved to {conf.clean_file_path_followup2}")

    # merge data frames for group 1 and group 2
    df = pd.concat([df1, df2]).reset_index()
    df.to_csv(conf.clean_file_path, index=False, compression='zip')
    df_follow_up = pd.concat([df_follow_up1, df_follow_up2]).reset_index()
    df_follow_up.to_csv(conf.clean_file_path_followup, index=False, compression='zip')

    # Compute metrics of how well participants performed in the discernment task
    logger.info(f"Computing metrics for group 1+2")
    df, results_participants, results_images = compute_metrics.get_metrics(df, conf.image_ratings_columns, conf.metrics_file, conf.image_metrics_file)

    logger.info(f"Computing metrics for follow up groups")
    df_follow_up, results_participants_follow_up, results_images_follow_up = compute_metrics.get_metrics(df_follow_up, conf.image_ratings_columns, conf.metrics_file_followup, conf.image_metrics_file_followup)

    # Merge the detection results for time main and follow up
    df_time = pd.merge(
        results_participants,
        results_participants_follow_up,
        on=['id', 'intervention_group', 'attrition'],  # merge on both ID and group
        suffixes=('_t1', '_t2')
    )
    df_time.to_csv('../Data/processed/pilot2/metrics_results_over_time.csv', index=False)

    # Merge the dataframes on id
    df_merged = pd.merge(df, results_participants, on=['id', 'intervention_group', 'attrition'])

    # recode variables
    logger.info("Recode variables")
    df_recoded = regression_prep.recode_variables(df)

    # # perform statistical tests on the results
    logger.info(f"Normality all:")
    statistical_tests.check_normality(results_participants, 'real_accuracy')
    statistical_tests.check_normality(results_participants, 'fake_accuracy')
    statistical_tests.check_normality(results_participants, 'real_sharing_rate')
    statistical_tests.check_normality(results_participants, 'fake_sharing_rate')
    logger.info(f"Normality follow up:")
    statistical_tests.check_normality(results_participants_follow_up, 'real_accuracy')
    statistical_tests.check_normality(results_participants_follow_up, 'fake_accuracy')
    statistical_tests.check_normality(results_participants_follow_up, 'real_sharing_rate')
    statistical_tests.check_normality(results_participants_follow_up, 'fake_sharing_rate')

    logger.info(f"T-tests group1+group2:")
    ttests_participants = statistical_tests.mann_whitney_u_tests(results_participants, conf.groups)
    ttests_sharing_intention = statistical_tests.mann_whitney_u_tests(results_participants, conf.groups, variable="fake_sharing_rate")


    logger.info(f"T-tests follow up data group1+group2:")
    ttests_participants = statistical_tests.mann_whitney_u_tests(results_participants_follow_up, conf.groups)
    ttests_sharing_intention = statistical_tests.mann_whitney_u_tests(results_participants_follow_up, conf.groups,
                                                                      variable="fake_sharing_rate")

    logger.info(f"Equivalence test group1+group2:")
    logger.info("Main data:")
    # equivalence test for control and each intervention group for real accuracy at main time point
    equ = statistical_tests.equivalence_test(results_participants, conf.groups)
    logger.info(f"Follow up data:")
    equ_follow_up = statistical_tests.equivalence_test(results_participants_follow_up, conf.groups)
    logger.info(f"Equivalence test for sharing intention group1+group2:")
    equ_sharing = statistical_tests.equivalence_test(results_participants, conf.groups, variable='real_sharing_rate')
    equ_sharing_follow_up = statistical_tests.equivalence_test(results_participants_follow_up, conf.groups, variable='real_sharing_rate')

    print("Done")





if __name__ == '__main__':
    main()

