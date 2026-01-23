from regression_prep import *
from regression_helper import *
import pickle
import config_val as conf

logger = get_logger(__name__)
# print all pandas columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# load results and participant data
results_participants = pd.read_csv(conf.metrics_file, compression='zip', header=0)
results_participants = results_participants.drop(columns=['intervention_group', 'attrition'], errors='ignore')
df = pd.read_csv(conf.clean_file_path, compression='zip', header=0)

results_participants_followup = pd.read_csv(conf.metrics_file_followup, compression='zip', header=0)
results_participants_followup = results_participants_followup.drop(columns=['intervention_group', 'attrition'], errors='ignore')


# Merge the dataframes on prolific_id
df_merged = pd.merge(df, results_participants, on='id')
describe_regression_data(df_merged)
df_merged_followup = pd.merge(df, results_participants_followup, on='id')
describe_regression_data(df_merged_followup)

# recode variables
df_merged = recode_variables(df_merged)
df_merged_followup = recode_variables(df_merged_followup)
# df_merged['overall_accuracy'] = df_merged['overall_accuracy'].astype('float64')
# df_merged.to_csv('../Data/processed/cleaned_processed_qualtrics_main_500_US.csv', index=False)
# uncomment to describe the data
# describe_recoded_data(df_merged)
# List of your recoded variables
recoded_vars = ['gender_binary', 'education_group', 'income_group', 'political_orientation',
                'age', 'ethnicity_group', 'religion_recoded']

# Find rows where ANY of these variables is NaN
nan_mask = df_merged[recoded_vars].isna().any(axis=1)
rows_with_nan = df_merged[nan_mask]

print(f"Number of rows with NaN in recoded variables: {nan_mask.sum()}")
print(len(rows_with_nan))
print(df_merged[recoded_vars].isna().sum())

# regression for sharing intention
logger.info("Running regressions for real sharing intention")
formula_real_sharing = ('real_sharing_rate ~ C(intervention_group)')
real_sharing_model = run_regression(df_merged, formula_real_sharing)
cohens_d_real = compute_cohens_d(df_merged, 'real_sharing_rate', real_sharing_model)
logger.info(f"Finished regression for real sharing intention")

logger.info("Running regressions for fake sharing intention")
logger.info("Main data:")
formula_fake_sharing = ('fake_sharing_rate ~ C(intervention_group)')
fake_sharing_model = run_regression(df_merged, formula_fake_sharing)
cohens_d_fake = compute_cohens_d(df_merged, 'fake_sharing_rate', fake_sharing_model)
logger.info("Follow Up data:")
fake_sharing_model_followup = run_regression(df_merged_followup, formula_fake_sharing)
cohens_d_fake_followup = compute_cohens_d(df_merged_followup, 'fake_sharing_rate', fake_sharing_model_followup)
logger.info(f"Finished regression for fake sharing intention")

# Corrected formula for your regression
formula_real = ('real_accuracy ~ C(intervention_group)')
formula_fake = ('fake_accuracy ~ C(intervention_group)')
formula_all = ('fake_accuracy ~ C(intervention_group) + ' +
           ' + '.join(config.demographic_regression + config.social_media_regression + config.media_literacy_regression
                      + config.crt_regression + config.confidence_regression))
formula_demo = ('fake_accuracy ~ C(intervention_group) + ' + ' + '.join(config.demographic_regression))
formula_social = ('fake_accuracy ~ C(intervention_group) + ' + ' + '.join(config.social_media_regression))
formula_media = ('fake_accuracy ~ C(intervention_group) + ' + ' + '.join(config.media_literacy_regression))
formulas = {'demographics': formula_demo, 'social_media': formula_social, 'media_literacy': formula_media} # exclude overall for now

# Create new formulas with interactions
formulas_with_interactions = {
    name: add_interactions(formula)
    for name, formula in formulas.items()
}

# Run regressions for real accuracy
logger.info("Running regressions for real accuracy")
real_model = run_regression(df_merged, formula_real)
cohens_d_real = compute_cohens_d(df_merged, 'real_accuracy', real_model)
logger.info(f"Finished regression for real accuracy")

# Run regressions for fake accuracy
logger.info("Running regressions for fake accuracy")
logger.info("Main data:")
fake_model = run_regression(df_merged, formula_fake)
cohens_d_fake = compute_cohens_d(df_merged, 'fake_accuracy', fake_model)
logger.info("Follow Up data:")
fake_model_followup = run_regression(df_merged_followup, formula_fake)
cohens_d_fake_followup = compute_cohens_d(df_merged_followup, 'fake_accuracy', fake_model_followup)
logger.info(f"Finished regression for fake accuracy")


# overall models (Table S3, left column "Overall")
models = {}
for name, formula in formulas.items():
    logger.info(f"Running regression for {name}")
    print(formula)
    models[name] = run_regression(df_merged, formula, group=name, check_multicollinearity=True)
    logger.info(f"Finished regression for {name}")


# control model (Table S3, right column "Control")
control_data = df_merged[df_merged['intervention_group'] == 'control']
control_models = {}
for name, formula in formulas.items():
    logger.info(f"Running regression for {name} for the control group")
    group_formula = formula.replace('C(intervention_group) + ', '')
    print(group_formula)
    control_models[name] = run_regression(control_data, group_formula, group=name, check_multicollinearity=True)


# Regressions for each intervention group + control (Table S3, middle columns)
intervention_models = {}
interaction_models = {}
for group in ['intervention_1', 'intervention_5', 'intervention_2', 'intervention_3', 'intervention_4']:
    group_data = df_merged[df_merged['intervention_group'].isin(['control', group])].copy()
    # Recode to binary
    group_data['intervention_group'] = (group_data['intervention_group'] == group).astype(int)
    # group_data = group_data.rename(columns={'intervention_group': f'{group}'})
    group_models = {}
    for name, formula in formulas.items():
        group_name = f"{group}_control_{name}"
        print(formula)
        # Remove intervention_group from formula for group-specific regressions
        group_models[name] = run_regression(group_data, formula, group=group_name, check_multicollinearity=True)
    intervention_models[group] = group_models
    group_models = {}
    for name, formula in formulas_with_interactions.items():
        group_name = f"{group}_control_{name}_interactions"
        print(formula)
        group_models[name] = run_regression(group_data, formula, group=group_name)
    interaction_models[group] = group_models

# save models
all_models = {'overall': models, **intervention_models, 'control': control_models}
group_names = {'overall':'Overall', 'intervention_2':'Textual', 'intervention_3': 'Visual', 'intervention_4':'Gamified',
               'intervention_5':'Feedback', 'intervention_1':'Knowledge', 'control':'Control'}
group_names_interaction = {'intervention_2':'Textual', 'intervention_3': 'Visual', 'intervention_4':'Gamified', 'intervention_5':'Feedback', 'intervention_1':'Knowledge'}
for group, group_models in all_models.items():
    group_name = group_names[group]
    for model_type, model in group_models.items():
        with open(f'../Results/pilot2/{group_name}_{model_type}.pkl', 'wb') as file:
            pickle.dump(model, file)

for group, group_models in interaction_models.items():
    group_name = group_names_interaction[group]
    for model_type, model in group_models.items():
        with open(f'../Results/pilot2/{group_name}_{model_type}_interactions.pkl', 'wb') as file:
            pickle.dump(model, file)


#
# # Compare R-squared values
# logger.info("\nR-squared Comparisons:")
# r_squared = {name: model.rsquared for name, model in all_models.items()}
# logger.info(pd.Series(r_squared))

# Now create the regression tables
dependent_var = 'fake accuracy'

# Demographics table
file_name = '../../Results/pilot2/regression_table_demographics.txt'
demographic_models = [all_models['overall']['demographics'],
                      all_models['intervention_2']['demographics'], all_models['intervention_3']['demographics'],
                      all_models['intervention_4']['demographics'], all_models['intervention_5']['demographics'],
                      all_models['intervention_1']['demographics'],all_models['control']['demographics']]
latex_table_demographics = create_regression_table(demographic_models, list(group_names.values()), dependent_var,
                                                   config.variable_display_names, file_name=file_name)

file_name = '../../Results/pilot2/regression_table_demographics_interactions.txt'
demographic_interaction_models = [interaction_models['intervention_2']['demographics'],
                                  interaction_models['intervention_3']['demographics'],
                                  interaction_models['intervention_4']['demographics'],
                                  interaction_models['intervention_5']['demographics'],
                                  interaction_models['intervention_1']['demographics']]
latex_table_demographics_interactions = create_regression_table(demographic_interaction_models, list(group_names_interaction.values()), dependent_var,
                                                                config.variable_display_names, file_name=file_name)

# Social Media table
file_name = '../../Results/pilot2/regression_table_social_media.txt'
social_media_models = [all_models['overall']['social_media'],
                       all_models['intervention_2']['social_media'], all_models['intervention_3']['social_media'],
                       all_models['intervention_4']['social_media'], all_models['intervention_5']['social_media'],
                       all_models['intervention_1']['social_media'], all_models['control']['social_media']]
latex_table_social_media = create_regression_table(social_media_models, list(group_names.values()), dependent_var,
                                                   config.variable_display_names, file_name=file_name)

file_name = '../../Results/pilot2/regression_table_social_media_interactions.txt'
social_media_interaction_models = [interaction_models['intervention_2']['social_media'],
                                   interaction_models['intervention_3']['social_media'],
                                   interaction_models['intervention_4']['social_media'],
                                   interaction_models['intervention_5']['social_media'],
                                   interaction_models['intervention_1']['social_media']]
latex_table_social_media_interactions = create_regression_table(social_media_interaction_models, list(group_names_interaction.values()), dependent_var,
                                                                config.variable_display_names, file_name=file_name)

# Media Literacy table
file_name = '../../Results/pilot2/regression_table_media_literacy.txt'
media_literacy_models = [all_models['overall']['media_literacy'],
                       all_models['intervention_2']['media_literacy'], all_models['intervention_3']['media_literacy'],
                       all_models['intervention_4']['media_literacy'], all_models['intervention_5']['media_literacy'],
                       all_models['intervention_1']['media_literacy'], all_models['control']['media_literacy']]
latex_table_media_literacy = create_regression_table(media_literacy_models, list(group_names.values()), dependent_var,
                                                     config.variable_display_names, file_name=file_name)

file_name = '../../Results/pilot2/regression_table_media_literacy_interactions.txt'
media_literacy_interaction_models = [interaction_models['intervention_2']['media_literacy'],
                                   interaction_models['intervention_3']['media_literacy'],
                                   interaction_models['intervention_4']['media_literacy'],
                                   interaction_models['intervention_5']['media_literacy'],
                                   interaction_models['intervention_1']['media_literacy']]
latex_table_media_literacy_interactions = create_regression_table(media_literacy_interaction_models, list(group_names_interaction.values()), dependent_var,
                                                                  config.variable_display_names, file_name=file_name)




