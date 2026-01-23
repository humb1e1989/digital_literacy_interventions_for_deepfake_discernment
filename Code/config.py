learning_effect = True
file_path_group1 = '../Data/group1_data_raw.zip'
file_path_group2 = '../Data/group2_data_raw.zip'  # Update to your actual CSV file path

clean_file_path_group1 = '../Data/group1_data_cleaned.zip'
clean_file_path_group2 = '../Data/group2_data_cleaned.zip'  # Update to your actual CSV file path
clean_file_path = '../Data/all_data_cleaned.zip'  # Update to your actual CSV file path

file_path_followup1 = '../Data/group1_followup_data_raw.zip'  # Update to your actual CSV file path
file_path_followup2 = '../Data/group2_followup_data_raw.zip'  # Update to your actual CSV file path

clean_file_path_followup1 = '../Data/group1_followup_data_cleaned.zip'  # Update to your actual CSV file path
clean_file_path_followup2 = '../Data/group2_followup_data_cleaned.zip'  # Update to your actual CSV file path
clean_file_path_followup = '../Data/all_followup_data_cleaned.zip'  # Update to your actual CSV file path

# column categories
meta_info_columns = ['meta_info_Browser', 'meta_info_Version', 'meta_info_Operating System', 'meta_info_Resolution', 'mobile']
info_columns = ['Prolific ID', 'prolific_id_timer_Page Submit', 'PROLIFIC_PID']
time_columns = ['Duration (in seconds)']
consent_columns = ['consent_timer_Page Submit', 'welcome_instr_timer_Page Submit']
attention_honesty_checks_columns = ['Attention check 1', 'attention_1_timer_Page Submit', 'attention_1_failed',
                            'Attention check 2', 'attention_2_timer_Page Submit', 'attention_2_failed',
                            'Honesty check 1', 'Honesty check 2', 'honesty_timer_Page Submit', 'honesty_1_failed', 'honesty_2_failed']
intervention_columns = ['intervention_group']
intervention_1_columns = ['explanation_timer_Page Submit']
intervention_2_columns = ['textual_tip_intro_page_submit', 'textual_tip_1_page_submit', 'textual_tip_2_page_submit',
                          'textual_tip_3_page_submit', 'textual_tip_4_page_submit', 'textual_tip_5_page_submit']
intervention_3_columns = ['visual_tip_intro_page_submit', 'visual_tip_1_page_submit', 'visual_tip_2_page_submit',
                          'visual_tip_3_page_submit', 'visual_tip_4_page_submit', 'visual_tip_5_page_submit']
intervention_4_columns = ['game_tip_intro_gameTime', 'game_tip_intro_page_submit', 'game_tip_1_gameTime', 'game_tip_1_page_submit',
                          'game_tip_2_gameTime', 'game_tip_2_page_submit', 'game_tip_3_gameTime', 'game_tip_3_page_submit',
                          'game_tip_4_gameTime', 'game_tip_4_page_submit', 'game_tip_5_gameTime', 'game_tip_5_page_submit',
                          'feedbackMessage','totalScore'] # 'game_tip_1_score', 'game_tip_2_score', 'game_tip_3_score', 'game_tip_4_score', 'game_tip_5_score'
intervention_5_columns = ['feedback_intro_timer_Page Submit', 'feedback_real1', 'feedback_real2', 'feedback_real3', 'feedback_real4', 'feedback_real5',
                          'feedback_real1_timer_Page Submit', 'feedback_real2_timer_Page Submit', 'feedback_real3_timer_Page Submit', 'feedback_real4_timer_Page Submit', 'feedback_real5_timer_Page Submit',
                          'feedback_fake1', 'feedback_fake2', 'feedback_fake3', 'feedback_fake4', 'feedback_fake5',
                          'feedback_fake1_timer_Page Submit', 'feedback_fake2_timer_Page Submit', 'feedback_fake3_timer_Page Submit', 'feedback_fake4_timer_Page Submit', 'feedback_fake5_timer_Page Submit']
detection_columns = ['detection_intro_time_Page Submit']
image_ratings_columns = ['3_task_real_or_fake', '4_task_real_or_fake', '5_task_real_or_fake', '6_task_real_or_fake',
                         '7_task_real_or_fake', '8_task_real_or_fake', '9_task_real_or_fake', '10_task_real_or_fake',
                         '11_task_real_or_fake', '12_task_real_or_fake', '13_task_real_or_fake', '14_task_real_or_fake',
                         '15_task_real_or_fake', '16_task_real_or_fake', '17_task_real_or_fake']
image_sharing_columns = ['3_task_sharing', '4_task_sharing', '5_task_sharing', '6_task_sharing',
                         '7_task_sharing', '8_task_sharing', '9_task_sharing', '10_task_sharing',
                         '11_task_sharing', '12_task_sharing', '13_task_sharing', '14_task_sharing',
                         '15_task_sharing', '16_task_sharing', '17_task_sharing']
image_validation_columns = ['3_task_seen_before', '4_task_seen_before', '5_task_seen_before', '6_task_seen_before',
                            '7_task_seen_before', '8_task_seen_before', '9_task_seen_before', '10_task_seen_before',
                            '11_task_seen_before', '12_task_seen_before', '13_task_seen_before', '14_task_seen_before',
                            '15_task_seen_before', '16_task_seen_before', '17_task_seen_before']
image_timers_columns = ['3_detection_timer_Page Submit', '4_detection_timer_Page Submit', '5_detection_timer_Page Submit', '6_detection_timer_Page Submit',
                        '7_detection_timer_Page Submit', '8_detection_timer_Page Submit', '9_detection_timer_Page Submit', '10_detection_timer_Page Submit',
                        '11_detection_timer_Page Submit', '12_detection_timer_Page Submit', '13_detection_timer_Page Submit', '14_detection_timer_Page Submit',
                        '15_detection_timer_Page Submit', '16_detection_timer_Page Submit', '17_detection_timer_Page Submit']
confidence_columns = ['slider_confidence_1', 'confidence_timer_Page Submit']
social_media_columns = ['platform_presence', 'platform_presence_7_TEXT', 'content_sharing', 'content_sharing_6_TEXT', 'time_social_media', 'social_media_timer_Page Submit']
media_literacy_columns = ['Experience deepfakes_1', 'Experience_detecting_1', 'Search_engine_1', 'reverse_image_search_1', 'genAI_1', 'media_literacy_timer_Page Submit']
CRT_columns = ['CRT_1', 'CRT_2', 'CRT_3', 'CRT_timer_Page Submit', 'CRT_1_correct', 'CRT_2_correct', 'CRT_3_correct']
demographics_columns = ['demo_intro_timer_Page Submit', 'gender', 'gender_4_TEXT', 'gender_timer_Page Submit',
                        'age', 'age_timer_Page Submit', 'ethnicity', 'ethnicity_timer_Page Submit',
                        'education', 'education_timer_Page Submit', 'religion', 'religion_timer_Page Submit',
                        'social_issues', 'economic_issues', 'political_timer_Page Submit',
                        'Income_US', 'income_timer_Page Submit', 'social_status_Q', 'social_status_timer_Page Submit']

emotion_task_columns = ['1_emotion_task', '1_emotion_sharing', '1_task_timer_Page Submit',
                        '2_emotion_task', '2_emotion_sharing', '2_task_timer_Page Submit',
                        '3_emotion_task', '3_emotion_sharing', '3_task_timer_Page Submit',
                        '4_emotion_task', '4_emotion_sharing', '4_task_timer_Page Submit',
                        '5_emotion_task', '5_emotion_sharing', '5_task_timer_Page Submit',
                        '6_emotion_task', '6_emotion_sharing', '6_task_timer_Page Submit',
                        '7_emotion_task', '7_emotion_sharing', '7_task_timer_Page Submit',
                        '8_emotion_task', '8_emotion_sharing', '8_task_timer_Page Submit',
                        '9_emotion_task', '9_emotion_sharing', '9_task_timer_Page Submit',
                        '10_emotion_task', '10_emotion_sharing', '10_task_timer_Page Submit',
                        '11_emotion_task', '11_emotion_sharing', '11_task_timer_Page Submit',
                        '12_emotion_task', '12_emotion_sharing', '12_task_timer_Page Submit',
                        '13_emotion_task', '13_emotion_sharing', '13_task_timer_Page Submit',
                        '14_emotion_task', '14_emotion_sharing', '14_task_timer_Page Submit',
                        '15_emotion_task', '15_emotion_sharing', '15_task_timer_Page Submit'
                        ]

image_groundtruth = []
image_groundtruth.extend(['real'] * 5) # images 3-7
image_groundtruth.extend(['fake'] * 10) # images 8-17, the viral images are also fake

metrics_file_group1 = '../Data/group1_participant_metrics.zip'
metrics_file_group2 = '../Data/group2_participant_metrics.zip'
metrics_file = '../Data/all_participant_metrics.zip'

image_metrics_file_group1 = '../Data/group1_image_metrics.zip'
image_metrics_file_group2 = '../Data/group2_image_metrics.zip'
image_metrics_file = '../Data/all_image_metrics.zip'

metrics_file_group1_followup = '../Data/group1_participant_metrics_followup.zip'
metrics_file_group2_followup = '../Data/group2_participant_metrics_followup.zip'
metrics_file_followup = '../Data/all_participant_metrics_followup.zip'

image_metrics_file_group1_followup = '../Data/group1_image_metrics_followup.zip'
image_metrics_file_group2_followup = '../Data/group2_image_metrics_followup.zip'
image_metrics_file_followup = '../Data/all_image_metrics_followup.zip'

accuracy_types = ['real_accuracy', 'fake_accuracy']
sharing_types = ['real_sharing_rate', 'fake_sharing_rate']
groups = ['control', 'intervention_2', 'intervention_3', 'intervention_4', 'intervention_5', 'intervention_1'] # control, knowledge, textual, visual, game

# regression variables
demographic_regression = ['gender_binary', 'age', 'ethnicity_group', 'education_group', 'religion_recoded',
                    'political_orientation','income_group'] # social_status_q
social_media_regression = ['platform_count', 'sharing_count', 'time_social_media']
media_literacy_regression = ['experience_deepfakes', 'experience_detecting', 'search_engine', 'reverse_image_search', 'genai']
crt_regression= ['crt']
confidence_regression=['slider_confidence_1']

variable_display_names = {'C(intervention_group)[T.intervention_2]': 'Textual',
                          'C(intervention_group)[T.intervention_3]': 'Visual',
                          'C(intervention_group)[T.intervention_4]': 'Gamified',
                          'C(intervention_group)[T.intervention_5]': 'Feedback',
                          'C(intervention_group)[T.intervention_1]': 'Knowledge',
                          'C(intervention_group)[T.1]': 'Intervention',
                          'gender_binary': 'Gender', 'age': 'Age', 'ethnicity_group': 'Ethnicity',
                            'education_group': 'Education', 'religion_recoded': 'Religion',
                            'political_orientation': 'PoliticalOrientation', 'income_group': 'Income',
                            'social_status_q': 'SocialStatus', 'platform_count': 'PlatformCount',
                            'sharing_count': 'SharingCount', 'time_social_media': 'TimeOnline',
                            'experience_deepfakes': 'KnowledgeDeepfakes', 'experience_detecting': 'ExpDetecting',
                            'search_engine': 'ExpSearchEngine', 'reverse_image_search': 'ExpImageSearch',
                            'genai': 'ExpGenAI', 'crt': 'CRT', 'slider_confidence_1': 'Confidence'}

follow_up_rating_columns = ['1_real_or_fake', '2_real_or_fake', '3_Q95', '4_Q95', '5_Q101', '6_Q101', '7_Q107', '8_Q107',
                            '9_Q113', '10_Q113', '11_Q119', '12_Q119', '13_Q125', '14_Q125', '15_Q131', '16_Q131',
                            '17_Q137', '18_Q137', '19_Q143', '20_Q143', '21_Q149', '22_Q149', '23_Q155', '24_Q155',
                            '25_Q161', '26_Q161', '27_Q167', '28_Q167', '29_Q173', '30_Q173']
follow_up_sharing_columns = ['1_sharing_intention', '2_sharing_intention', '3_Q96', '4_Q96', '5_Q102', '6_Q102', '7_Q108', '8_Q108',
                             '9_Q114', '10_Q114', '11_Q120', '12_Q120', '13_Q126', '14_Q126', '15_Q132', '16_Q132',
                            '17_Q138', '18_Q138', '19_Q144', '20_Q144', '21_Q150', '22_Q150', '23_Q156', '24_Q156',
                            '25_Q162', '26_Q162', '27_Q168', '28_Q168', '29_Q174', '30_Q174']
follow_up_validation_columns = ['1_seen image before', '2_seen image before', '3_Q97', '4_Q97', '5_Q103', '6_Q103', '7_Q109', '8_Q109',
                                '9_Q115', '10_Q115', '11_Q121', '12_Q121', '13_Q127', '14_Q127', '15_Q133', '16_Q133',
                                '17_Q139', '18_Q139', '19_Q145', '20_Q145', '21_Q151', '22_Q151', '23_Q157', '24_Q157',
                                '25_Q163', '26_Q163', '27_Q169', '28_Q169', '29_Q175', '30_Q175']
follow_up_timers_columns = ['1_detection_timer_Page Submit', '2_detection_timer_Page Submit', '3_Q99_Page Submit', '4_Q99_Page Submit',
                            '5_Q105_Page Submit', '6_Q105_Page Submit', '7_Q111_Page Submit', '8_Q111_Page Submit',
                            '9_Q117_Page Submit', '10_Q117_Page Submit', '11_Q123_Page Submit', '12_Q123_Page Submit',
                            '13_Q129_Page Submit', '14_Q129_Page Submit', '15_Q135_Page Submit', '16_Q135_Page Submit',
                            '17_Q141_Page Submit', '18_Q141_Page Submit', '19_Q147_Page Submit', '20_Q147_Page Submit',
                            '21_Q153_Page Submit', '22_Q153_Page Submit', '23_Q159_Page Submit', '24_Q159_Page Submit',
                            '25_Q165_Page Submit', '26_Q165_Page Submit', '27_Q171_Page Submit', '28_Q171_Page Submit',
                            '29_Q177_Page Submit', '30_Q177_Page Submit']

