from scipy.stats import chi2_contingency
import config
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

# compute the correctness of the image ratings
# mapping ratings to predictions
def map_ratings_to_predictions(rating):
    if rating in ['Probably fake', 'Definitely fake']:
        return 'fake'
    elif rating in ['Probably real', 'Definitely real']:
        return 'real'
    else:
        return None  # In case of missing or invalid data


# Function to calculate metrics
def calculate_metrics_per_participants(row, rating_columns, groundtruth):
    true_positive = false_positive = true_negative = false_negative = 0
    correct = 0
    total_fake = total_real = correct_fake = correct_real = 0
    valid_predictions = 0  # To track the number of predictions made (should be 15 per participant)

    # Iterate through each rating column
    for i, col in enumerate(rating_columns):
        prediction = row[col]

        # Skip if the prediction is None/NaN
        if pd.isna(prediction):
            continue

        # Get the corresponding groundtruth
        groundtruth_value = groundtruth[i]
        valid_predictions += 1

        if prediction == groundtruth_value:
            correct += 1
            if groundtruth_value == 'fake':
                correct_fake += 1
                true_negative += 1
            elif groundtruth_value == 'real':
                correct_real += 1
                true_positive += 1
        else:
            if prediction == 'fake' and groundtruth_value == 'real':
                false_negative += 1
            elif prediction == 'real' and groundtruth_value == 'fake':
                false_positive += 1

        # Count total fake/real instances in the ground truth
        if groundtruth_value == 'fake':
            total_fake += 1
        elif groundtruth_value == 'real':
            total_real += 1

    # Accuracy
    overall_accuracy = correct / valid_predictions if valid_predictions > 0 else 0

    # Fake class accuracy (true negatives / total fake instances)
    fake_accuracy = correct_fake / total_fake if total_fake > 0 else 0

    # Real class accuracy (true positives / total real instances)
    real_accuracy = correct_real / total_real if total_real > 0 else 0

    # True/False positive/negative rates
    true_positive_rate = true_positive / total_real if total_real > 0 else 0  # Sensitivity
    false_positive_rate = false_positive / total_fake if total_fake > 0 else 0
    true_negative_rate = true_negative / total_fake if total_fake > 0 else 0  # Specificity
    false_negative_rate = false_negative / total_real if total_real > 0 else 0

    # Balanced Accuracy
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    return {
        'overall_accuracy': overall_accuracy * 100,
        'balanced_accuracy': balanced_accuracy * 100,
        'fake_accuracy': fake_accuracy * 100,
        'real_accuracy': real_accuracy * 100,
        'true_positive_rate': true_positive_rate * 100,
        'false_positive_rate': false_positive_rate * 100,
        'true_negative_rate': true_negative_rate * 100,
        'false_negative_rate': false_negative_rate * 100
    }



def calculate_metrics_per_image(df, ground_truth):
    # per column calculate accuracy for the image
    accuracy_per_image = {}

    # Compute accuracy for each image (i.e., each column)
    for i, col in enumerate(df.columns):  # Iterate through columns (images)
        image_ratings = df[col].dropna()  # Drop missing values (participants who didn't rate this image)

        # Compare the ratings to the ground truth for this image
        correct_ratings = image_ratings == ground_truth[i]

        # Calculate the accuracy as the proportion of correct ratings
        accuracy = correct_ratings.mean()  # Mean of True/False values gives the accuracy
        accuracy_per_image[col] = accuracy*100

    # Step 4: Convert the results to a DataFrame or print the results
    accuracy_df = pd.DataFrame(list(accuracy_per_image.items()), columns=['image', 'accuracy'])
    accuracy_df['groundtruth'] = ground_truth
    category = []
    category.extend(['real'] * 5)
    category.extend(['fake'] * 5)
    category.extend(['viral'] * 5)
    accuracy_df['category'] = category
    return accuracy_df

def calculate_sharing_rates_per_participant(row, sharing_columns, groundtruth):
    """
    Calculate sharing rates for real and fake images for each participant.
    """
    real_shared = 0
    fake_shared = 0
    total_real = 0
    total_fake = 0

    # Iterate through each sharing column
    for i, col in enumerate(sharing_columns):
        sharing_intention = row[col]

        # Skip if the response is missing
        if pd.isna(sharing_intention):
            continue

        groundtruth_value = groundtruth[i]

        # Count based on ground truth
        if groundtruth_value == 'real':
            total_real += 1
            if sharing_intention == 'Yes':
                real_shared += 1
        elif groundtruth_value == 'fake':
            total_fake += 1
            if sharing_intention == 'Yes':
                fake_shared += 1

    # Calculate sharing rates
    real_sharing_rate = (real_shared / total_real * 100) if total_real > 0 else 0
    fake_sharing_rate = (fake_shared / total_fake * 100) if total_fake > 0 else 0

    return {
        'real_sharing_rate': real_sharing_rate,
        'fake_sharing_rate': fake_sharing_rate
    }

def calculate_sharing_rates_by_belief(row, sharing_columns, judgment_columns, groundtruth):
    """
    Compute sharing rates conditional on participant judgment and on correctness.
    Assumes:
      - sharing_columns: per-item 'Yes'/'No' (or boolean) share intentions
      - judgment_columns: per-item judgments ('real'/'fake' or compatible)
      - groundtruth: list/array of per-item ground truth labels ('real'/'fake')
    Returns percentages (0–100). Missing entries are skipped per item.
    """

    # counters
    judged_real_total = judged_fake_total = 0
    judged_real_share = judged_fake_share = 0

    # positive = real, negative = fake
    TP = TN = FP = FN = 0
    TP_share = TN_share = FP_share = FN_share = 0

    for i, share_col in enumerate(sharing_columns):
        share = row.get(share_col, np.nan)
        if pd.isna(share):
            continue

        share_yes = (share == 'Yes') or (share is True) or (share == 1)

        gt = groundtruth[i]  # 'real' or 'fake'
        j = row.get(judgment_columns[i])  # 'real' or 'fake'
        if pd.isna(j):
            continue
        j = str(j).strip().lower()
        gt = str(gt).strip().lower()

        # by belief only
        if j == 'real':
            judged_real_total += 1
            if share_yes:
                judged_real_share += 1
        elif j == 'fake':
            judged_fake_total += 1
            if share_yes:
                judged_fake_share += 1

        # by correctness (positive=real, negative=fake)
        if gt == 'real' and j == 'real':  # TP
            TP += 1
            if share_yes: TP_share += 1  # ★ was TN_share
        elif gt == 'fake' and j == 'real':  # FP
            FP += 1
            if share_yes: FP_share += 1  # ★ was FN_share
        elif gt == 'fake' and j == 'fake':  # TN
            TN += 1
            if share_yes: TN_share += 1  # ★ was TP_share
        elif gt == 'real' and j == 'fake':  # FN
            FN += 1
            if share_yes: FN_share += 1  # ★ was FP_share

    def rate(num, den):  # percent
        return (num / den * 100) if den > 0 else 0.0

    return {
        # sharing conditional on participant belief
        'share_given_judged_real': rate(judged_real_share, judged_real_total),
        'share_given_judged_fake': rate(judged_fake_share, judged_fake_total),

        # sharing conditional on correctness cells
        'share_on_TN': rate(TN_share, TN),  # deepfake judged fake
        'share_on_FP': rate(FP_share, FP),  # deepfake judged real (risky)
        'share_on_FN': rate(FN_share, FN),  # authentic judged fake
        'share_on_TP': rate(TP_share, TP),  # authentic judged real
    }

def get_metrics(df, rating_columns, output_file_participants, output_file_images):
    # map ratings to predictions ("definitely fake" and "probably fake" to "fake", "definitely real" and "probably real" to "real")
    # Create new columns with a suffix (e.g., "_mapped") for each rating column
    image_ratings_columns = [c.lower() for c in rating_columns]
    for col in image_ratings_columns:
        new_col_name = col + "_mapped"  # New column name
        df[new_col_name] = df[col].apply(map_ratings_to_predictions)
    prediction_columns = [col + "_mapped" for col in image_ratings_columns]

    # Compute metrics for each participant
    metrics_participants = df.apply(
        lambda row: calculate_metrics_per_participants(row, prediction_columns, config.image_groundtruth),
        axis=1
    )
    metrics_participants_df = pd.DataFrame(metrics_participants.tolist())

    sharing_columns_lower = [c.lower() for c in config.image_sharing_columns]
    sharing_rates = df.apply(
        lambda row: calculate_sharing_rates_per_participant(
            row, sharing_columns_lower, config.image_groundtruth
        ),
        axis=1
    )
    sharing_rates_df = pd.DataFrame(sharing_rates.tolist())

    # Combine accuracy and sharing metrics
    for col in sharing_rates_df.columns:
        metrics_participants_df[col] = sharing_rates_df[col]

    belief_rates = df.apply(
    lambda r: calculate_sharing_rates_by_belief(r, sharing_columns_lower, prediction_columns, config.image_groundtruth),
    axis=1, result_type='expand'
    )

    for col in belief_rates.columns:
        metrics_participants_df[col] = belief_rates[col]

    # add the intervention group
    metrics_participants_df['intervention_group'] = df['intervention_group']
    metrics_participants_df['id'] = df['id']
    metrics_participants_df['attrition'] = df['attrition']
    metrics_participants_df.to_csv(output_file_participants, index=False, compression='zip')
    logger.info(f"Metrics participants saved to {output_file_participants}")

    # calculate accuracy per image
    metrics_images_df = calculate_metrics_per_image(df[prediction_columns], config.image_groundtruth)
    metrics_images_df.to_csv(output_file_images, index=False, compression='zip')
    logger.info(f"Metrics images saved to {output_file_images}")

    return df, metrics_participants_df, metrics_images_df

def compute_sharing(df, filename):
    # compute the sharing of the images
    # compute the sharing of the images
    sharing_columns = [c.lower() for c in config.image_sharing_columns]

    # count the number of participants that shared an image
    sharing_counts = df[sharing_columns].apply(pd.Series.value_counts, axis=0)

    # Separate real and fake images
    real_image_counts = sharing_counts.iloc[:, :5].sum(axis=1)
    fake_image_counts = sharing_counts.iloc[:, 5:].sum(axis=1)

    # Combine into a single DataFrame
    contingency_table = pd.DataFrame({
        'Real': real_image_counts,
        'Fake': fake_image_counts
    })

    logger.info(f"Contingency Table:\n{contingency_table}")
    contingency_table.to_csv(filename)

    # Calculate percentages (excluding NaN)
    def calculate_share_percentage(column):
        total = column.sum() - column.get('NaN', 0)
        return (column['Yes'] / total) * 100 if total > 0 else 0

    share_real = calculate_share_percentage(contingency_table['Real'])
    share_fake = calculate_share_percentage(contingency_table['Fake'])

    logger.info(f"Percentage willing to share real news: {share_real:.2f}%")
    logger.info(f"Percentage willing to share fake news: {share_fake:.2f}%")

    # Perform chi-square test (excluding NaN)
    chi2_table = contingency_table.loc[["Don't know", "No", "Yes"]]
    chi2, p_value, dof, expected = chi2_contingency(chi2_table)
    logger.info(f"Chi-square test statistic: {chi2:.4f}")
    logger.info(f"Chi-square test p-value: {p_value:.4f}")

    return df






