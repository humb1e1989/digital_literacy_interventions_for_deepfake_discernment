# Designing Effective Digital Literacy Interventions for Boosting Deepfake Discernment

This is the code and data for the paper Designing Effective Digital Literacy Interventions for Boosting Deepfake Discernment.

The code has been tested on Python 3.8

## Abstract
Deepfakes images can erode trust in institutions and compromise election outcomes, as people often struggle to discern real images from deepfake images. Improving digital literacy can help address these challenges. Here, we compare the efficacy of five digital literacy interventions to boost people's ability to discern deepfakes: (1) textual guidance on common indicators of deepfakes; (2) visual demonstrations of these indicators; (3) a gamified exercise for identifying deepfakes; (4) implicit learning through repeated exposure and feedback; and (5) explanations of how deepfakes are generated with the help of AI. We conducted an experiment with $N=1,200$ participants from the United States to test the immediate and long-term effectiveness of our interventions. Our results show that our lightweight, easy-to-understand interventions can boost deepfake image discernment by up to 13 percentage points while maintaining trust in real images. 

## Repository Structure
Code: Folder to store the code used for the analysis.\
Data: Folder to store the data.\
Survey: Folder to store the survey qsf.

## Data
We anonymize participants profilic IDs in the data.
The data files are in CSV format.
The data is split into group 1 and group 2 based on the order they saw the image sets.
Data is also split by main study and follow-up study.

## Code usage

Run the [main.py](Code/main.py) file. It will 
- read and process the data from the data folder. 
- compute the discernment ability for each participant.
- check normality assumptions of variables 
- compute t-tests and equivalence tests to answer our hypotheses

Run [regressions.py](Code/regressions.py) file for robustness regression checks.

Run [visualizations.py](Code/visualizations.py) file to generate the figures used in the paper.