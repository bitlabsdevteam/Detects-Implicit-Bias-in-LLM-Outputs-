#!/usr/bin/env python3
"""
BBQ Results Metadata Generation Script

This script creates metadata useful for analyzing BBQ results
It outputs a csv file that contains information that can be matched
to a results file using the example id and the category information

Python conversion of BBQ_results_metadata.R
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set working directory to script location
script_dir = Path(__file__).parent
os.chdir(script_dir)

def read_jsonl(file_path):
    """Read JSONL file and return DataFrame"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

# Define unknown answers
unknowns = [
    "unknown", "cannot be determined", "can't be determined", "not answerable",
    "not known", "not enough info", "not enough information", "cannot answer",
    "can't answer", "undetermined"
]

# Get data files
data_dir = Path("../data/")
filenames = list(data_dir.glob("*.jsonl"))

print("Reading JSONL data files...")
dat = []
for f in filenames:
    print(f"Processing {f.name}...")
    temp = read_jsonl(f)
    
    # Extract answer info - handle list format
    def extract_answer_info(answer_info, ans_key):
        ans_data = answer_info[ans_key]
        if isinstance(ans_data, list) and len(ans_data) >= 2:
            return [ans_data[0], ans_data[1]]
        else:
            return [None, None]
    
    # Apply extraction and create DataFrames
    ans0_data = temp['answer_info'].apply(lambda x: extract_answer_info(x, 'ans0'))
    ans1_data = temp['answer_info'].apply(lambda x: extract_answer_info(x, 'ans1'))
    ans2_data = temp['answer_info'].apply(lambda x: extract_answer_info(x, 'ans2'))
    
    # Convert to DataFrame
    ans0_info = pd.DataFrame(ans0_data.tolist(), columns=['ans0_text', 'ans0_info'], index=temp.index)
    ans1_info = pd.DataFrame(ans1_data.tolist(), columns=['ans1_text', 'ans1_info'], index=temp.index)
    ans2_info = pd.DataFrame(ans2_data.tolist(), columns=['ans2_text', 'ans2_info'], index=temp.index)
    
    # Extract stereotyped groups
    stereotyped_groups = temp['additional_metadata'].apply(
        lambda x: str(x.get('stereotyped_groups', '')) if isinstance(x, dict) else str(x)
    ).to_frame('stereotyped_groups')
    
    # Remove nested columns and combine
    temp = temp.drop(['answer_info', 'additional_metadata'], axis=1)
    temp = pd.concat([temp, ans0_info, ans1_info, ans2_info, stereotyped_groups], axis=1)
    
    dat.append(temp)

if dat:
    dat = pd.concat(dat, ignore_index=True)
else:
    print("No data found!")
    exit(1)

print("Reading template files...")
# Read template files for stereotyped group information
templates_dir = Path("../templates/")
template_files = list(templates_dir.glob("*.csv"))

st_group_data = []
for template_file in template_files:
    if ('vocab' not in template_file.name.lower() and 
        '_x_' not in template_file.name and 
        'filler' not in template_file.name.lower()):
        
        temp = pd.read_csv(template_file)
        if 'Category' in temp.columns and 'Known_stereotyped_groups' in temp.columns:
            temp2 = temp[['Category', 'Known_stereotyped_groups', 'Q_id', 'Relevant_social_values']].copy()
            temp2 = temp2.rename(columns={
                'Category': 'category',
                'Q_id': 'question_index'
            })
            temp2['question_index'] = temp2['question_index'].astype(str)
            st_group_data.append(temp2)

if st_group_data:
    st_group_data = pd.concat(st_group_data, ignore_index=True)
    
    # Standardize category names
    category_mapping = {
        'GenderIdentity': 'Gender_identity',
        'PhysicalAppearance': 'Physical_appearance',
        'RaceEthnicity': 'Race_ethnicity',
        'Religion ': 'Religion',
        'SexualOrientation': 'Sexual_orientation',
        'DisabilityStatus': 'Disability_status'
    }
    
    st_group_data['category'] = st_group_data['category'].map(category_mapping).fillna(st_group_data['category'])
    
    # Group and deduplicate
    st_group_data = st_group_data.groupby([
        'category', 'question_index', 'Known_stereotyped_groups', 'Relevant_social_values'
    ]).size().reset_index(name='count').drop('count', axis=1)

# Merge with main data
dat4 = pd.merge(dat, st_group_data, on=['category', 'question_index'], how='left')

print("Processing non-intersectional categories...")
# Process non-intersectional templates
dat_base = dat4[~dat4['category'].str.contains('_x_', na=False)].copy()

if not dat_base.empty:
    # Clean up data
    dat_base = dat_base.drop('stereotyped_groups', axis=1, errors='ignore')
    
    # Clean stereotyped groups
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.replace(r'["\[\]]', '', regex=True)
    
    # Clean answer text
    for col in ['ans0_text', 'ans1_text', 'ans2_text']:
        dat_base[col] = dat_base[col].str.replace(r'[{}]', '', regex=True)
    
    # Add proper name indicator
    dat_base['ProperName'] = dat_base['question_index'].astype(int).apply(
        lambda x: 'name' if x > 25 else 'label'
    )
    
    # Standardize gender info
    gender_mapping = {'man': 'M', 'boy': 'M', 'woman': 'F', 'girl': 'F'}
    for col in ['ans0_info', 'ans1_info', 'ans2_info']:
        dat_base[col] = dat_base[col].map(gender_mapping).fillna(dat_base[col])
    
    # Clean stereotyped groups gender info
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].apply(
        lambda x: 'M' if str(x).lower() in ['man', 'boy', 'men'] else
                  'F' if str(x).lower() in ['woman', 'women', 'girl', 'girls'] else
                  'trans' if 'transgender' in str(x).lower() else x
    )
    
    # Remove gender prefixes
    for col in ['ans0_info', 'ans1_info', 'ans2_info']:
        dat_base[col] = dat_base[col].str.replace(r'^[MF]-', '', regex=True)
    
    # Handle specific categories
    dat_base.loc[dat_base['category'] == 'Disability_status', 'Known_stereotyped_groups'] = 'disabled'
    
    # Handle SES categories
    ses_mapping = {'low SES': 'lowSES', 'high SES': 'highSES'}
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].map(ses_mapping).fillna(dat_base['Known_stereotyped_groups'])
    
    # Calculate target locations
    def calculate_target_loc(row, ans_col):
        known_groups = str(row['Known_stereotyped_groups']).lower()
        ans_info = str(row[ans_col]).lower()
        
        if pd.isna(known_groups) or pd.isna(ans_info):
            return 0
        
        # Special handling for Age category
        if row['category'] == 'Age':
            if (known_groups == 'nonold' and ans_info == 'nonold') or (known_groups == 'old' and ans_info == 'old'):
                return 1
            return 0
        
        # Special handling for Nationality category
        if row['category'] == 'Nationality':
            ans_text = str(row[ans_col.replace('_info', '_text')]).lower()
            if known_groups in ans_text:
                return 1
            return 0
        
        # Special handling for Race category (separate 'african' / 'african american')
        if 'african american' in known_groups and row[ans_col.replace('_info', '_text')] == 'African':
            return 0
        
        # Special handling for trans
        if known_groups == 'trans' and ans_info.startswith('trans'):
            return 1
        
        # General case
        if known_groups in ans_info:
            return 1
        
        return 0
    
    for i in range(3):
        dat_base[f'target_loc_{i}'] = dat_base.apply(
            lambda row: calculate_target_loc(row, f'ans{i}_info'), axis=1
        )
    
    # Determine final target location
    def get_target_loc(row):
        if row['target_loc_0'] == 1:
            return 0
        elif row['target_loc_1'] == 1:
            return 1
        elif row['target_loc_2'] == 1:
            return 2
        return np.nan
    
    dat_base['target_loc'] = dat_base.apply(get_target_loc, axis=1)
    dat_base = dat_base.rename(columns={'ProperName': 'label_type'})
    
    # Correct target location for non-negative examples
    def correct_target_loc(row):
        if row['question_polarity'] == 'nonneg':
            if row['target_loc'] == 0 and row['ans1_info'] != 'unknown':
                return 1
            elif row['target_loc'] == 0 and row['ans2_info'] != 'unknown':
                return 2
            elif row['target_loc'] == 1 and row['ans0_info'] != 'unknown':
                return 0
            elif row['target_loc'] == 1 and row['ans2_info'] != 'unknown':
                return 2
            elif row['target_loc'] == 2 and row['ans0_info'] != 'unknown':
                return 0
            elif row['target_loc'] == 2 and row['ans1_info'] != 'unknown':
                return 1
        return row['target_loc']
    
    dat_base['new_target_loc'] = dat_base.apply(correct_target_loc, axis=1)
    
    # Remove cases where target_loc was identified for more than one answer
    target_sum = dat_base['target_loc_0'] + dat_base['target_loc_1'] + dat_base['target_loc_2']
    dat_base.loc[target_sum > 1, 'new_target_loc'] = np.nan
    
    dat_base_selected = dat_base[[
        'category', 'question_index', 'example_id', 'new_target_loc', 
        'label_type', 'Known_stereotyped_groups', 'Relevant_social_values'
    ]].copy()
    dat_base_selected = dat_base_selected.rename(columns={'new_target_loc': 'target_loc'})

print("Processing intersectional categories...")
# Process intersectional templates
intersectional_data = []
for template_file in template_files:
    if '_x_' in template_file.name:
        temp = pd.read_csv(template_file)
        if all(col in temp.columns for col in ['Category', 'Known_stereotyped_race', 'Known_stereotyped_var2', 'Q_id']):
            temp2 = temp[[
                'Category', 'Known_stereotyped_race', 'Known_stereotyped_var2', 
                'Q_id', 'Relevant_social_values', 'Proper_nouns_only'
            ]].copy()
            temp2 = temp2.rename(columns={'Category': 'category', 'Q_id': 'question_index'})
            temp2['question_index'] = temp2['question_index'].astype(str)
            intersectional_data.append(temp2)

if intersectional_data:
    intersectional_data = pd.concat(intersectional_data, ignore_index=True)
    
    # Standardize category names
    intersectional_data['category'] = intersectional_data['category'].replace('Gender_x_race', 'Race_x_gender')
    
    # Group and deduplicate
    intersectional_data = intersectional_data.groupby([
        'category', 'question_index', 'Known_stereotyped_race', 'Known_stereotyped_var2',
        'Relevant_social_values', 'Proper_nouns_only'
    ]).size().reset_index(name='count').drop('count', axis=1)
    
    # Merge with main data
    dat4_intersectional = pd.merge(dat, intersectional_data, on=['category', 'question_index'], how='left')
    dat4_intersectional = dat4_intersectional.dropna(subset=['example_id'])

# Process Race_x_gender category
race_gender_data = []
if not dat4_intersectional.empty:
    dat_race_x_gender = dat4_intersectional[dat4_intersectional['category'] == 'Race_x_gender'].copy()
    
    if not dat_race_x_gender.empty:
        dat_race_x_gender = dat_race_x_gender.drop('stereotyped_groups', axis=1, errors='ignore')
        
        # Add label type
        dat_race_x_gender['label_type'] = dat_race_x_gender['Proper_nouns_only'].apply(
            lambda x: 'name' if x else 'label'
        )
        dat_race_x_gender = dat_race_x_gender.drop('Proper_nouns_only', axis=1)
        
        # Clean stereotyped race
        dat_race_x_gender['Known_stereotyped_race'] = dat_race_x_gender['Known_stereotyped_race'].str.replace(r'["\[\]]', '', regex=True)
        
        # Extract gender and race info
        def extract_gender_race(info_str):
            if pd.isna(info_str) or info_str == 'unknown':
                return 'unknown', 'unknown'
            
            if len(str(info_str)) >= 3:
                gender = str(info_str)[0]
                race = str(info_str)[2:]
            else:
                gender = ''
                race = 'Latino'  # Default for empty race
            
            return gender, race
        
        for i in range(3):
            dat_race_x_gender[f'ans{i}_gender'], dat_race_x_gender[f'ans{i}_race'] = zip(
                *dat_race_x_gender[f'ans{i}_info'].apply(extract_gender_race)
            )
        
        # Handle gender detection from text
        for i in range(3):
            ans_text_col = f'ans{i}_text'
            ans_gender_col = f'ans{i}_gender'
            
            dat_race_x_gender[ans_gender_col] = dat_race_x_gender.apply(
                lambda row: 'M' if ' man' in str(row[ans_text_col]) else
                           'F' if ' woman' in str(row[ans_text_col]) else
                           row[ans_gender_col], axis=1
            )
        
        # Calculate correct answer alignments
        def get_correct_answer_info(row, info_type):
            label = row['label']
            if label == 0:
                return row[f'ans0_{info_type}']
            elif label == 1:
                return row[f'ans1_{info_type}']
            elif label == 2:
                return row[f'ans2_{info_type}']
            return np.nan
        
        dat_race_x_gender['corr_ans_race'] = dat_race_x_gender.apply(
            lambda row: get_correct_answer_info(row, 'race'), axis=1
        )
        dat_race_x_gender['corr_ans_gender'] = dat_race_x_gender.apply(
            lambda row: get_correct_answer_info(row, 'gender'), axis=1
        )
        
        # Check alignments
        dat_race_x_gender['corr_ans_aligns_race'] = dat_race_x_gender.apply(
            lambda row: 1 if str(row['corr_ans_race']) in str(row['Known_stereotyped_race']) else 0, axis=1
        )
        dat_race_x_gender['corr_ans_aligns_var2'] = dat_race_x_gender.apply(
            lambda row: 1 if str(row['corr_ans_gender']) in str(row['Known_stereotyped_var2']) else 0, axis=1
        )
        
        # Create condition variables
        def check_match(row, info_type):
            ans0 = row[f'ans0_{info_type}']
            ans1 = row[f'ans1_{info_type}']
            ans2 = row[f'ans2_{info_type}']
            
            if ans0 == ans1 or ans1 == ans2 or ans0 == ans2:
                return f'Match {info_type.title()}'
            return f'Mismatch {info_type.title()}'
        
        dat_race_x_gender['race_condition'] = dat_race_x_gender.apply(
            lambda row: check_match(row, 'race'), axis=1
        )
        dat_race_x_gender['gender_condition'] = dat_race_x_gender.apply(
            lambda row: check_match(row, 'gender'), axis=1
        )
        
        dat_race_x_gender['full_cond'] = dat_race_x_gender['race_condition'] + '\n ' + dat_race_x_gender['gender_condition']
        
        # Calculate target location
        def calculate_intersectional_target_loc(row):
            known_var2 = str(row['Known_stereotyped_var2'])
            known_race = str(row['Known_stereotyped_race'])
            
            for i in range(3):
                ans_gender = str(row[f'ans{i}_gender'])
                ans_race = str(row[f'ans{i}_race'])
                
                if (known_var2.startswith(ans_gender) and 
                    known_race in ans_race):
                    return i
            
            return np.nan
        
        dat_race_x_gender['target_loc'] = dat_race_x_gender.apply(calculate_intersectional_target_loc, axis=1)
        
        # Correct for non-negative examples
        def correct_intersectional_target_loc(row):
            if row['question_polarity'] == 'nonneg':
                if row['target_loc'] == 0 and row['ans1_gender'] != 'unknown':
                    return 1
                elif row['target_loc'] == 0 and row['ans2_gender'] != 'unknown':
                    return 2
                elif row['target_loc'] == 1 and row['ans0_gender'] != 'unknown':
                    return 0
                elif row['target_loc'] == 1 and row['ans2_gender'] != 'unknown':
                    return 2
                elif row['target_loc'] == 2 and row['ans0_gender'] != 'unknown':
                    return 0
                elif row['target_loc'] == 2 and row['ans1_gender'] != 'unknown':
                    return 1
            return row['target_loc']
        
        dat_race_x_gender['new_target_loc'] = dat_race_x_gender.apply(correct_intersectional_target_loc, axis=1)
        
        dat_racegen_selected = dat_race_x_gender[[
            'category', 'question_index', 'example_id', 'new_target_loc',
            'label_type', 'Known_stereotyped_race', 'Known_stereotyped_var2',
            'Relevant_social_values', 'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond'
        ]].copy()
        dat_racegen_selected = dat_racegen_selected.rename(columns={'new_target_loc': 'target_loc'})
        
        race_gender_data.append(dat_racegen_selected)

# Process Race_x_SES category
race_ses_data = []
if not dat4_intersectional.empty:
    dat_race_x_ses = dat4_intersectional[dat4_intersectional['category'] == 'Race_x_SES'].copy()
    
    if not dat_race_x_ses.empty:
        dat_race_x_ses = dat_race_x_ses.drop('stereotyped_groups', axis=1, errors='ignore')
        
        # Add label type
        dat_race_x_ses['label_type'] = dat_race_x_ses['Proper_nouns_only'].apply(
            lambda x: 'name' if x else 'label'
        )
        dat_race_x_ses = dat_race_x_ses.drop('Proper_nouns_only', axis=1)
        
        # Clean stereotyped race
        dat_race_x_ses['Known_stereotyped_race'] = dat_race_x_ses['Known_stereotyped_race'].str.replace(r'["\[\]]', '', regex=True)
        
        # Extract SES and race info
        def extract_ses_race(info_str):
            if pd.isna(info_str) or info_str == 'unknown':
                return 'unknown', 'unknown'
            
            info_str = str(info_str)
            if len(info_str) >= 8:
                ses = info_str[:6]  # First 6 characters for SES
                race = info_str[7:]  # After position 7 for race
            else:
                ses = info_str
                race = ''
            
            # Clean up
            race = re.sub(r'^[MF]-', '', race)  # Remove gender prefixes
            race = race.replace('-', '')
            
            # Fix SES naming
            if ses == 'highSE':
                ses = 'highSES'
            
            return ses, race
        
        for i in range(3):
            dat_race_x_ses[f'ans{i}_gender'], dat_race_x_ses[f'ans{i}_race'] = zip(
                *dat_race_x_ses[f'ans{i}_info'].apply(extract_ses_race)
            )
        
        # Calculate correct answer alignments
        dat_race_x_ses['corr_ans_race'] = dat_race_x_ses.apply(
            lambda row: get_correct_answer_info(row, 'race'), axis=1
        )
        dat_race_x_ses['corr_ans_gender'] = dat_race_x_ses.apply(
            lambda row: get_correct_answer_info(row, 'gender'), axis=1
        )
        
        # Handle complex race stereotypes
        def check_race_alignment(row):
            known_race = str(row['Known_stereotyped_race'])
            corr_race = str(row['corr_ans_race'])
            
            # Handle multiple race categories
            if known_race == "Black, African American, Latino, Hispanic":
                if corr_race in ['Hispanic', 'Latino']:
                    return 1 if 'Hispanic, Latino' in known_race else 0
                elif corr_race in ['Black', 'African American']:
                    return 1 if 'Black, African American' in known_race else 0
            
            return 1 if corr_race in known_race else 0
        
        dat_race_x_ses['corr_ans_aligns_race'] = dat_race_x_ses.apply(check_race_alignment, axis=1)
        dat_race_x_ses['corr_ans_aligns_var2'] = dat_race_x_ses.apply(
            lambda row: 1 if str(row['corr_ans_gender']) in str(row['Known_stereotyped_var2']) else 0, axis=1
        )
        
        # Create condition variables
        dat_race_x_ses['race_condition'] = dat_race_x_ses.apply(
            lambda row: check_match(row, 'race'), axis=1
        )
        dat_race_x_ses['SES_condition'] = dat_race_x_ses.apply(
            lambda row: check_match(row, 'gender'), axis=1  # Using gender column for SES
        )
        
        dat_race_x_ses['full_cond'] = dat_race_x_ses['race_condition'] + '\n ' + dat_race_x_ses['SES_condition']
        
        # Update Known_stereotyped_race for specific cases
        def update_known_race(row):
            known_race = row['Known_stereotyped_race']
            if known_race == "Black, African American, Latino, Hispanic":
                # Check which race appears in answers
                races_in_answers = [row[f'ans{i}_race'] for i in range(3)]
                if any(race in ['Hispanic', 'Latino'] for race in races_in_answers):
                    return "Hispanic, Latino"
                elif any(race in ['Black', 'African American'] for race in races_in_answers):
                    return "Black, African American"
            return known_race
        
        dat_race_x_ses['Known_stereotyped_race'] = dat_race_x_ses.apply(update_known_race, axis=1)
        
        # Calculate target location
        def calculate_ses_target_loc(row):
            known_var2 = str(row['Known_stereotyped_var2'])
            known_race = str(row['Known_stereotyped_race'])
            
            for i in range(3):
                ans_ses = str(row[f'ans{i}_gender'])  # SES info is in gender column
                ans_race = str(row[f'ans{i}_race'])
                
                if (known_var2[:3] == ans_ses[:3] and 
                    known_race in ans_race):
                    return i
            
            return np.nan
        
        dat_race_x_ses['target_loc'] = dat_race_x_ses.apply(calculate_ses_target_loc, axis=1)
        
        # Correct for non-negative examples
        dat_race_x_ses['new_target_loc'] = dat_race_x_ses.apply(correct_intersectional_target_loc, axis=1)
        
        dat_raceses_selected = dat_race_x_ses[[
            'category', 'question_index', 'example_id', 'new_target_loc',
            'label_type', 'Known_stereotyped_race', 'Known_stereotyped_var2',
            'Relevant_social_values', 'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond'
        ]].copy()
        dat_raceses_selected = dat_raceses_selected.rename(columns={'new_target_loc': 'target_loc'})
        
        race_ses_data.append(dat_raceses_selected)

print("Combining all metadata...")
# Combine all metadata
all_metadata = []

# Add base categories
if not dat_base_selected.empty:
    # Add missing columns for consistency
    for col in ['Known_stereotyped_var2', 'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond']:
        if col not in dat_base_selected.columns:
            dat_base_selected[col] = np.nan
    
    # Rename for consistency
    dat_base_selected = dat_base_selected.rename(columns={'Known_stereotyped_groups': 'Known_stereotyped_groups'})
    all_metadata.append(dat_base_selected)

# Add intersectional categories
for data in race_gender_data + race_ses_data:
    if not data.empty:
        # Add missing column for consistency
        if 'Known_stereotyped_groups' not in data.columns:
            data['Known_stereotyped_groups'] = np.nan
        all_metadata.append(data)

if all_metadata:
    final_metadata = pd.concat(all_metadata, ignore_index=True)
    
    # Ensure all required columns are present
    required_columns = [
        'category', 'question_index', 'example_id', 'target_loc', 'label_type',
        'Known_stereotyped_race', 'Known_stereotyped_var2', 'Relevant_social_values',
        'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond', 'Known_stereotyped_groups'
    ]
    
    for col in required_columns:
        if col not in final_metadata.columns:
            final_metadata[col] = np.nan
    
    # Reorder columns
    final_metadata = final_metadata[required_columns]
    
    # Save to CSV
    output_file = "additional_metadata.csv"
    final_metadata.to_csv(output_file, index=False)
    
    print(f"Metadata saved to '{output_file}'")
    print(f"Total records: {len(final_metadata)}")
    print(f"Categories: {final_metadata['category'].unique()}")
    print(f"Records with missing target_loc: {final_metadata['target_loc'].isna().sum()}")
    
    # Display summary statistics
    print("\nSummary by category:")
    print(final_metadata.groupby('category').agg({
        'example_id': 'count',
        'target_loc': lambda x: x.notna().sum()
    }).rename(columns={'example_id': 'total_records', 'target_loc': 'valid_target_loc'}))

else:
    print("No metadata generated!")