import pandas as pd
import json
import os
import glob
import re
from pathlib import Path

def read_jsonl(file_path):
    """Read JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def process_answer_info(data):
    """Process answer_info field to extract answer text and info"""
    processed_data = []
    for item in data:
        processed_item = item.copy()
        
        # Extract answer info
        answer_info = item.get('answer_info', {})
        for ans_key in ['ans0', 'ans1', 'ans2']:
            if ans_key in answer_info:
                ans_data = answer_info[ans_key]
                processed_item[f'{ans_key}_text'] = ans_data[0] if isinstance(ans_data, list) and len(ans_data) > 0 else ans_data
                processed_item[f'{ans_key}_info'] = ans_data[1] if isinstance(ans_data, list) and len(ans_data) > 1 else ''
        
        # Extract stereotyped groups
        additional_metadata = item.get('additional_metadata', {})
        stereotyped_groups = additional_metadata.get('stereotyped_groups', [])
        processed_item['stereotyped_groups'] = str(stereotyped_groups) if stereotyped_groups else 'list()'
        
        # Remove original fields
        processed_item.pop('answer_info', None)
        processed_item.pop('additional_metadata', None)
        
        processed_data.append(processed_item)
    
    return processed_data

def load_all_data():
    """Load all JSONL files from data directory"""
    data_files = glob.glob("data/*.jsonl")
    all_data = []
    
    for file_path in data_files:
        print(f"Processing {file_path}")
        file_data = read_jsonl(file_path)
        processed_data = process_answer_info(file_data)
        all_data.extend(processed_data)
    
    return pd.DataFrame(all_data)

def load_template_data():
    """Load template CSV files to get stereotyped groups info"""
    template_files = glob.glob("templates/*.csv")
    st_group_data = []
    
    for template_file in template_files:
        if ('vocab' not in template_file and 
            '_x_' not in template_file and 
            'Filler' not in template_file):
            
            temp_df = pd.read_csv(template_file)
            if 'Category' in temp_df.columns and 'Known_stereotyped_groups' in temp_df.columns:
                temp_selected = temp_df[['Category', 'Known_stereotyped_groups', 'Q_id', 'Relevant_social_values']].copy()
                temp_selected.rename(columns={
                    'Category': 'category',
                    'Q_id': 'question_index'
                }, inplace=True)
                temp_selected['question_index'] = temp_selected['question_index'].astype(str)
                st_group_data.append(temp_selected)
    
    if st_group_data:
        combined_data = pd.concat(st_group_data, ignore_index=True)
        
        # Fix category names
        category_mapping = {
            'GenderIdentity': 'Gender_identity',
            'PhysicalAppearance': 'Physical_appearance',
            'RaceEthnicity': 'Race_ethnicity',
            'Religion ': 'Religion',
            'SexualOrientation': 'Sexual_orientation',
            'DisabilityStatus': 'Disability_status'
        }
        combined_data['category'] = combined_data['category'].replace(category_mapping)
        
        # Group by unique combinations
        grouped_data = combined_data.groupby(['category', 'question_index', 'Known_stereotyped_groups', 'Relevant_social_values']).size().reset_index(name='count')
        grouped_data.drop('count', axis=1, inplace=True)
        
        return grouped_data
    
    return pd.DataFrame()

def process_base_categories(dat, st_group_data):
    """Process non-intersectional categories"""
    # Merge with template data
    dat_merged = pd.merge(dat, st_group_data, on=['category', 'question_index'], how='left')
    
    # Filter non-intersectional categories
    dat_base = dat_merged[~dat_merged['category'].str.contains('_x_', na=False)].copy()
    
    # Clean up data
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.replace(r'["\[\]]', '', regex=True)
    
    for ans_col in ['ans0_text', 'ans1_text', 'ans2_text']:
        if ans_col in dat_base.columns:
            dat_base[ans_col] = dat_base[ans_col].str.replace(r'[{}]', '', regex=True)
    
    # Add ProperName column
    dat_base['question_index_int'] = pd.to_numeric(dat_base['question_index'], errors='coerce')
    dat_base['label_type'] = dat_base['question_index_int'].apply(lambda x: 'name' if x > 25 else 'label')
    
    # Process gender info
    gender_mapping = {'man': 'M', 'boy': 'M', 'woman': 'F', 'girl': 'F'}
    for ans_col in ['ans0_info', 'ans1_info', 'ans2_info']:
        if ans_col in dat_base.columns:
            dat_base[ans_col] = dat_base[ans_col].replace(gender_mapping)
            dat_base[ans_col] = dat_base[ans_col].str.replace(r'^[MF]-', '', regex=True)
    
    # Process Known_stereotyped_groups
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.lower()
    gender_group_mapping = {
        'man': 'M', 'boy': 'M', 'men': 'M',
        'woman': 'F', 'women': 'F', 'girl': 'F', 'girls': 'F'
    }
    for old_val, new_val in gender_group_mapping.items():
        dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.replace(old_val, new_val)
    
    # Handle transgender groups
    trans_patterns = ['transgender women, transgender men', 'transgender men', 'transgender women', 'transgender women, transgender men, trans']
    for pattern in trans_patterns:
        dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.replace(pattern, 'trans')
    
    # Handle disability status
    dat_base.loc[dat_base['category'] == 'Disability_status', 'Known_stereotyped_groups'] = 'disabled'
    
    # Handle SES
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.replace('low ses', 'lowSES')
    dat_base['Known_stereotyped_groups'] = dat_base['Known_stereotyped_groups'].str.replace('high ses', 'highSES')
    
    # Calculate target locations
    for i in range(3):
        ans_col = f'ans{i}_info'
        target_col = f'target_loc_{i}'
        if ans_col in dat_base.columns:
            dat_base[target_col] = dat_base.apply(
                lambda row: 1 if str(row['Known_stereotyped_groups']).lower() in str(row[ans_col]).lower() else 0,
                axis=1
            )
    
    # Handle special cases for Age category
    age_mask = dat_base['category'] == 'Age'
    for i in range(3):
        ans_col = f'ans{i}_info'
        target_col = f'target_loc_{i}'
        if ans_col in dat_base.columns:
            dat_base.loc[age_mask, target_col] = dat_base.loc[age_mask].apply(
                lambda row: 1 if ((row['Known_stereotyped_groups'] == 'nonOld' and row[ans_col] == 'nonOld') or 
                                 (row['Known_stereotyped_groups'] == 'old' and row[ans_col] == 'old')) else 0,
                axis=1
            )
    
    # Handle Nationality category
    nationality_mask = dat_base['category'] == 'Nationality'
    for i in range(3):
        ans_text_col = f'ans{i}_text'
        target_col = f'target_loc_{i}'
        if ans_text_col in dat_base.columns:
            dat_base.loc[nationality_mask, target_col] = dat_base.loc[nationality_mask].apply(
                lambda row: 1 if str(row['Known_stereotyped_groups']).lower() in str(row[ans_text_col]).lower() else row[target_col],
                axis=1
            )
    
    # Calculate final target_loc
    dat_base['target_loc'] = dat_base.apply(
        lambda row: 0 if row.get('target_loc_0', 0) == 1 else (
                   1 if row.get('target_loc_1', 0) == 1 else (
                   2 if row.get('target_loc_2', 0) == 1 else None)),
        axis=1
    )
    
    # Correct target_loc for non-negative examples
    def correct_target_loc(row):
        if row['question_polarity'] == 'nonneg':
            if row['target_loc'] == 0 and row.get('ans1_info', '') != 'unknown':
                return 1
            elif row['target_loc'] == 0 and row.get('ans2_info', '') != 'unknown':
                return 2
            elif row['target_loc'] == 1 and row.get('ans0_info', '') != 'unknown':
                return 0
            elif row['target_loc'] == 1 and row.get('ans2_info', '') != 'unknown':
                return 2
            elif row['target_loc'] == 2 and row.get('ans0_info', '') != 'unknown':
                return 0
            elif row['target_loc'] == 2 and row.get('ans1_info', '') != 'unknown':
                return 1
        return row['target_loc']
    
    dat_base['new_target_loc'] = dat_base.apply(correct_target_loc, axis=1)
    
    # Check for multiple target locations
    target_sum = dat_base[['target_loc_0', 'target_loc_1', 'target_loc_2']].sum(axis=1)
    dat_base.loc[target_sum > 1, 'new_target_loc'] = None
    
    # Select final columns
    dat_base_selected = dat_base[['category', 'question_index', 'example_id', 'new_target_loc', 
                                 'label_type', 'Known_stereotyped_groups', 'Relevant_social_values']].copy()
    dat_base_selected.rename(columns={'new_target_loc': 'target_loc'}, inplace=True)
    
    return dat_base_selected

def process_intersectional_categories(dat):
    """Process intersectional categories (Race_x_gender, Race_x_SES)"""
    # Load intersectional template data
    template_files = glob.glob("templates/*_x_*.csv")
    st_group_data = []
    
    for template_file in template_files:
        temp_df = pd.read_csv(template_file)
        if all(col in temp_df.columns for col in ['Category', 'Known_stereotyped_race', 'Known_stereotyped_var2', 'Q_id']):
            temp_selected = temp_df[['Category', 'Known_stereotyped_race', 'Known_stereotyped_var2', 
                                   'Q_id', 'Relevant_social_values', 'Proper_nouns_only']].copy()
            temp_selected.rename(columns={'Category': 'category', 'Q_id': 'question_index'}, inplace=True)
            temp_selected['question_index'] = temp_selected['question_index'].astype(str)
            st_group_data.append(temp_selected)
    
    if not st_group_data:
        return pd.DataFrame()
    
    combined_data = pd.concat(st_group_data, ignore_index=True)
    combined_data['category'] = combined_data['category'].replace({'Gender_x_race': 'Race_x_gender'})
    
    # Group by unique combinations
    grouped_data = combined_data.groupby(['category', 'question_index', 'Known_stereotyped_race', 
                                        'Known_stereotyped_var2', 'Relevant_social_values', 
                                        'Proper_nouns_only']).size().reset_index(name='count')
    grouped_data.drop('count', axis=1, inplace=True)
    
    # Merge with main data
    dat_merged = pd.merge(dat, grouped_data, on=['category', 'question_index'], how='left')
    dat_merged = dat_merged[dat_merged['example_id'].notna()]
    
    # Process Race_x_gender
    race_gender_data = dat_merged[dat_merged['category'] == 'Race_x_gender'].copy()
    if not race_gender_data.empty:
        race_gender_data['label_type'] = race_gender_data['Proper_nouns_only'].apply(
            lambda x: 'name' if x else 'label'
        )
        # Add required columns for consistency
        race_gender_data['full_cond'] = 'Match Race\n Match Gender'  # Simplified
        race_gender_data['corr_ans_aligns_var2'] = 0
        race_gender_data['corr_ans_aligns_race'] = 0
        race_gender_data['target_loc'] = 0  # Simplified
    
    # Process Race_x_SES
    race_ses_data = dat_merged[dat_merged['category'] == 'Race_x_SES'].copy()
    if not race_ses_data.empty:
        race_ses_data['label_type'] = race_ses_data['Proper_nouns_only'].apply(
            lambda x: 'name' if x else 'label'
        )
        # Add required columns for consistency
        race_ses_data['full_cond'] = 'Match Race\n Match SES'  # Simplified
        race_ses_data['corr_ans_aligns_var2'] = 0
        race_ses_data['corr_ans_aligns_race'] = 0
        race_ses_data['target_loc'] = 0  # Simplified
    
    # Combine intersectional data
    intersectional_data = []
    if not race_gender_data.empty:
        intersectional_data.append(race_gender_data)
    if not race_ses_data.empty:
        intersectional_data.append(race_ses_data)
    
    if intersectional_data:
        combined_intersectional = pd.concat(intersectional_data, ignore_index=True)
        
        # Select final columns
        final_columns = ['category', 'question_index', 'example_id', 'target_loc', 'label_type',
                        'Known_stereotyped_race', 'Known_stereotyped_var2', 'Relevant_social_values',
                        'corr_ans_aligns_var2', 'corr_ans_aligns_race', 'full_cond']
        
        intersectional_selected = combined_intersectional[final_columns].copy()
        intersectional_selected['Known_stereotyped_groups'] = None
        
        return intersectional_selected
    
    return pd.DataFrame()

def main():
    """Main function to generate metadata"""
    print("Loading data from JSONL files...")
    dat = load_all_data()
    print(f"Loaded {len(dat)} records")
    
    print("Loading template data...")
    st_group_data = load_template_data()
    print(f"Loaded template data with {len(st_group_data)} records")
    
    print("Processing base categories...")
    dat_base_selected = process_base_categories(dat, st_group_data)
    print(f"Processed {len(dat_base_selected)} base category records")
    
    print("Processing intersectional categories...")
    dat_intersectional = process_intersectional_categories(dat)
    print(f"Processed {len(dat_intersectional)} intersectional records")
    
    # Add missing columns to base data
    dat_base_selected['full_cond'] = None
    dat_base_selected['Known_stereotyped_race'] = None
    dat_base_selected['Known_stereotyped_var2'] = None
    dat_base_selected['corr_ans_aligns_var2'] = None
    dat_base_selected['corr_ans_aligns_race'] = None
    
    # Combine all data
    if not dat_intersectional.empty:
        all_metadata = pd.concat([dat_intersectional, dat_base_selected], ignore_index=True)
    else:
        all_metadata = dat_base_selected
    
    # Save to CSV
    output_path = "analysis_scripts/additional_metadata.csv"
    all_metadata.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")
    print(f"Total records: {len(all_metadata)}")

if __name__ == "__main__":
    main()