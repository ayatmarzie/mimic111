import pandas as pd

def multi_hot_encode_icd9(file_path):
    df = pd.read_csv(file_path)

    # Convert string representation of list to actual list
    if isinstance(df['ICD9_CODE'].iloc[0], str):
        df['ICD9_CODE'] = df['ICD9_CODE'].apply(eval)

    # Explode ICD9_CODE list into rows and one-hot encode
    icd_multi_hot = df['ICD9_CODE'].explode().dropna().astype(str)
    icd_dummies = pd.get_dummies(icd_multi_hot, prefix='ICD9')
    icd_dummies['HADM_ID'] = df.loc[icd_multi_hot.index, 'HADM_ID'].values

    # Group by HADM_ID and aggregate with max
    icd_encoded = icd_dummies.groupby('HADM_ID').max().reset_index()

    # Merge with original dataframe (drop old ICD9_CODE list)
    df = df.drop(columns=['ICD9_CODE'])
    df_final = df.merge(icd_encoded, on='HADM_ID', how='left').fillna(0)

    # Save final version
    output_file = file_path.replace('.csv', '_multi_hot.csv')
    df_final.to_csv(output_file, index=False)
    print(f"✅ Multi-hot encoded file saved as: {output_file}")
multi_hot_encode_icd9('combination.csv')

