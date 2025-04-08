import pandas as pd
df=pd.read_csv("CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_grouped.csv")
notes=pd.read_csv("notes_tokenized.csv")
final=df.merge(notes, on='HADM_ID', how='left').fillna(0)
final.to_csv("combination.csv", index=False)

