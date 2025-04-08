import pandas as pd
from transformers import AutoTokenizer

# Load ClinicalBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# File paths
input_file = "NOTEEVENTS.csv"
output_file = "notes_tokenized.csv"

batch_size = 8
first_batch = True  # For writing headers only once

# Loop through file in chunks
for chunk in pd.read_csv(input_file, chunksize=batch_size):
    batch_texts = chunk['TEXT'].tolist()
    hadm_ids = chunk['HADM_ID'].tolist()

    # Tokenize text
    tokenized = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors=None  # Return lists instead of tensors
    )

    # Create a DataFrame for the tokenized data
    tokenized_df = pd.DataFrame({
        'HADM_ID': hadm_ids,
        'input_ids': [str(ids) for ids in tokenized['input_ids']],
        #'attention_mask': [str(mask) for mask in tokenized['attention_mask']],
    })

    # Save to output file
    tokenized_df.to_csv(output_file, mode='a', index=False, header=first_batch)
    first_batch = False  # Only write header for the first batch

    print(f"Processed batch of {len(batch_texts)} rows.")

print("✅ Tokenization complete. Saved to file.")