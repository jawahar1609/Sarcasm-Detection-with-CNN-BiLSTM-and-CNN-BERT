import pandas as pd
from pathlib import Path

from .utils import clean_text_basic, clean_text, split_train_val_test, print_split_summary
from .config import RANDOM_SEED, TEST_SIZE, VAL_SIZE, ensure_dir


def preprocess_headlines(
        json_path,
        out_dir="processed_data/sarcasm_news",
        save_to_disk=False,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED):
    print(f"Loading Headlines JSON: {json_path}")
    if save_to_disk:
        ensure_dir(out_dir)

    # Load data
    data = pd.read_json(json_path, lines=True)
    data = data[['headline', 'is_sarcastic']]
    data = data.rename(columns={'headline': 'text', 'is_sarcastic': 'label'})
    
    print(f"Total samples: {len(data)}")
    print("\nLabel distribution:")
    print(data['label'].value_counts())

    # Clean text
    data['text'] = data['text'].apply(clean_text_basic)
    data = data[data['text'].str.strip() != ""].reset_index(drop=True)
    data = data.drop_duplicates(subset="text").reset_index(drop=True)
    
    print(f"\nSamples after cleaning: {len(data)}")

    # Use centralized splitting function
    train_data, val_data, test_data = split_train_val_test(
        data,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed
    )

    # Use centralized print function
    print_split_summary(train_data, val_data, test_data)

    # Save processed data (optional)
    if save_to_disk:
        out_path = Path(out_dir)
        train_data.to_csv(out_path / "train.csv", index=False)
        val_data.to_csv(out_path / "val.csv", index=False)
        test_data.to_csv(out_path / "test.csv", index=False)
        data.to_csv(out_path / "sarcasm_headlines_cleaned.csv", index=False)
        print(f"\nSaved processed files to: {out_dir}")
    else:
        print(f"\nPreprocessing complete (in-memory only)")
    
    return train_data, val_data, test_data


def preprocess_sarc(
        data_dir,
        train_file="train-balanced-sarcasm.csv",
        out_dir="processed_data/sarc",
        save_to_disk=False,
        include_context=True,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED):
    print(f"Loading SARC data from: {data_dir}")
    if save_to_disk:
        ensure_dir(out_dir)

    # Load train file
    train_path = Path(data_dir) / train_file
    df = pd.read_csv(train_path)

    # Select and rename columns
    columns_to_keep = ["label", "comment", "parent_comment"]
    df = df[columns_to_keep]
    df = df.rename(columns={
        "comment": "text",
        "parent_comment": "context"
    })

    print(f"Loaded rows: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)
    
    if include_context and "context" in df.columns:
        df["clean_context"] = df["context"].apply(clean_text)
    else:
        df["clean_context"] = ""

    # Combine context with text if requested
    if include_context:
        df["final_text"] = df["clean_context"] + " [SEP] " + df["clean_text"]
    else:
        df["final_text"] = df["clean_text"]

    # Remove empty rows and duplicates
    df = df[df["final_text"].str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset="final_text").reset_index(drop=True)

    print(f"\nRows after cleaning: {len(df)}")

    # Use centralized splitting function
    train_df, val_df, test_df = split_train_val_test(
        df,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed
    )

    # Use centralized print function
    print_split_summary(train_df, val_df, test_df)

    # Save processed files (optional)
    if save_to_disk:
        out_path = Path(out_dir)
        train_df[["final_text", "label"]].to_csv(out_path / "train.csv", index=False)
        val_df[["final_text", "label"]].to_csv(out_path / "val.csv", index=False)
        test_df[["final_text", "label"]].to_csv(out_path / "test.csv", index=False)
        
        # Full cleaned file
        df.to_csv(out_path / "sarc_cleaned_full.csv", index=False)
        print(f"\nSaved processed files to: {out_dir}")
    else:
        print(f"\nPreprocessing complete (in-memory only)")
    
    print("SARC preprocessing complete!")

    # Return only the necessary columns
    return train_df[["final_text", "label"]], val_df[["final_text", "label"]], test_df[["final_text", "label"]]