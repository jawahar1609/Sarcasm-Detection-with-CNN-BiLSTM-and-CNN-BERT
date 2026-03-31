import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Import preprocessing
from preprocessing.download import auto_download_dataset
from preprocessing import prepare_data

# Import scripts
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.xai import run_xai_analysis


# Available options
AVAILABLE_MODELS = ['cnn', 'lstm', 'bert', 'cnn_bilstm', 'cnn_bert']
AVAILABLE_DATASETS = ['sarcasm_news', 'sarc']


def run_pipeline(
    model_name,
    dataset_name,
    skip_train=False,
    skip_eval=False,
    skip_xai=False,
    epochs=10,
    batch_size=64,
):
    print(f"\n{'='*80}")
    print(f"PIPELINE: {model_name.upper()} on {dataset_name}")
    print(f"{'='*80}\n")
    
    # Step 1: Download raw data
    print("STEP 1: Downloading raw data...")
    try:
        raw_data_path = auto_download_dataset(dataset_name)
        print(f"Data ready at: {raw_data_path}")
    except Exception as e:
        print(f"Failed to download data: {e}")
        return False
    
    # Step 2: Preprocess data
    print("\nSTEP 2: Preprocessing data...")
    try:
        train_df, val_df, test_df, _ = prepare_data(
            dataset=dataset_name,
            raw_data_path=str(raw_data_path),
            save_to_disk=False,
            min_freq=2
        )
        print(f"Preprocessed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    except Exception as e:
        print(f"Failed to preprocess data: {e}")
        return False
    
    # Step 3: Train model
    model_path = f"results/models/{model_name}_{dataset_name}_model.pt"
    
    if skip_train and Path(model_path).exists():
        print(f"\nSTEP 3: Training (SKIPPED - model exists)")
        print(f"Using existing model: {model_path}")
    else:
        print(f"\nSTEP 3: Training model...")
        try:
            model_path = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                output_dir="results/models",
                epochs=epochs,
                batch_size=batch_size,
            )
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"Failed to train model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 4: Evaluate model
    if skip_eval:
        print(f"\nSTEP 4: Evaluation (SKIPPED)")
    else:
        print(f"\nSTEP 4: Evaluating model...")
        try:
            metrics = evaluate_model(
                model_path=model_path,
                dataset_name=dataset_name,
                test_df=test_df,
                output_dir="results/metrics",
                batch_size=batch_size,
            )
            print(f"Evaluation complete: Accuracy={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Failed to evaluate model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 5: XAI analysis
    if skip_xai:
        print(f"\nSTEP 5: XAI Analysis (SKIPPED)")
    else:
        print(f"\nSTEP 5: Running XAI analysis...")
        try:
            viz_paths = run_xai_analysis(
                model_path=model_path,
                dataset_name=dataset_name,
                test_df=test_df,
                output_dir="results/xai",
                num_samples=10,
            )
            print(f"XAI analysis complete: {len(viz_paths)} visualizations generated")
        except Exception as e:
            print(f"Failed to run XAI analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE: {model_name.upper()} on {dataset_name}")
    print(f"{'='*80}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Unified pipeline for sarcasm detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models on all datasets
  python run.py
  
  # Run specific model and dataset
  python run.py --model cnn --dataset sarcasm_news
  
  # Run with custom settings
  python run.py --model cnn_bert --dataset sarc --epochs 15 --skip-xai
  
  # Skip training if models exist
  python run.py --skip-train
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all'] + AVAILABLE_MODELS,
        help='Model to train (default: all)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all'] + AVAILABLE_DATASETS,
        help='Dataset to use (default: all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training if model exists'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation'
    )
    parser.add_argument(
        '--skip-xai',
        action='store_true',
        help='Skip XAI analysis'
    )
    
    args = parser.parse_args()
    
    # Determine which models and datasets to run
    models = AVAILABLE_MODELS if args.model == 'all' else [args.model]
    datasets = AVAILABLE_DATASETS if args.dataset == 'all' else [args.dataset]
    
    print("\n" + "="*80)
    print("HYBRID SARCASM DETECTION - UNIFIED PIPELINE")
    print("="*80)
    print(f"\nModels: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Skip Training: {args.skip_train}")
    print(f"Skip Evaluation: {args.skip_eval}")
    print(f"Skip XAI: {args.skip_xai}")
    print("="*80)
    
    # Run pipeline for each combination
    total = len(models) * len(datasets)
    completed = 0
    failed = 0
    
    for dataset in datasets:
        for model in models:
            success = run_pipeline(
                model_name=model,
                dataset_name=dataset,
                skip_train=args.skip_train,
                skip_eval=args.skip_eval,
                skip_xai=args.skip_xai,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            
            if success:
                completed += 1
            else:
                failed += 1
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total pipelines: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print("="*80 + "\n")
    
    # Results organization
    print("Results are organized as follows:")
    print("  results/models/   - Trained model checkpoints")
    print("  results/metrics/  - Evaluation results (JSON)")
    print("  results/xai/      - XAI visualizations (PNG)")
    print()
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
