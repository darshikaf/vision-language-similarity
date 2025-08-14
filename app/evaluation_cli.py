"""
Simple CLIP Evaluation Script

A minimal, synchronous script for evaluating images with CLIP scores.
Much easier to debug than the async version.

Usage:
    python evaluation_cli.py <csv_path> [--service-url URL]
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
import requests
from typing import Optional

try:
    import pandas as pd
except ModuleNotFoundError as e:
    if e.name == "pandas":
        raise ImportError(
            "'pandas' library is required to run this script."
        ) from e
    else:
        raise


def extract_user_id_from_url(url: str) -> Optional[str]:
    """Extract the user ID (first UUID) from Leonardo AI URL."""
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    matches = re.findall(uuid_pattern, url, re.IGNORECASE)
    return matches[0] if matches else None


def find_local_image(url: str, data_dir: Path) -> Optional[str]:
    """Find local image file corresponding to the URL."""
    user_id = extract_user_id_from_url(url)
    if not user_id:
        return None
    
    for ext in [".png", ".jpg", ".jpeg"]:
        image_path = data_dir / f"{user_id}{ext}"
        if image_path.exists():
            return str(image_path.absolute())
    return None


def evaluate_single(service_url: str, image_input: str, text_prompt: str, model_config: str = "fast"):
    """Evaluate a single image-text pair."""
    request_data = {
        "image_input": image_input,
        "text_prompt": text_prompt,
        "model_config_name": model_config
    }
    
    try:
        response = requests.post(
            f"{service_url}/evaluator/v1/evaluation/single",
            json=request_data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "clip_score": None}
    except Exception as e:
        return {"error": str(e), "clip_score": None}


def evaluate_batch(service_url: str, evaluations: list, model_config: str = "fast", batch_size: int = 32):
    """Evaluate multiple image-text pairs in batch."""
    request_data = {
        "evaluations": [
            {
                "image_input": eval_item["image_input"],
                "text_prompt": eval_item["text_prompt"],
                "model_config_name": model_config
            }
            for eval_item in evaluations
        ],
        "batch_size": batch_size,
        "show_progress": True
    }
    
    try:
        response = requests.post(
            f"{service_url}/evaluator/v1/evaluation/batch",
            json=request_data,
            timeout=300
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}


def run_single_evaluation_with_stats(service_url: str, df: pd.DataFrame):
    """Run single evaluation and collect statistics."""
    start_time = time.time()
    results = []
    api_calls = 0
    
    for i, row in df.iterrows():
        result = evaluate_single(service_url, row["url"], row["caption"])
        results.append(result)
        api_calls += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    successful = [r for r in results if r.get("clip_score") is not None]
    clip_scores = [r["clip_score"] for r in successful]
    
    stats = {
        "method": "single",
        "total_images": len(df),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_time_seconds": total_time,
        "api_calls": api_calls,
        "avg_clip_score": sum(clip_scores) / len(clip_scores) if clip_scores else 0.0,
        "images_per_second": len(successful) / total_time if total_time > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    return results, stats


def run_batch_evaluation_with_stats(service_url: str, df: pd.DataFrame, batch_size: int):
    """Run batch evaluation and collect statistics."""
    start_time = time.time()
    
    evaluations = [
        {"image_input": row["url"], "text_prompt": row["caption"]}
        for _, row in df.iterrows()
    ]
    
    batch_result = evaluate_batch(service_url, evaluations, batch_size=batch_size)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    results = batch_result.get("results", [])
    successful = [r for r in results if r.get("clip_score") is not None]
    clip_scores = [r["clip_score"] for r in successful]
    
    stats = {
        "method": "batch",
        "total_images": len(df),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_time_seconds": total_time,
        "api_calls": 1,
        "avg_clip_score": sum(clip_scores) / len(clip_scores) if clip_scores else 0.0,
        "images_per_second": len(successful) / total_time if total_time > 0 else 0,
        "batch_size": batch_size,
        "timestamp": datetime.now().isoformat()
    }
    
    return results, stats


def save_comparison_stats(stats_list, csv_path):
    """Save comparison statistics to CSV."""
    stats_df = pd.DataFrame(stats_list)
    stats_path = csv_path.parent / f"{csv_path.stem}_comparison_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Comparison statistics saved to: {stats_path}")
    return stats_path


def main():
    parser = argparse.ArgumentParser(description="Simple CLIP evaluation")
    parser.add_argument("csv_path", help="Path to CSV file or directory")
    parser.add_argument("--service-url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--batch", action="store_true", help="Use batch evaluation instead of single")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for batch evaluation")
    parser.add_argument("--compare", action="store_true", help="Run both single and batch evaluations and compare performance")
    
    args = parser.parse_args()
    
    # Find CSV file
    csv_path = Path(args.csv_path)
    if csv_path.is_file():
        data_dir = csv_path.parent
    else:
        data_dir = csv_path
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {data_dir}")
            return
        csv_path = csv_files[0]
    
    print(f"Processing: {csv_path}")
    print(f"Data directory: {data_dir}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    df["clip_score"] = None
    stats_list = []
    
    if args.compare:
        print("Running comparison mode: both single and batch evaluation")
        
        print("\n=== Running Single Evaluation ===")
        single_results, single_stats = run_single_evaluation_with_stats(args.service_url, df)
        stats_list.append(single_stats)
        print(f"Single evaluation completed: {single_stats['successful']}/{single_stats['total_images']} successful")
        print(f"Time: {single_stats['total_time_seconds']:.2f}s, Speed: {single_stats['images_per_second']:.2f} images/sec")
        
        print("\n=== Running Batch Evaluation ===")
        batch_results, batch_stats = run_batch_evaluation_with_stats(args.service_url, df, args.batch_size)
        stats_list.append(batch_stats)
        print(f"Batch evaluation completed: {batch_stats['successful']}/{batch_stats['total_images']} successful")
        print(f"Time: {batch_stats['total_time_seconds']:.2f}s, Speed: {batch_stats['images_per_second']:.2f} images/sec")
        
        # Use batch results for the output CSV (prefer batch if successful, otherwise single)
        results_to_use = batch_results if batch_stats["successful"] > 0 else single_results
        for i, result in enumerate(results_to_use):
            clip_score = result.get("clip_score")
            if clip_score is not None:
                df.at[i, "clip_score"] = clip_score
        
        save_comparison_stats(stats_list, csv_path)
        
        print(f"\n=== Performance Comparison ===")
        print(f"Single: {single_stats['total_time_seconds']:.2f}s ({single_stats['images_per_second']:.2f} img/s)")
        print(f"Batch:  {batch_stats['total_time_seconds']:.2f}s ({batch_stats['images_per_second']:.2f} img/s)")
        speedup = single_stats['total_time_seconds'] / batch_stats['total_time_seconds'] if batch_stats['total_time_seconds'] > 0 else 0
        print(f"Speedup: {speedup:.2f}x faster with batch")
    
    elif args.batch:
        print(f"Using batch evaluation with batch size {args.batch_size}")
        
        evaluations = [
            {"image_input": row["url"], "text_prompt": row["caption"]}
            for _, row in df.iterrows()
        ]
        
        batch_result = evaluate_batch(args.service_url, evaluations, batch_size=args.batch_size)
        
        if "results" in batch_result:
            results = batch_result["results"]
            for i, result in enumerate(results):
                clip_score = result.get("clip_score")
                if clip_score is not None:
                    df.at[i, "clip_score"] = clip_score
                    print(f"Row {i+1}: {clip_score}")
                else:
                    error = result.get("error", "Unknown error")
                    print(f"Row {i+1}: Error - {error}")
        else:
            print(f"Batch evaluation failed: {batch_result.get('error', 'Unknown error')}")
    
    else:
        print("Using single evaluation")
        
        for i, row in df.iterrows():
            url = row["url"]
            caption = row["caption"]
            
            print(f"\nRow {i+1}/{len(df)}")
            print(f"Caption: {caption[:50]}...")
            
            result = evaluate_single(args.service_url, url, caption)
            
            clip_score = result.get("clip_score")
            error = result.get("error")
            
            if clip_score is not None and error is None:
                print(f"CLIP Score: {clip_score}")
                df.at[i, "clip_score"] = clip_score
            else:
                print(f"Error: {error or 'Unknown error'}")
                df.at[i, "clip_score"] = None
    
    output_path = csv_path.parent / f"{csv_path.stem}_with_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    successful = df["clip_score"].notna().sum()
    print(f"Successfully processed {successful}/{len(df)} images")


if __name__ == "__main__":
    main()