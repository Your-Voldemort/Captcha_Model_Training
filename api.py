import os
import sys
import time
import csv
import argparse
import requests

def solve_captcha(image_path: str) -> str | None:
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return None

    url = "https://api.azapi.ai/t0001c"
    headers = {
        'Authorization': 'sand-da38b67e8e8ec926c0c5d7582928083507861e87b21962bae051997f125d6306',
        'Content-Type': 'image/png',  # Default content type
    }

    # Automatically set Content-Type based on file extension
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        headers['Content-Type'] = 'image/jpeg'
    elif image_path.lower().endswith('.png'):
        headers['Content-Type'] = 'image/png'

    try:
        with open(image_path, 'rb') as f:
            response = requests.post(url, headers=headers, data=f)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "Success":
            return result.get("output", {}).get("captcha")
        else:
            print(f"API failure response for {image_path}: {result.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {image_path}: {e}")
        return None

def load_existing_labels(labels_path: str) -> set[str]:
    if not os.path.exists(labels_path):
        return set()
    labeled = set()
    try:
        with open(labels_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename")
                if filename:
                    labeled.add(filename)
    except Exception as e:
        print(f"Warning reading labels: {e}")
    return labeled

def append_label(labels_path: str, filename: str, label: str):
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    file_exists = os.path.exists(labels_path)
    with open(labels_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(labels_path) == 0:
            writer.writerow(["filename", "label"])
        writer.writerow([filename, label])

def main():
    parser = argparse.ArgumentParser(description="Solve captcha using external API")
    parser.add_argument(
        "--image", 
        type=str, 
        default=None, 
        help="Path to a single captcha image to solve"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode to label multiple images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of images to label in this batch run"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API requests to avoid rate limits"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels.csv",
        help="Path to the labels CSV file"
    )
    args = parser.parse_args()

    # Batch Mode Execution
    if args.batch:
        image_dir = "Images"
        if not os.path.exists(image_dir):
            print(f"Error: Images directory '{image_dir}' does not exist.")
            sys.exit(1)

        labeled_files = load_existing_labels(args.labels)
        print(f"Loaded {len(labeled_files)} existing labels from {args.labels}.")

        all_files = sorted(os.listdir(image_dir))
        unlabeled_files = [
            f for f in all_files 
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f not in labeled_files
        ]

        total_unlabeled = len(unlabeled_files)
        print(f"Found {total_unlabeled} unlabeled images in '{image_dir}'.")

        to_process = unlabeled_files[:args.limit]
        if not to_process:
            print("No unlabeled images left to process.")
            sys.exit(0)

        print(f"Starting batch labeling for up to {len(to_process)} images with a {args.delay}s delay...")

        consecutive_failures = 0
        success_count = 0

        for i, filename in enumerate(to_process):
            image_path = os.path.join(image_dir, filename)
            
            # Solve Captcha
            captcha = solve_captcha(image_path)
            
            if captcha:
                append_label(args.labels, filename, captcha)
                success_count += 1
                consecutive_failures = 0
                print(f"[{i+1}/{len(to_process)}] Solved {filename} -> {captcha}")
            else:
                consecutive_failures += 1
                print(f"[{i+1}/{len(to_process)}] Failed to solve {filename}")

            if consecutive_failures >= 3:
                print("\nAborting: Encountered 3 consecutive API failures. Rate limit or quota might be exceeded.")
                break

            # Add protective delay between requests (except on the last request)
            if i < len(to_process) - 1 and args.delay > 0:
                time.sleep(args.delay)

        print(f"\nBatch labeling finished. Successfully labeled {success_count} new images.")
        sys.exit(0)

    # Single Image Mode Execution
    image_path = args.image
    if not image_path:
        image_dir = "Images"
        if os.path.exists(image_dir):
            images = sorted(os.listdir(image_dir))
            valid_images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if valid_images:
                image_path = os.path.join(image_dir, valid_images[0])
                print(f"No --image argument provided. Defaulting to: {image_path}")
            else:
                print(f"Error: No PNG or JPEG images found in '{image_dir}' directory.")
                sys.exit(1)
        else:
            print("Error: Please specify an image path using --image.")
            sys.exit(1)

    captcha = solve_captcha(image_path)
    if captcha:
        print(f"\n🎉 Solved CAPTCHA: {captcha}")
    else:
        print("\n❌ Failed to solve CAPTCHA.")

if __name__ == "__main__":
    main()