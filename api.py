import os
import sys
import time
import csv
import argparse
import requests
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent

DEFAULT_PROXIES = [
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_55830096_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_83071110_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_33680263_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_57401005_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_16243689_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_92778968_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_96357235_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_59456675_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_08918194_time_10:3300382@change4.owlproxy.com:7778",
    "http://Etl3DUG5aM20_custom_zone_BZ_st__city_sid_93647356_time_10:3300382@change4.owlproxy.com:7778"
]

PROXY_POOL = itertools.cycle(DEFAULT_PROXIES)
UA_GENERATOR = UserAgent()

class ThreadSafeProxyPool:
    def __init__(self, proxies: list[str]):
        self.proxies = list(proxies)
        self.index = 0
        self.lock = threading.Lock()

    def get_next(self) -> dict | None:
        if not self.proxies:
            return {}
        with self.lock:
            p_url = self.proxies[self.index]
            self.index = (self.index + 1) % len(self.proxies)
        return {
            "http": p_url,
            "https": p_url
        }

class BatchState:
    def __init__(self):
        self.lock = threading.Lock()
        self.success_count = 0
        self.consecutive_failures = 0
        self.abort_requested = False

    def report_success(self):
        with self.lock:
            self.success_count += 1
            self.consecutive_failures = 0

    def report_failure(self) -> bool:
        with self.lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.abort_requested = True
            return self.abort_requested

def get_masked_proxy_url(proxy_dict: dict | None) -> str:
    if not proxy_dict or "https" not in proxy_dict:
        return "Direct Connection"
    url = proxy_dict["https"]
    try:
        if "@" in url:
            auth, host_port = url.split("@", 1)
            scheme, credentials = auth.split("//", 1)
            user, password = credentials.split(":", 1)
            masked_user = user[-12:] if len(user) > 12 else user
            return f"{scheme}//...{masked_user}:***@{host_port}"
    except Exception:
        pass
    return url

def solve_captcha(image_path: str, proxies: dict | None = None) -> str | None:
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return None

    if proxies is None:
        proxy_url = next(PROXY_POOL)
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }

    # Generate random User-Agent
    try:
        user_agent = UA_GENERATOR.random
    except Exception:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    # Log connection details
    masked_p = get_masked_proxy_url(proxies)
    print(f"   -> Connection: {masked_p} | User-Agent: {user_agent}")

    url = "https://api.azapi.ai/t0001c"
    headers = {
        'Authorization': 'sand-6cbc83e7d66462c51b81611ec3f927ad5dfdb4ab0daca393b87ecc9e4cfc96b5',
        'Content-Type': 'image/png',  # Default content type
        'User-Agent': user_agent,
    }

    # Automatically set Content-Type based on file extension
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        headers['Content-Type'] = 'image/jpeg'
    elif image_path.lower().endswith('.png'):
        headers['Content-Type'] = 'image/png'

    try:
        with open(image_path, 'rb') as f:
            response = requests.post(url, headers=headers, data=f, proxies=proxies)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "Success":
            return result.get("output", {}).get("captcha")
        else:
            print(f"API failure response for {image_path}: {result.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {image_path}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response body: {e.response.text}")
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

_append_label_lock = threading.Lock()

def append_label(labels_path: str, filename: str, label: str):
    with _append_label_lock:
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
    parser.add_argument(
        "--proxy",
        type=str,
        default=",".join(DEFAULT_PROXIES),
        help="Comma-separated list of proxy URLs to rotate (set to empty string to disable)"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="Images",
        help="Path to the images directory for batch or default single mode"
    )
    parser.add_argument(
        "--bots",
        type=int,
        default=1,
        help="Number of parallel bots (worker threads) to use in batch mode"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process files starting from the end of the alphabetical list"
    )
    args = parser.parse_args()

    proxies_list = []
    if args.proxy:
        proxies_list = [p.strip() for p in args.proxy.split(",") if p.strip()]

    proxy_pool = ThreadSafeProxyPool(proxies_list)

    def get_next_proxy():
        if args.proxy == "":
            return {}
        return proxy_pool.get_next()

    # Batch Mode Execution
    if args.batch:
        image_dir = args.images
        if not os.path.exists(image_dir):
            print(f"Error: Images directory '{image_dir}' does not exist.")
            sys.exit(1)

        labeled_files = load_existing_labels(args.labels)
        print(f"Loaded {len(labeled_files)} existing labels from {args.labels}.")

        all_files = sorted(os.listdir(image_dir), reverse=args.reverse)
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

        print(f"Starting batch labeling for up to {len(to_process)} images with {args.bots} bots and a {args.delay}s delay...")

        state = BatchState()

        def worker_task(filename, task_index):
            if state.abort_requested:
                return

            image_path = os.path.join(image_dir, filename)
            current_proxy = get_next_proxy()
            
            # Print start message
            print(f"[{task_index}/{len(to_process)}] Processing {filename}...")
            
            captcha = solve_captcha(image_path, proxies=current_proxy)
            
            if state.abort_requested:
                return

            if captcha:
                append_label(args.labels, filename, captcha)
                state.report_success()
                print(f"[{task_index}/{len(to_process)}] Solved {filename} -> {captcha}")
            else:
                aborted = state.report_failure()
                print(f"[{task_index}/{len(to_process)}] Failed to solve {filename}")
                if aborted:
                    print("\nAborting: Encountered 3 consecutive API failures. Rate limit or quota might be exceeded.")

            # Add protective delay between requests (if configured)
            if args.delay > 0:
                time.sleep(args.delay)

        if args.bots > 1:
            with ThreadPoolExecutor(max_workers=args.bots) as executor:
                # Distribute tasks
                futures = [
                    executor.submit(worker_task, filename, i + 1)
                    for i, filename in enumerate(to_process)
                ]
                # Wait for all tasks to complete or check for abort
                for future in futures:
                    future.result()  # Propagates exceptions if any
        else:
            # Single-threaded fallback/default execution
            for i, filename in enumerate(to_process):
                if state.abort_requested:
                    break
                worker_task(filename, i + 1)

        print(f"\nBatch labeling finished. Successfully labeled {state.success_count} new images.")
        sys.exit(0)

    # Single Image Mode Execution
    image_path = args.image
    if not image_path:
        image_dir = args.images
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

    print(f"Processing {image_path}...")
    current_proxy = get_next_proxy()
    captcha = solve_captcha(image_path, proxies=current_proxy)
    if captcha:
        print(f"\n🎉 Solved CAPTCHA: {captcha}")
    else:
        print("\n❌ Failed to solve CAPTCHA.")

if __name__ == "__main__":
    main()