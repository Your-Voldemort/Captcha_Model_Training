import requests
import uuid
import os
from pathlib import Path

# Create the output folder (if it doesn't exist)
output_folder = Path("1")
output_folder.mkdir(exist_ok=True)

base_url = "https://gop.captcha.garena.com/image?key="
total_attempts = 10
success_count = 0

print(f"Starting generation of {total_attempts} images...")

for i in range(total_attempts):
    # Generate a random UUID
    new_key = str(uuid.uuid4())
    url = base_url + new_key

    try:
        response = requests.get(url, timeout=10)
        # Check if the image was returned (status 200 and valid content)
        if response.status_code == 200 and len(response.content) > 100:
            # Save the image as PNG (or detect format from headers)
            file_path = output_folder / f"{new_key}.png"
            with open(file_path, "wb") as f:
                f.write(response.content)
            success_count += 1
            print(f"[{i+1}/{total_attempts}] SUCCESS – saved {file_path.name}")
        else:
            print(f"[{i+1}/{total_attempts}] Failed (status {response.status_code})")
    except Exception as e:
        print(f"[{i+1}/{total_attempts}] Error: {e}")

print(f"\nDone. Successfully generated and saved {success_count} out of {total_attempts} images in folder '1'.")
