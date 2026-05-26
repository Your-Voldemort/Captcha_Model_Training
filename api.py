import os
import sys
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

    print(f"Solving captcha for image: {image_path}...")
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(url, headers=headers, data=f)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "Success":
            return result.get("output", {}).get("captcha")
        else:
            print("API failure response:", result.get("message", "Unknown error"))
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Solve captcha using external API")
    parser.add_argument(
        "--image", 
        type=str, 
        default=None, 
        help="Path to the captcha image to solve"
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Fall back to first image in Images folder
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