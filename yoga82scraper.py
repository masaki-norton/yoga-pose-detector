import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def setup_session():
    """Set up a requests session with retry logic and custom headers."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 404, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    })
    return session

def download_image(args):
    """Download an image from a URL and save it to the specified path. Log failed downloads."""
    image_url, save_path, failed_log_path, session = args
    try:
        response = session.get(image_url, timeout=10)  # 10 seconds timeout
        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"HTTP error {response.status_code}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        failed_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_log_path, 'a') as failed_file:
            failed_file.write(f"{save_path.name}\t{image_url}\n")

def process_txt_files(folder_path):
    """Process all .txt files in the specified folder and handle failed downloads using parallel execution."""
    tasks = []
    session = setup_session()  # Set up a session for reuse
    for txt_file in Path(folder_path).glob('*.txt'):
        failed_log_path = Path(folder_path).parent / "failed_downloads" / (txt_file.stem + "_failed.txt")
        with open(txt_file, 'r') as file:
            for line in file:
                parts = line.split('\t')
                if len(parts) == 2:
                    image_url = parts[1].strip()
                    image_name_with_path = parts[0].replace('/', '_')
                    save_dir = Path(folder_path).parent / "images" / txt_file.stem
                    save_path = save_dir / image_name_with_path
                    tasks.append((image_url, save_path, failed_log_path, session))

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_image, tasks)

folder_path = Path('New Pose Data/Yoga-82/yoga_dataset_links')
process_txt_files(folder_path)
