import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.ncei.noaa.gov/data/oceans/argo/gdac/incois/"
SAVE_DIR = "argo_prof_files"

os.makedirs(SAVE_DIR, exist_ok=True)

def get_links(url):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)]
    return links

# Step 1: Get float directories (only numeric folders like "1902669/")
float_dirs = [link for link in get_links(BASE_URL) if link.endswith("/") and link[0].isdigit()]

print(f"Found {len(float_dirs)} float directories.")

# Step 2: Loop through float directories and fetch *_prof.nc
for fdir in float_dirs:
    float_url = BASE_URL + fdir
    files = get_links(float_url)

    # Only grab files ending with "_prof.nc"
    prof_files = [file for file in files if file.endswith("_prof.nc")]

    for pfile in prof_files:
        file_url = float_url + pfile
        save_path = os.path.join(SAVE_DIR, pfile)

        if not os.path.exists(save_path):  # avoid re-downloading
            print(f"Downloading {file_url}")
            r = requests.get(file_url, stream=True)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)