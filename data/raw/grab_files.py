import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
import time

# 1. Cấu hình đường dẫn gốc
BASE_URL = "http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/CSV/"
DEST_ROOT = "./data/raw"

# 2. Danh sách các thư mục con 
TARGET_DIRS = [
    "Backdoor_Malware/",
    "Benign_Final/",
    "BrowserHijacking/",
    "CommandInjection/",
    "DDoS-ACK_Fragmentation/",
    "DDoS-HTTP_Flood/",
    "DDoS-ICMP_Flood/",
    "DDoS-ICMP_Fragmentation/",
    "DDoS-PSHACK_FLOOD/",
    "DDoS-RSTFINFLOOD/",
    "DDoS-SYN_Flood/",
    "DDoS-SlowLoris/",
    "DDoS-SynonymousIP_Flood/",
    "DDoS-TCP_Flood/",
    "DDoS-UDP_Flood/",
    "DDoS-UDP_Fragmentation/",
    "DNS_Spoofing/",
    "DictionaryBruteForce/",
    "DoS-HTTP_Flood/",
    "DoS-SYN_Flood/",
    "DoS-TCP_Flood/",
    "DoS-UDP_Flood/",
    "MITM-ArpSpoofing/",
    "Mirai-greeth_flood/",
    "Mirai-greip_flood/",
    "Mirai-udpplain/",
    "Recon-HostDiscovery/",
    "Recon-OSScan/",
    "Recon-PingSweep/",
    "Recon-PortScan/",
    "SqlInjection/",
    "Uploading_Attack/",
    "VulnerabilityScan/",
    "XSS/",
]

def get_soup(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Timeout 30s để tránh treo nếu mạng lag
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 404:
            print(f" [!] Cannot find (404): {url}")
            return None
            
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f" [!] Error connecting {url}: {e}")
        return None

def download_file(file_url, local_folder):
    # Tạo tên file từ URL
    file_name = unquote(file_url.split('/')[-1])
    save_path = os.path.join(local_folder, file_name)

    # Kiểm tra nếu file đã tải rồi
    if os.path.exists(save_path):
        if os.path.getsize(save_path) > 0:
            print(f"   [Skip] Already has: {file_name}")
            return
    
    print(f"   -> Downloading: {file_name}")
    try:
        # Stream=True cực quan trọng để không tràn RAM khi tải file lớn
        with requests.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"   -> [Error] Cannot download {file_name}: {e}")

def process_directory(sub_dir):
    # Tạo URL đầy đủ: BASE + SUB_DIR
    target_url = urljoin(BASE_URL, sub_dir)
    
    # Tạo thư mục lưu trữ cục bộ tương ứng (Ví dụ: CICIoT2023/DDoS-SlowLoris)
    # Việc này giúp dữ liệu gọn gàng hơn là ném hết vào 1 chỗ
    local_dir = os.path.join(DEST_ROOT, sub_dir.replace("/", ""))
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    print(f"\n==================================================")
    print(f"Processing directory: {sub_dir}")
    print(f"URL: {target_url}")
    print(f"Storing at: {local_dir}")
    print(f"==================================================")

    soup = get_soup(target_url)
    if not soup: return

    links = soup.find_all('a')
    csv_count = 0

    for link in links:
        href = link.get('href')
        if not href: continue

        # Chỉ tải file .csv
        if href.lower().endswith('.csv'):
            full_file_url = urljoin(target_url, href)
            download_file(full_file_url, local_dir)
            csv_count += 1
            
    if csv_count == 0:
        print(" -> Warning: Cannot find any .csv files.")

if __name__ == "__main__":
    print(f"Starting downloading dataset CIC IoT 2023...")
    print(f"Sum of directory needing to scan: {len(TARGET_DIRS)}")
    
    for sub_dir in TARGET_DIRS:
        process_directory(sub_dir)
        time.sleep(1)

    print("\nFinished!")