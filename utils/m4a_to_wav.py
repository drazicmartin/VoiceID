import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

N = 32  # number of parallel workers

def convert_to_wav(file_path):
    output_path = file_path.with_suffix('.wav')
    subprocess.run([
        'ffmpeg', '-loglevel', 'panic', '-y', '-i', str(file_path), '-ar', '16000', str(output_path)    
    ], check=True)
    file_path.unlink()  # delete original .m4a file
    

def main(folder_path = '.'):
    m4a_files = list(Path(folder_path).rglob('*.m4a'))
    print(f"{len(m4a_files)=}")
    with ThreadPoolExecutor(max_workers=N) as executor:
        futures = {executor.submit(convert_to_wav, f): f for f in m4a_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting files"):
            pass

if __name__ == '__main__':
    folder_path = r"I:\Dataset\aac"
    main(folder_path)
    
