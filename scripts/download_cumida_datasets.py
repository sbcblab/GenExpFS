import os
import requests
from urllib.parse import urlparse

from tqdm import tqdm


DATASETS_PATH = 'datasets/cumida'

DATASETS_URLS = [
    'http://sbcb.inf.ufrgs.br/carbm/static/cumida/Genes/Liver/GSE22405/Liver_GSE22405.csv',
    'http://sbcb.inf.ufrgs.br/carbm/static/cumida/Genes/Prostate/GSE6919_U95C/Prostate_GSE6919_U95C.csv',
    'http://sbcb.inf.ufrgs.br/carbm/static/cumida/Genes/Breast/GSE70947/Breast_GSE70947.csv',
    'http://sbcb.inf.ufrgs.br/carbm/static/cumida/Genes/Renal/GSE53757/Renal_GSE53757.csv',
    'http://sbcb.inf.ufrgs.br/carbm/static/cumida/Genes/Colorectal/GSE44861/Colorectal_GSE44861.csv'
]

if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)

for url in DATASETS_URLS:
    res = requests.get(url, stream=True)

    file_name = os.path.basename(urlparse(url).path)
    file_path = os.path.join(DATASETS_PATH, file_name)

    content_length = int(res.headers.get('content-length', 0))
    chunk_size = 1024
    with tqdm(total=content_length, unit='iB', unit_scale=True) as progress_bar:
        with open(file_path, 'wb') as f:
            for data in res.iter_content(chunk_size):
                progress_bar.update(len(data))
                f.write(data)

    print(f"Saved dataset to {file_path}")
