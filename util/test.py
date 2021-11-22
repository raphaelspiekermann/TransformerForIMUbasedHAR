import urllib.request
from tqdm import tqdm
import zipfile
import logging

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, tmp_path=None, extract_archive=False):
    logging.info('Downloading from {}'.format(url))
    filename = tmp_path if extract_archive else output_path
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        tmp_path, _ = urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
    logging.info('File stored at {}'.format(filename))
    if extract_archive:
        logging.ingo('Extracting {} to {}'.format(tmp_path, output_path))
        with zipfile.ZipFile(tmp_path) as myZip:
            myZip.extractall(output_path)

def rename_dir(path, new_dir_name=''):
    if new_dir_name is not '':
        pass 

download_url('https://zenodo.org/record/3862782/files/IMU%20data.zip?download=1', 'D:/Data_Test', True)
