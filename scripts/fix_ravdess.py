# Update these based upon where the download from kaggle is and where you are writing the files
source_file = '/tmp/archive.zip'
destination_dir = '/tmp/fixed_ravdess'

import os
import zipfile 
import re
from pathlib import Path
import shutil

directory_to_extract_to = '/tmp/orig_ravdess'

with zipfile.ZipFile(source_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

for actor_dir in os.listdir(directory_to_extract_to):
    match = re.match(r'Actor_([0-9]+)', actor_dir)
    if match is None:
        continue
        
    # Create the destination directory
    new_actor_dir = os.path.join(destination_dir, f'Actor_{int(match[1]):03}')
    Path(new_actor_dir).mkdir(parents=True, exist_ok=True)

    # Now, just pad each field with an extra zero; this is needed to accommodate other datasets later
    for filename in os.listdir(os.path.join(directory_to_extract_to, actor_dir)):

        match = re.match(r'([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).wav', filename)

        if not match is None:
            new_filename = f'0{match[1]}-0{match[2]}-0{match[3]}-0{match[4]}-0{match[5]}-0{match[6]}-0{match[7]}.wav'
            shutil.copyfile(os.path.join(directory_to_extract_to, actor_dir, filename), os.path.join(new_actor_dir, new_filename))
        
