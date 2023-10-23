# Update these based upon where the download from kaggle is and where you are writing the files
source_file = '/tmp/archive.zip'
destination_dir = '/tmp/fixed_tess'

actor_names_to_ids = {'OAF': 26, 'YAF': 28}
emotions_to_ids = {'angry': 5, 'fear': 6, 'happy': 3, 'neutral': 1, 'disgust': 7}

import os
import sys
import zipfile 
import re
from pathlib import Path
import shutil

dir_for_this_script = os.path.split(__file__)[0]
if not dir_for_this_script in sys.path:
    sys.path.append(dir_for_this_script)

# Import this b/c it has all of the tess sentence mappings
import transcription

directory_to_extract_to = '/tmp/orig_tess'

with zipfile.ZipFile(source_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

speaker_dirs = os.listdir(os.path.join(directory_to_extract_to, 'TESS Toronto emotional speech set data'))

for directory in speaker_dirs:
    match = re.match(r'(.*)_(.*)', directory)
    if not match is None and match[1] in actor_names_to_ids and match[2] in emotions_to_ids:
        actor_id = actor_names_to_ids[match[1]]
        emo_id = emotions_to_ids[match[2]]
        
        for audio_file in os.listdir(os.path.join(directory_to_extract_to, 'TESS Toronto emotional speech set data', directory)):
            match = re.match(r'.*_(.*)_.*', audio_file)
            if not match is None:
                phrase = f'say the word {match[1]}'
                if phrase in transcription.tess_sentences:
                    utterance_id = transcription.tess_sentences[phrase]
                    
                    # Copy the file
                    new_actor_dir = os.path.join(destination_dir, f'Actor_{actor_id:03}')
                    Path(new_actor_dir).mkdir(parents=True, exist_ok=True)
                    new_filename = f'003-001-{emo_id:03}-001-{utterance_id:03}-001-{actor_id:03}.wav'

                    shutil.copyfile(os.path.join(directory_to_extract_to, 'TESS Toronto emotional speech set data', directory, audio_file), 
                                    os.path.join(new_actor_dir, new_filename))

                    
