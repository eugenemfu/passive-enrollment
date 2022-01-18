sdk_path = '../DSP_1.0/device-unlock-sdk-linux-v1.1.2-dev-phoenix-no-encryption-x64-gcc5.4-neh/'
init_data_path = '../DSP_1.0/device-unlock-sdk-linux-v1.1.2-dev-phoenix-no-encryption-x64-gcc5.4-neh/init_data/desktop/phoenix-1.1.2'

import sys
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(sdk_path + '/python/%i.%i' % (sys.version_info[0], sys.version_info[1]))

from voicesdk_iot import SessionManager, AudioType


WAV_LIST = 'data/vox2-test.txt'
OUTPUT = 'data/embeddings_labeled.pkl'


with open(WAV_LIST) as f:
	wav_paths, labels = zip(*[line.split() for line in f.readlines()])
print(len(wav_paths))

session_manager = SessionManager(init_data_path)

embeddings = []

for i in tqdm(range(len(wav_paths))):
    enroll_session = session_manager.create_enroll_template_session()
    enroll_session.add_entry(wav_paths[i], AudioType.COMMAND)
    embedding = enroll_session.complete().get_command_embeddings()['./model1/tfgraph.cfg@Identity'][0]
    #print(embedding)
    embeddings.append(embedding)
    
with open(OUTPUT, 'wb') as f:
    pickle.dump((embeddings, labels), f)
