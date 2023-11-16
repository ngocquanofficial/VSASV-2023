import pickle
import os
import sys 
sys.path.append(os.getcwd()) # NOQA
# Specify the path to your pickle file
file_path = 'catboost_info/stft_eer.pk'

# Load the pickle file
with open(file_path, 'rb') as file:
    cqt_data = pickle.load(file)
print(cqt_data)
print(cqt_data[71])
min_eer = min(cqt_data)
for i in range(100) :
    if cqt_data[i] == min_eer :
        print("MIN STFT IDX", i)
# Now, 'loaded_data' contains the data from the pickle file
