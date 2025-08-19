from datetime import datetime
import os

datasetpath = '/home/user/project/dataset/modified/'  # folder
savefile = '/home/user/project/verifyme.txt'
kmers = {}
k = 3  # change k as needed

def processdata(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read().replace('\n', '') 
            for i in range(len(content) - k + 1):
                substr = content[i:i+k]
                if substr in kmers:
                    kmers[substr] += 1
                else:
                    kmers[substr] = 1
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Recursively walk through the directory
for root, dirs, files in os.walk(datasetpath):
    for file in files:
        if file.endswith('.txt'):
            processdata(os.path.join(root, file))

try:
    with open(savefile, "a") as f:
        f.write(str(kmers) + "\n")
except Exception as e:
    print(f"Error saving to {savefile}: {e}")
