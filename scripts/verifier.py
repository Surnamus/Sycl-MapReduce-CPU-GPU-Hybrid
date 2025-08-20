import re

# --- Load the dumped dictionary from text file ---
mydict = {}
with open("/home/user/project/verifyme.txt") as f:
    text = f.read()

# regex to extract all key-value pairs
pattern = re.compile(r"['\"]?([A-Z]+)['\"]?\s*:\s*(\d+)")
for match in pattern.findall(text):
    key, val = match
    mydict[key] = int(val)

# --- Load the file.txt into a dictionary ---
filedict = {}
with open("/home/user/project/output.txt") as f:
    for line in f:
        if ":" not in line:
            continue
        key, val = line.strip().split(":", 1)
        key = key.strip()
        val = int(val.strip())
        filedict[key] = val

# --- Compare keys regardless of order ---
all_keys = set(mydict.keys()) | set(filedict.keys())  # union of all keys

for key in all_keys:
    dict_val = mydict.get(key)
    file_val = filedict.get(key)

    if dict_val is None:
        print(f"Key {key} missing in verifyme.txt")
    elif file_val is None:
        print(f"Key {key} missing in output.txt")
    elif dict_val == file_val:
        print(f"Match {key}:{dict_val}")
    else:
        print(f"Mismatch {key} -> dict={dict_val}, file={file_val}")
