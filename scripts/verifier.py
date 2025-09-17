import re

mydict = {}
with open("/home/user/project/verifyme.txt") as f:
    text = f.read()

pattern = re.compile(r"['\"]?([A-Z]+)['\"]?\s*:\s*(\d+)")
for match in pattern.findall(text):
    key, val = match
    mydict[key] = int(val)

filedict = {}
with open("/home/user/project/output.txt") as f:
    for line in f:
        if ":" not in line:
            continue
        key, val = line.strip().split(":", 1)
        key = key.strip()
        val = int(val.strip())
        filedict[key] = val

all_keys = set(mydict.keys()) | set(filedict.keys())  # union of all keys

for key in all_keys:
    dict_val = mydict.get(key)
    file_val = filedict.get(key)

    if dict_val is None:
        print(f"Key {key} missing in verifyme.txt")
    elif file_val is None:
        print(f"Key {key} missing in output.txt")
    elif dict_val == file_val:
        #print(f"Match {key}:{dict_val}")
        pass
    else:
        print(f"Mismatch {key} -> dict={dict_val}, file={file_val}")
