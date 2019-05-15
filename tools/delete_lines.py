import re

with open("a.txt", "r+") as f:
    p = re.compile(deleteid)
    lines = [line for line in f.readlines() if p.search(line) is None]
    f.seek(0)
    f.truncate(0)
    f.writelines(lines)
