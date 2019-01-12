import numpy as np

file = open('attributes_per_class.txt')
l = file.readlines()
f2 = []
f1 = []
for r in l:
    r = r.replace('\n', '')
    r = r.split('\t')
    f2.append(r[0])
    for rr in r[1:]:
        f1.append(float(rr))

f2 = np.array(f2)
f1 = np.array(f1)
print(f2)
print(f1)

