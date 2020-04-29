from pathlib import Path
import numpy as np
import subprocess
import os, sys


def getlines(verbose=True):
    lines = 0

    # open each file and count the lines
    # put the results in the dictionary: fnam_counts[number of lines] = filename
    fnam_counts = {}
    for ext in ['*.py', '*.cl', '*.rst', '*.ini']:
        for filename in Path('./').rglob('*.py'):
            with open(filename, 'r') as f:
                fnam_count = len(f.readlines())
                fnam_counts[fnam_count] = filename
            lines += fnam_count

    # print a sorted list of the results
    if verbose: 
        for n in np.sort([k for k in fnam_counts.keys()]):
            print('{0:6} {1}'.format(n, fnam_counts[n]))

        # show the total number of lines 
        print('{0:6} {1}'.format(lines, 'total'))
    return lines

# checkout the latest 
out = subprocess.run(['git', 'checkout', sys.argv[1]], stderr=subprocess.PIPE, stdout=subprocess.PIPE)

# get commit hashes
out = subprocess.run(['git', 'log', '--pretty=format:%h'], stdout=subprocess.PIPE)
hashes = str(out.stdout.decode("utf-8")).split('\n')

# checkout each commit and count lines
print('{0:6} {1}'.format('lines', 'git commit'))
lines = []
for com in hashes[::-1]:
    #os.system("git checkout " + com)
    out = subprocess.run(['git', 'checkout', com], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    n = getlines(False)
    lines.append(n)
    print('{0:6} {1}'.format(n, com))

out = subprocess.run(['git', 'checkout', sys.argv[1]], stderr=subprocess.PIPE, stdout=subprocess.PIPE)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = np.arange(len(lines))
y = np.array(lines)
ax.bar(x, y, width=0.8)
ax.set_title(sys.argv[1] + ' branch')
ax.set_xlabel('commit number')
ax.set_ylabel('number of lines')

plt.tight_layout()
plt.show()
