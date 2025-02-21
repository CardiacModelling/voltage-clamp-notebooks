#!/usr/bin/env python3
#
# Copy in updated files
#
import os
import shutil

origin = '/home/michael/dat/patch/20250214-filter-utrecht/data/'
target = os.path.dirname(__file__)
print(target)

ok = True
todo = []
for fname in os.listdir(target):
    base, ext = os.path.splitext(fname)
    if ext != '.zip':
        continue
    source = os.path.join(origin, fname)
    if os.path.isfile(source):
        todo.append(source)
    else:
        print(f'Source file not found: {source}')
        ok = False

if ok:
    if len(todo) == 0:
        print('Nothing to do')
    else:
        for source in todo:
            shutil.copy2(source, target)
        print(f'Copied {len(todo)} files.')
else:
    print('Operation aborted: copied 0 files.')

