import os
import subprocess
from collections import Counter

all_files = []
for root, dirs, files in os.walk('.'):
    # Skip .git, .venv, __pycache__, .idea, .pytest_cache, .vtcode internals
    if any(skip in root.split(os.sep) for skip in ['.git', '.venv', '__pycache__', '.idea', '.pytest_cache', '.vtcode']):
        continue
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, '.')
        rel = rel.replace('\\', '/')
        if not any(skip in rel for skip in ['.git/', '.venv/', '__pycache__/', '.idea/', '.pytest_cache/', '.vtcode/']):
            all_files.append(rel)

all_files.sort()
print(f'Total files (excl caches): {len(all_files)}')

# Check git tracked
result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, cwd='.')
git_files = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

# Count by extension
ext_counts = Counter()
for f in all_files:
    ext = os.path.splitext(f)[1].lower()
    ext_counts[ext] += 1

print('Extension counts:')
for ext, count in ext_counts.most_common():
    print(f'  {ext}: {count}')

# Git tracked
git_tracked = [f for f in all_files if f in git_files]
print(f'\nGit tracked: {len(git_tracked)}')
print(f'Untracked: {len(all_files) - len(git_tracked)}')

# Check for stale cache
stale_cache = [f for f in all_files if '__pycache__' in f and f.endswith('.pyc')]
print(f'\nStale .pyc in __pycache__: {len(stale_cache)}')
for f in stale_cache:
    print(f'  {f}')