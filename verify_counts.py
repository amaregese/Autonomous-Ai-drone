import os
import re

# Count unique files in filesystem (excluding .git, .venv, __pycache__, .idea, .pytest_cache, .vtcode)
all_files = []
for root, dirs, files in os.walk('.'):
    if any(skip in root for skip in ['.git', '.venv', '__pycache__', '.idea', '.pytest_cache', '.vtcode']):
        continue
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, '.')
        all_files.append(rel)

print(f'Total files in filesystem (excl .git, .venv, caches): {len(all_files)}')

# Count unique paths in V2 audit table
with open('PROJECT_FILE_USAGE_AUDIT_V2.md', 'r') as f:
    content = f.read()
paths = re.findall(r'`([^`]+)`', content)
unique_audit = set(paths)
print(f'Unique paths in V2 audit table: {len(unique_audit)}')
print(f'Total occurrences in audit: {len(paths)}')

# Check which audit paths don't exist in filesystem
missing = []
for p in unique_audit:
    if not os.path.exists(p):
        missing.append(p)
        
print(f'Paths in audit but not in filesystem: {len(missing)}')
for m in sorted(missing):
    print(f'  MISSING: {m}')

# Check which filesystem files are missing from audit
audit_paths = set(p for p in unique_audit if not p.startswith('.'))
fs_paths = set(all_files)
missing_from_audit = fs_paths - audit_paths
print(f'Files in filesystem but missing from audit: {len(missing_from_audit)}')
for m in sorted(missing_from_audit):
    print(f'  MISSING FROM AUDIT: {m}')

# Also check duplicate entries
from collections import Counter
counts = Counter(paths)
duplicates = {path: count for path, count in counts.items() if count > 1}
print(f'\nDuplicate entries in V2 audit:')
for path, count in sorted(duplicates.items()):
    print(f'  {path}: {count} occurrences')