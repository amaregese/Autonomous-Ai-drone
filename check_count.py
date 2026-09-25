import os

all_files = []
for root, dirs, files in os.walk('.'):
    # Skip virtual env, cache, IDE
    if any(skip in root for skip in ['.venv', '__pycache__', '.idea', '.pytest_cache', '.vtcode', '.git']):
        continue
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, '.')
        all_files.append(rel)

# Sort and print
for f in sorted(all_files):
    print(f)
print(f'\nTotal: {len(all_files)} files')