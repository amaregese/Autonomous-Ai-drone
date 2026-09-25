import re

with open('PROJECT_FILE_USAGE_AUDIT_V2.md', 'r') as f:
    content = f.read()

# Find all file paths in the table (lines with backticks)
paths = re.findall(r'`([^`]+)`', content)

# Count occurrences
from collections import Counter
counts = Counter(paths)

# Find duplicates
duplicates = {path: count for path, count in counts.items() if count > 1}

print('Duplicate entries in V2 audit:')
for path, count in sorted(duplicates.items()):
    print(f'  {path}: {count} occurrences')

print(f'\nTotal unique paths in audit: {len(set(paths))}')
print(f'Total path occurrences in audit: {len(paths)}')