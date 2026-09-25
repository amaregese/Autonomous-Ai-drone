import os
import csv
import subprocess

all_files = []
for root, dirs, files in os.walk('.'):
    # Skip .git internals only
    if '.git' in root.split(os.sep):
        continue
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, '.')
        # Normalize path separators
        rel = rel.replace('\\', '/')
        all_files.append(rel)

# Sort deterministically
all_files.sort()

# Check git tracked
result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, cwd='.')
git_files = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

# Check each file
inventory = []
for rel in all_files:
    exists = os.path.exists(rel)
    git_tracked = rel in git_files
    
    # Determine category and classification
    ext = os.path.splitext(rel)[1].lower()
    
    # Determine classification based on path and previous analysis
    classification = 'I'  # default UNKNOWN
    category = 'unknown'
    platform = 'cross-platform'
    production_relevance = 'unknown'
    evidence = ''
    
    # Classification logic based on path
    if rel.startswith('modules/'):
        if rel.startswith('modules/distance_estimator/'):
            if os.path.basename(rel) in ['report.py', 'analysis.py', 'dataset.py', 'evaluation.py']:
                classification = 'D'
                category = 'development'
                production_relevance = 'dev-only'
                evidence = 'Only imported by tools/tests, not in __init__.py exports'
            else:
                classification = 'A'
                category = 'production'
                production_relevance = 'core'
                evidence = 'Used by main, estimator, person_follow'
        elif rel.startswith('modules/yolo11_detector/') or rel.startswith('modules/detector_yolo11'):
            classification = 'A'
            category = 'production'
            production_relevance = 'core'
            evidence = 'Core YOLO detection/tracking'
        elif rel == 'modules/person_follow.py':
            classification = 'A'
            category = 'production'
            production_relevance = 'core'
            evidence = 'PersonFollowController used by main'
        elif rel == 'modules/auto_calibrate.py':
            classification = 'A'
            category = 'production'
            production_relevance = 'core'
            evidence = 'AutoCalibrator used by main'
        elif rel == 'modules/tracking.py':
            classification = 'A'
            category = 'production'
            production_relevance = 'core'
            evidence = 'TrackingSession used by main'
        elif rel.startswith('modules/control_system/'):
            if 'visualizer' in rel:
                if 'visualizer_ui' in rel or rel == 'modules/drone_visualizer.py':
                    if 'visualizer_ui' in rel:
                        classification = 'D'
                        category = 'development'
                        production_relevance = 'debug'
                        evidence = 'Debug visualizer implementation'
                    else:
                        classification = 'A'
                        category = 'production'
                        production_relevance = 'wrapper'
                        evidence = 'Wrapper for visualizer_ui'
                else:
                    classification = 'A'
                    category = 'production'
                    production_relevance = 'core'
                    evidence = 'PID control interface'
            else:
                classification = 'A'
                category = 'production'
                production_relevance = 'core'
                evidence = 'Control system component'
        elif rel == 'modules/navigation.py':
            classification = 'G'
            category = 'legacy'
            production_relevance = 'none'
            evidence = 'Only used by evaluation.py (tool) and benchmark tool'
        elif rel == 'modules/vision.py':
            classification = 'G'
            category = 'legacy'
            production_relevance = 'none'
            evidence = 'Legacy vision module, replaced by distance_estimator/vision.py'
        elif rel.startswith('modules/vision_utils/'):
            classification = 'G'
            category = 'legacy'
            production_relevance = 'none'
            evidence = 'Only used by legacy vision/navigation chain'
        elif rel.startswith('modules/'):
            classification = 'A'
            category = 'production'
            production_relevance = 'core'
            evidence = 'Core module'
        elif rel.startswith('jetson/'):
            classification = 'F'
            category = 'platform-specific'
            platform = 'jetson/linux'
            production_relevance = 'deployment'
            evidence = 'Jetson-specific communication/streaming'
        elif rel.startswith('shared/'):
            classification = 'A'
            category = 'production'
            production_relevance = 'core'
            evidence = 'Shared communication/transport'
        elif rel.startswith('tests/'):
            classification = 'C'
            category = 'test'
            production_relevance = 'test-only'
            evidence = 'Test file'
        elif rel.startswith('tools/'):
            classification = 'D'
            category = 'development'
            production_relevance = 'dev-only'
            evidence = 'Development tool'
        elif rel.startswith('benchmarks/'):
            classification = 'D'
            category = 'benchmark-data'
            production_relevance = 'benchmark-only'
            evidence = 'Benchmark data/documentation'
        elif rel in ['requirements.txt', '.gitignore']:
            classification = 'E'
            category = 'deployment'
            production_relevance = 'deployment'
            evidence = 'Deployment configuration'
        elif rel == 'benchmarks/distance/manifest.json':
            classification = 'B'
            category = 'config'
            production_relevance = 'config'
            evidence = 'Camera intrinsics config'
        elif rel == 'YOLO/yolo11n.pt':
            classification = 'A'
            category = 'model'
            production_relevance = 'core'
            evidence = 'Production model weights'
        elif rel.endswith('.md'):
            if any(x in rel.upper() for x in ['AUDIT', 'RECONCILIATION', 'VERIFICATION', 'CLEANUP', 'REPORT']):
                classification = 'D'
                category = 'audit-report'
                production_relevance = 'audit'
                evidence = 'Audit report generated during verification'
            elif rel in ['DISTANCE_ESTIMATOR.md', 'PROJECT_REPORT.md', 'SGC_TRACKER_OVERLAY_SPEC.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md']:
                classification = 'D'
                category = 'documentation'
                production_relevance = 'documentation'
                evidence = 'Technical documentation'
            else:
                classification = 'D'
                category = 'documentation'
                production_relevance = 'documentation'
                evidence = 'Documentation'
        elif rel.startswith('.idea/') or rel.startswith('.vtcode/'):
            classification = 'F'
            category = 'ide-config'
            platform = 'windows'
            production_relevance = 'ide-config'
            evidence = 'IDE configuration'
        elif rel.startswith('.pytest_cache/') or rel.startswith('__pycache__/'):
            classification = 'H'
            category = 'cache'
            production_relevance = 'none'
            evidence = 'Cache file'
        elif rel in ['.gitignore', 'requirements.txt']:
            classification = 'E'
            category = 'config'
            production_relevance = 'deployment'
            evidence = 'Configuration file'
        elif rel.endswith('.pyc'):
            classification = 'H'
            category = 'cache'
            production_relevance = 'none'
            evidence = 'Compiled bytecode'
        else:
            classification = 'I'
            category = 'unknown'
            production_relevance = 'unknown'
            evidence = 'Unclassified'
    
    inventory.append({
        'path': rel,
        'exists': 'YES' if exists else 'NO',
        'git_tracked': 'YES' if git_tracked else 'NO',
        'category': category,
        'classification': classification,
        'platform': platform,
        'production_relevance': production_relevance,
        'evidence': evidence
    })

# Write CSV
with open('AUTHORITATIVE_FILE_INVENTORY.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'exists', 'git_tracked', 'category', 'classification', 'platform', 'production_relevance', 'evidence'])
    writer.writeheader()
    writer.writerows(inventory)

# Write Markdown
with open('AUTHORITATIVE_FILE_INVENTORY.md', 'w', encoding='utf-8') as f:
    f.write('# AUTHORITATIVE_FILE_INVENTORY.md\n\n')
    f.write('## Complete File Inventory\n\n')
    f.write('| path | exists | git_tracked | category | classification | platform | production_relevance | evidence |\n')
    f.write('|------|--------|-------------|----------|----------------|----------|---------------------|----------|\n')
    for item in inventory:
        f.write(f"| {item['path']} | {item['exists']} | {item['git_tracked']} | {item['category']} | {item['classification']} | {item['platform']} | {item['production_relevance']} | {item['evidence']} |\n")
    f.write(f'\nTotal files: {len(inventory)}\n')

print(f'Total files: {len(inventory)}')
print('CSV and MD written.')