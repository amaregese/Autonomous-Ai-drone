import os
import csv
import subprocess
from collections import Counter

def classify_file(path):
    path = path.replace('\\', '/')
    
    if path.startswith('modules/'):
        if path.startswith('modules/distance_estimator/'):
            if os.path.basename(path) in ['report.py', 'analysis.py', 'dataset.py', 'evaluation.py']:
                return ('D', 'development', 'cross-platform', 'dev-only', 'Only imported by tools/tests, not in __init__.py exports')
            else:
                return ('A', 'production', 'cross-platform', 'core', 'Used by main, estimator, person_follow')
        if path.startswith('modules/yolo11_detector/') or path.startswith('modules/detector_yolo11'):
            return ('A', 'production', 'cross-platform', 'core', 'Core YOLO detection/tracking')
        if path == 'modules/person_follow.py':
            return ('A', 'production', 'cross-platform', 'core', 'PersonFollowController used by main')
        if path == 'modules/auto_calibrate.py':
            return ('A', 'production', 'cross-platform', 'core', 'AutoCalibrator used by main')
        if path == 'modules/tracking.py':
            return ('A', 'production', 'cross-platform', 'core', 'TrackingSession used by main')
        if path.startswith('modules/control_system/'):
            if 'visualizer' in path:
                if 'visualizer_ui' in path or path == 'modules/drone_visualizer.py':
                    if 'visualizer_ui' in path:
                        return ('D', 'development', 'cross-platform', 'debug', 'Debug visualizer implementation')
                    else:
                        return ('A', 'production', 'cross-platform', 'wrapper', 'Wrapper for visualizer_ui')
                else:
                    return ('A', 'production', 'cross-platform', 'core', 'PID control interface')
        if path == 'modules/navigation.py':
            return ('G', 'legacy', 'cross-platform', 'none', 'Only used by evaluation.py (tool) and benchmark tool')
        if path == 'modules/vision.py':
            return ('G', 'legacy', 'cross-platform', 'none', 'Legacy vision module, replaced by distance_estimator/vision.py')
        if path.startswith('modules/vision_utils/'):
            return ('G', 'legacy', 'cross-platform', 'none', 'Only used by legacy vision/navigation chain')
        return ('A', 'production', 'cross-platform', 'core', 'Core module')
    
    if path.startswith('jetson/'):
        return ('F', 'platform-specific', 'jetson/linux', 'deployment', 'Jetson-specific communication/streaming')
    
    if path.startswith('shared/'):
        return ('A', 'production', 'cross-platform', 'core', 'Shared communication/transport')
    
    if path.startswith('tests/'):
        return ('C', 'test', 'cross-platform', 'test-only', 'Test file')
    
    if path.startswith('tools/'):
        return ('D', 'development', 'cross-platform', 'dev-only', 'Development tool')
    
    if path.startswith('benchmarks/'):
        return ('D', 'benchmark-data', 'cross-platform', 'benchmark-only', 'Benchmark data/documentation')
    
    if path in ['requirements.txt', '.gitignore']:
        return ('E', 'config', 'cross-platform', 'deployment', 'Deployment configuration')
    
    if path == 'benchmarks/distance/manifest.json':
        return ('B', 'config', 'cross-platform', 'config', 'Camera intrinsics config')
    
    if path == 'YOLO/yolo11n.pt':
        return ('A', 'model', 'cross-platform', 'core', 'Production model weights')
    
    if path == 'autonomous_drone_main.py':
        return ('A', 'production', 'cross-platform', 'entry-point', 'Main production entry point')
    
    if path.endswith('.md'):
        if any(x in path.upper() for x in ['AUDIT', 'RECONCILIATION', 'VERIFICATION', 'CLEANUP', 'REPORT']):
            return ('D', 'audit-report', 'cross-platform', 'audit', 'Audit report generated during verification')
        if path in ['DISTANCE_ESTIMATOR.md', 'PROJECT_REPORT.md', 'SGC_TRACKER_OVERLAY_SPEC.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md']:
            return ('D', 'documentation', 'cross-platform', 'documentation', 'Technical documentation')
        return ('D', 'documentation', 'cross-platform', 'documentation', 'Documentation')
    
    if path.startswith('.idea/') or path.startswith('.vtcode/'):
        return ('F', 'ide-config', 'windows', 'ide-config', 'IDE configuration')
    
    if path.startswith('.pytest_cache/') or path.startswith('__pycache__/'):
        return ('H', 'cache', 'cross-platform', 'none', 'Cache file')
    
    if path in ['.gitignore', 'requirements.txt']:
        return ('E', 'config', 'cross-platform', 'deployment', 'Deployment configuration')
    
    if path == 'benchmarks/distance/manifest.json':
        return ('B', 'config', 'cross-platform', 'config', 'Camera intrinsics config')
    
    if path == 'YOLO/yolo11n.pt':
        return ('A', 'model', 'cross-platform', 'core', 'Production model weights')
    
    if path == 'autonomous_drone_main.py':
        return ('A', 'production', 'cross-platform', 'entry-point', 'Main production entry point')
    
    if path.endswith('.pyc'):
        return ('H', 'cache', 'cross-platform', 'none', 'Compiled bytecode')
    
    if path in ['.gitignore', 'requirements.txt']:
        return ('E', 'config', 'cross-platform', 'deployment', 'Configuration file')
    
    audit_scripts = ['check_count.py', 'check_duplicates.py', 'verify_counts.py', 'generate_inventory.py', 'build_final_inventory.py', 'count_files.py']
    if os.path.basename(path) in audit_scripts:
        return ('D', 'audit-script', 'cross-platform', 'audit-tool', 'Audit script generated during verification')
    
    generated_audits = ['FILE_AUDIT_RECONCILIATION.md', 'FILE_COUNT_RECONCILIATION.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md', 'PROJECT_CLEANUP_PLAN.md', 'PROJECT_DEPENDENCY_MAP.md', 'PROJECT_FILE_USAGE_AUDIT.md', 'PROJECT_FILE_USAGE_AUDIT_V2.md', 'PROJECT_REPORT.md', 'SAFE_CLEANUP_CANDIDATES.md', 'SAFE_CLEANUP_CANDIDATES_V2.md', 'UNUSED_FILES_CANDIDATES.md', 'YOLO_IMPLEMENTATION_AUDIT.md', 'DISTANCE_ESTIMATOR_FILE_VERIFICATION.md', 'FINAL_FILE_CLEANUP_RISK_REPORT.md', 'FINAL_JETSON_DEPLOYMENT_FILE_MAP.md', 'FINAL_LEGACY_VERIFICATION.md', 'FINAL_REPOSITORY_STATUS.md', 'FINAL_UNIQUE_FILE_INVENTORY.md', 'FINAL_YOLO_IMPLEMENTATION_AUDIT.md', 'VISUALIZER_DEPENDENCY_VERIFICATION.md', 'YOLO_IMPLEMENTATION_AUDIT.md', 'DISTANCE_ESTIMATOR_FILE_VERIFICATION.md', 'INVENTORY_NONFILE_CORRECTIONS.md', 'AUTHORITATIVE_GIT_STATUS.md', 'FINAL_FILE_COUNT_REPORT.md', 'SG_CONFIG_VERIFICATION.md', 'AUTHORITATIVE_FILE_INVENTORY.md', 'AUTHORITATIVE_FILE_INVENTORY.csv']
    if os.path.basename(path) in generated_audits:
        return ('D', 'audit-output', 'cross-platform', 'audit-output', 'Generated during verification')
    
    audit_scripts = ['check_count.py', 'check_duplicates.py', 'verify_counts.py', 'generate_inventory.py', 'build_final_inventory.py', 'count_files.py']
    if os.path.basename(path) in audit_scripts:
        return ('D', 'audit-script', 'cross-platform', 'audit-tool', 'Audit script generated during verification')
    
    if path == 'benchmarks/camera_calibration/checkerboard_9x6.png':
        return ('D', 'calibration-target', 'cross-platform', 'calibration-target', 'Calibration target image')
    
    return ('I', 'unknown', 'cross-platform', 'unknown', 'Unclassified')


# Get all files (excluding caches)
all_files = []
for root, dirs, files in os.walk('.'):
    if any(skip in root.split(os.sep) for skip in ['.git', '.venv', '__pycache__', '.idea', '.pytest_cache', '.vtcode']):
        continue
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, '.')
        rel = rel.replace('\\', '/')
        if not any(skip in rel for skip in ['.git/', '.venv/', '__pycache__/', '.idea/', '.pytest_cache/', '.vtcode/']):
            all_files.append(rel)

all_files.sort()

# Get git tracked
import subprocess
result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, cwd='.')
git_files = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

inventory = []
for rel in all_files:
    exists = os.path.exists(rel)
    git_tracked = rel in git_files
    classification, category, platform, prod_rel, evidence = classify_file(rel)
    
    inventory.append({
        'path': rel,
        'exists': 'YES',
        'git_tracked': 'YES' if rel in git_files else 'NO',
        'category': 'development' if classification in ['C', 'D'] else ('production' if classification in ['A', 'B'] else ('legacy' if classification == 'G' else ('cache' if classification == 'H' else 'unknown'))),
        'classification': classification,
        'platform': 'jetson/linux' if rel.startswith('jetson/') else ('windows' if rel.startswith('.idea/') or rel.startswith('.vtcode/') else 'cross-platform'),
        'production_relevance': 'core' if classification in ['A', 'B'] else ('dev-only' if classification == 'D' else ('test-only' if classification == 'C' else ('none' if classification in ['G', 'H'] else 'unknown'))),
        'evidence': 'TBD'
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

print(f'Total files: {len(all_files)}')
print('CSV and MD written.')

# Verify uniqueness
paths = [item['path'] for item in inventory]
unique_paths = set(paths)
print(f'Unique paths: {len(unique_paths)}')
print(f'Total rows: {len(paths)}')
print(f'Duplicates: {len(paths) - len(unique_paths)}')

# Classification counts
counts = Counter([item['classification'] for item in inventory])
print('\nClassification counts:')
for cls in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
    print(f'  {cls}: {counts.get(cls, 0)}')
print(f'  Total: {sum(counts.values())}')