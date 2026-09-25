import os
import subprocess
import csv
from collections import Counter

def classify_file(path):
    path = path.replace('\\', '/')
    
    # Specific file checks FIRST (before prefix checks)
    if path == 'benchmarks/distance/manifest.json':
        return ('B', 'config', 'cross-platform', 'config', 'Camera intrinsics config')
    if path == 'YOLO/yolo11n.pt':
        return ('A', 'model', 'cross-platform', 'core', 'Production model weights')
    if path == 'autonomous_drone_main.py':
        return ('A', 'production', 'cross-platform', 'entry-point', 'Main production entry point')
    if path == 'benchmarks/camera_calibration/checkerboard_9x6.png':
        return ('D', 'calibration-target', 'cross-platform', 'calibration-target', 'Calibration target image')
    
    # Cache files (check before shared/ and other prefix checks)
    if path.startswith('.pytest_cache/') or path.startswith('__pycache__/') or '/__pycache__/' in path:
        return ('H', 'cache', 'cross-platform', 'none', 'Cache file')
    if path.endswith('.pyc'):
        return ('H', 'cache', 'cross-platform', 'none', 'Compiled bytecode')
    
    # Specific legacy files FIRST (before general modules/ check)
    if path == 'modules/navigation.py':
        return ('G', 'legacy', 'cross-platform', 'none', 'Only used by evaluation.py (tool) and benchmark tool')
    if path == 'modules/vision.py':
        return ('G', 'legacy', 'cross-platform', 'none', 'Legacy vision module, replaced by distance_estimator/vision.py')
    if path == 'modules/vision_utils/geometry.py':
        return ('G', 'legacy', 'cross-platform', 'none', 'Only used by legacy vision/navigation chain')
    if path == 'modules/vision_utils/legacy.py':
        return ('G', 'legacy', 'cross-platform', 'none', 'Only used by legacy vision.py')
    if path == 'modules/vision_utils/__init__.py':
        return ('G', 'legacy', 'cross-platform', 'none', 'Legacy package init')
    
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
    
    if path.startswith('.pytest_cache/') or path.startswith('__pycache__/') or '/__pycache__/' in path:
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
        return ('E', 'config', 'cross-platform', 'deployment', 'Deployment configuration')
    
    # Audit scripts generated during verification
    audit_scripts = ['check_count.py', 'check_duplicates.py', 'verify_counts.py', 'generate_inventory.py', 'build_final_inventory.py', 'count_files.py']
    if os.path.basename(path) in audit_scripts:
        return ('D', 'audit-script', 'cross-platform', 'audit-tool', 'Audit script generated during verification')
    
    # Generated audit reports
    generated_audits = ['FILE_AUDIT_RECONCILIATION.md', 'FILE_COUNT_RECONCILIATION.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md', 'PROJECT_CLEANUP_PLAN.md', 'PROJECT_DEPENDENCY_MAP.md', 'PROJECT_FILE_USAGE_AUDIT.md', 'PROJECT_FILE_USAGE_AUDIT_V2.md', 'PROJECT_REPORT.md', 'SAFE_CLEANUP_CANDIDATES.md', 'SAFE_CLEANUP_CANDIDATES_V2.md', 'UNUSED_FILES_CANDIDATES.md', 'YOLO_IMPLEMENTATION_AUDIT.md', 'DISTANCE_ESTIMATOR_FILE_VERIFICATION.md', 'FINAL_FILE_CLEANUP_RISK_REPORT.md', 'FINAL_JETSON_DEPLOYMENT_FILE_MAP.md', 'FINAL_LEGACY_VERIFICATION.md', 'FINAL_REPOSITORY_STATUS.md', 'FINAL_UNIQUE_FILE_INVENTORY.md', 'FINAL_YOLO_IMPLEMENTATION_AUDIT.md', 'VISUALIZER_DEPENDENCY_VERIFICATION.md', 'YOLO_IMPLEMENTATION_AUDIT.md', 'DISTANCE_ESTIMATOR_FILE_VERIFICATION.md', 'INVENTORY_NONFILE_CORRECTIONS.md', 'AUTHORITATIVE_GIT_STATUS.md', 'FINAL_FILE_COUNT_REPORT.md', 'SG_CONFIG_VERIFICATION.md', 'AUTHORITATIVE_FILE_INVENTORY.md', 'AUTHORITATIVE_FILE_INVENTORY.csv']
    if os.path.basename(path) in generated_audits:
        return ('D', 'audit-output', 'cross-platform', 'audit-output', 'Generated during verification')
    
    # Audit scripts
    audit_scripts = ['check_count.py', 'check_duplicates.py', 'verify_counts.py', 'generate_inventory.py', 'build_final_inventory.py', 'count_files.py']
    if os.path.basename(path) in audit_scripts:
        return ('D', 'audit-script', 'cross-platform', 'audit-tool', 'Audit script generated during verification')
    
    # Other files
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
counts = {}
for rel in all_files:
    p = rel
    # Classify inline (same logic as classify_file but inline for performance)
    if p == 'benchmarks/distance/manifest.json': c = 'B'
    elif p == 'YOLO/yolo11n.pt': c = 'A'
    elif p == 'autonomous_drone_main.py': c = 'A'
    elif p == 'benchmarks/camera_calibration/checkerboard_9x6.png': c = 'D'
    elif p.startswith('.pytest_cache/') or p.startswith('__pycache__/') or '/__pycache__/' in p: c = 'H'
    elif p.endswith('.pyc'): c = 'H'
    elif p == 'modules/navigation.py': c = 'G'
    elif p == 'modules/vision.py': c = 'G'
    elif p == 'modules/vision_utils/geometry.py': c = 'G'
    elif p == 'modules/vision_utils/legacy.py': c = 'G'
    elif p == 'modules/vision_utils/__init__.py': c = 'G'
    elif p.startswith('modules/'):
        if p.startswith('modules/distance_estimator/'):
            if os.path.basename(p) in ['report.py', 'analysis.py', 'dataset.py', 'evaluation.py']: c = 'D'
            else: c = 'A'
        elif p.startswith('modules/yolo11_detector/') or p.startswith('modules/detector_yolo11'): c = 'A'
        elif p == 'modules/person_follow.py': c = 'A'
        elif p == 'modules/auto_calibrate.py': c = 'A'
        elif p == 'modules/tracking.py': c = 'A'
        elif p.startswith('modules/control_system/'):
            if 'visualizer' in p:
                if 'visualizer_ui' in p or p == 'modules/drone_visualizer.py':
                    if 'visualizer_ui' in p: c = 'D'
                    else: c = 'A'
                else: c = 'A'
            else: c = 'A'
    elif p.startswith('jetson/'): c = 'F'
    elif p.startswith('shared/'): c = 'A'
    elif p.startswith('tests/'): c = 'C'
    elif p.startswith('tools/'): c = 'D'
    elif p.startswith('benchmarks/'): c = 'D'
    elif p in ['requirements.txt', '.gitignore']: c = 'E'
    elif p == 'benchmarks/distance/manifest.json': c = 'B'
    elif p == 'YOLO/yolo11n.pt': c = 'A'
    elif p == 'autonomous_drone_main.py': c = 'A'
    elif p.endswith('.md'):
        if any(x in p.upper() for x in ['AUDIT', 'RECONCILIATION', 'VERIFICATION', 'CLEANUP', 'REPORT']): c = 'D'
        elif p in ['DISTANCE_ESTIMATOR.md', 'PROJECT_REPORT.md', 'SGC_TRACKER_OVERLAY_SPEC.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md']: c = 'D'
        else: c = 'D'
    elif p.startswith('.idea/') or p.startswith('.vtcode/'): c = 'F'
    elif p.startswith('.pytest_cache/') or p.startswith('__pycache__/') or '/__pycache__/' in p: c = 'H'
    elif p in ['.gitignore', 'requirements.txt']: c = 'E'
    elif p == 'benchmarks/distance/manifest.json': c = 'B'
    elif p == 'YOLO/yolo11n.pt': c = 'A'
    elif p == 'autonomous_drone_main.py': c = 'A'
    elif p.endswith('.pyc'): c = 'H'
    elif p in ['.gitignore', 'requirements.txt']: c = 'E'
    else: c = 'I'
    
    inventory.append({'path': rel, 'classification': c, 'git_tracked': 'YES' if rel in git_files else 'NO'})
    counts[c] = counts.get(c, 0) + 1

print('Classification counts:')
for cls in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
    print(f'  {cls}: {counts.get(cls, 0)}')
print(f'Total: {sum(counts.values())}')

# Write CSV
with open('AUTHORITATIVE_FILE_INVENTORY.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'classification', 'git_tracked'])
    for rel in all_files:
        p = rel
        # classify inline (same logic as above)
        p_norm = p.replace('\\', '/')
        if p_norm == 'benchmarks/distance/manifest.json': c = 'B'
        elif p_norm == 'YOLO/yolo11n.pt': c = 'A'
        elif p_norm == 'autonomous_drone_main.py': c = 'A'
        elif p_norm == 'benchmarks/camera_calibration/checkerboard_9x6.png': c = 'D'
        elif p_norm.startswith('.pytest_cache/') or p_norm.startswith('__pycache__/') or '/__pycache__/' in p_norm: c = 'H'
        elif p_norm.endswith('.pyc'): c = 'H'
        elif p_norm == 'modules/navigation.py': c = 'G'
        elif p_norm == 'modules/vision.py': c = 'G'
        elif p_norm == 'modules/vision_utils/geometry.py': c = 'G'
        elif p_norm == 'modules/vision_utils/legacy.py': c = 'G'
        elif p_norm == 'modules/vision_utils/__init__.py': c = 'G'
        elif p_norm.startswith('modules/'):
            if p_norm.startswith('modules/distance_estimator/'):
                if os.path.basename(p_norm) in ['report.py', 'analysis.py', 'dataset.py', 'evaluation.py']: c = 'D'
                else: c = 'A'
            elif p_norm.startswith('modules/yolo11_detector/') or p_norm.startswith('modules/detector_yolo11'): c = 'A'
            elif p_norm == 'modules/person_follow.py': c = 'A'
            elif p_norm == 'modules/auto_calibrate.py': c = 'A'
            elif p_norm == 'modules/tracking.py': c = 'A'
            elif p_norm.startswith('modules/control_system/'):
                if 'visualizer' in p_norm:
                    if 'visualizer_ui' in p_norm or p_norm == 'modules/drone_visualizer.py':
                        if 'visualizer_ui' in p_norm: c = 'D'
                        else: c = 'A'
                    else: c = 'A'
                else: c = 'A'
        elif p_norm.startswith('jetson/'): c = 'F'
        elif p_norm.startswith('shared/'): c = 'A'
        elif p_norm.startswith('tests/'): c = 'C'
        elif p_norm.startswith('tools/'): c = 'D'
        elif p_norm.startswith('benchmarks/'): c = 'D'
        elif p_norm in ['requirements.txt', '.gitignore']: c = 'E'
        elif p_norm == 'benchmarks/distance/manifest.json': c = 'B'
        elif p_norm == 'YOLO/yolo11n.pt': c = 'A'
        elif p_norm == 'autonomous_drone_main.py': c = 'A'
        elif p_norm.endswith('.md'):
            if any(x in p_norm.upper() for x in ['AUDIT', 'RECONCILIATION', 'VERIFICATION', 'CLEANUP', 'REPORT']): c = 'D'
            elif p_norm in ['DISTANCE_ESTIMATOR.md', 'PROJECT_REPORT.md', 'SGC_TRACKER_OVERLAY_SPEC.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md']: c = 'D'
            else: c = 'D'
        elif p_norm.startswith('.idea/') or p_norm.startswith('.vtcode/'): c = 'F'
        elif p_norm.startswith('.pytest_cache/') or p_norm.startswith('__pycache__/') or '/__pycache__/' in p_norm: c = 'H'
        elif p_norm in ['.gitignore', 'requirements.txt']: c = 'E'
        elif p_norm == 'benchmarks/distance/manifest.json': c = 'B'
        elif p_norm == 'YOLO/yolo11n.pt': c = 'A'
        elif p_norm == 'autonomous_drone_main.py': c = 'A'
        elif p_norm.endswith('.pyc'): c = 'H'
        elif p_norm in ['.gitignore', 'requirements.txt']: c = 'E'
        else: c = 'I'
        writer.writerow([rel, c, 'YES' if rel in git_files else 'NO'])

# Write Markdown
with open('AUTHORITATIVE_FILE_INVENTORY.md', 'w', encoding='utf-8') as f:
    f.write('# AUTHORITATIVE_FILE_INVENTORY.md\n\n')
    f.write('## Complete File Inventory\n\n')
    f.write('| path | classification | git_tracked |\n')
    f.write('|------|-------------|------------|\n')
    for rel in all_files:
        p = rel
        p_norm = p.replace('\\', '/')
        if p_norm == 'benchmarks/distance/manifest.json': c = 'B'
        elif p_norm == 'YOLO/yolo11n.pt': c = 'A'
        elif p_norm == 'autonomous_drone_main.py': c = 'A'
        elif p_norm == 'benchmarks/camera_calibration/checkerboard_9x6.png': c = 'D'
        elif p_norm.startswith('.pytest_cache/') or p_norm.startswith('__pycache__/') or '/__pycache__/' in p_norm: c = 'H'
        elif p_norm.endswith('.pyc'): c = 'H'
        elif p_norm == 'modules/navigation.py': c = 'G'
        elif p_norm == 'modules/vision.py': c = 'G'
        elif p_norm == 'modules/vision_utils/geometry.py': c = 'G'
        elif p_norm == 'modules/vision_utils/legacy.py': c = 'G'
        elif p_norm == 'modules/vision_utils/__init__.py': c = 'G'
        elif p_norm.startswith('modules/'):
            if p_norm.startswith('modules/distance_estimator/'):
                if os.path.basename(p_norm) in ['report.py', 'analysis.py', 'dataset.py', 'evaluation.py']: c = 'D'
                else: c = 'A'
            elif p_norm.startswith('modules/yolo11_detector/') or p_norm.startswith('modules/detector_yolo11'): c = 'A'
            elif p_norm == 'modules/person_follow.py': c = 'A'
            elif p_norm == 'modules/auto_calibrate.py': c = 'A'
            elif p_norm == 'modules/tracking.py': c = 'A'
            elif p_norm.startswith('modules/control_system/'):
                if 'visualizer' in p_norm:
                    if 'visualizer_ui' in p_norm or p_norm == 'modules/drone_visualizer.py':
                        if 'visualizer_ui' in p_norm: c = 'D'
                        else: c = 'A'
                    else: c = 'A'
                else: c = 'A'
        elif p_norm.startswith('jetson/'): c = 'F'
        elif p_norm.startswith('shared/'): c = 'A'
        elif p_norm.startswith('tests/'): c = 'C'
        elif p_norm.startswith('tools/'): c = 'D'
        elif p_norm.startswith('benchmarks/'): c = 'D'
        elif p_norm in ['requirements.txt', '.gitignore']: c = 'E'
        elif p_norm == 'benchmarks/distance/manifest.json': c = 'B'
        elif p_norm == 'YOLO/yolo11n.pt': c = 'A'
        elif p_norm == 'autonomous_drone_main.py': c = 'A'
        elif p_norm.endswith('.md'):
            if any(x in p_norm.upper() for x in ['AUDIT', 'RECONCILIATION', 'VERIFICATION', 'CLEANUP', 'REPORT']): c = 'D'
            elif p_norm in ['DISTANCE_ESTIMATOR.md', 'PROJECT_REPORT.md', 'SGC_TRACKER_OVERLAY_SPEC.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md']: c = 'D'
            else: c = 'D'
        elif p_norm.startswith('.idea/') or p_norm.startswith('.vtcode/'): c = 'F'
        elif p_norm.startswith('.pytest_cache/') or p_norm.startswith('__pycache__/') or '/__pycache__/' in p_norm: c = 'H'
        elif p_norm in ['.gitignore', 'requirements.txt']: c = 'E'
        elif p_norm == 'benchmarks/distance/manifest.json': c = 'B'
        elif p_norm == 'YOLO/yolo11n.pt': c = 'A'
        elif p_norm == 'autonomous_drone_main.py': c = 'A'
        elif p_norm.endswith('.pyc'): c = 'H'
        elif p_norm in ['.gitignore', 'requirements.txt']: c = 'E'
        else: c = 'I'
        f.write(f'| {rel} | {c} | {"YES" if rel in git_files else "NO"} |\n')
    f.write(f'\nTotal files: {len(all_files)}\n')

print('Classification counts:')
counts = Counter()
for rel in all_files:
    p = rel
    if p == 'benchmarks/distance/manifest.json': c = 'B'
    elif p == 'YOLO/yolo11n.pt': c = 'A'
    elif p == 'autonomous_drone_main.py': c = 'A'
    elif p == 'benchmarks/camera_calibration/checkerboard_9x6.png': c = 'D'
    elif p.startswith('.pytest_cache/') or p.startswith('__pycache__/') or '/__pycache__/' in p: c = 'H'
    elif p.endswith('.pyc'): c = 'H'
    elif p == 'modules/navigation.py': c = 'G'
    elif p == 'modules/vision.py': c = 'G'
    elif p == 'modules/vision_utils/geometry.py': c = 'G'
    elif p == 'modules/vision_utils/legacy.py': c = 'G'
    elif p == 'modules/vision_utils/__init__.py': c = 'G'
    elif p.startswith('modules/'):
        if p.startswith('modules/distance_estimator/'):
            if os.path.basename(p) in ['report.py', 'analysis.py', 'dataset.py', 'evaluation.py']: c = 'D'
            else: c = 'A'
        elif p.startswith('modules/yolo11_detector/') or p.startswith('modules/detector_yolo11'): c = 'A'
        elif p == 'modules/person_follow.py': c = 'A'
        elif p == 'modules/auto_calibrate.py': c = 'A'
        elif p == 'modules/tracking.py': c = 'A'
        elif p.startswith('modules/control_system/'):
            if 'visualizer' in p:
                if 'visualizer_ui' in p or p == 'modules/drone_visualizer.py':
                    if 'visualizer_ui' in p: c = 'D'
                    else: c = 'A'
                else: c = 'A'
            else: c = 'A'
    elif p.startswith('jetson/'): c = 'F'
    elif p.startswith('shared/'): c = 'A'
    elif p.startswith('tests/'): c = 'C'
    elif p.startswith('tools/'): c = 'D'
    elif p.startswith('benchmarks/'): c = 'D'
    elif p in ['requirements.txt', '.gitignore']: c = 'E'
    elif p == 'benchmarks/distance/manifest.json': c = 'B'
    elif p == 'YOLO/yolo11n.pt': c = 'A'
    elif p == 'autonomous_drone_main.py': c = 'A'
    elif p.endswith('.md'):
        if any(x in p.upper() for x in ['AUDIT', 'RECONCILIATION', 'VERIFICATION', 'CLEANUP', 'REPORT']): c = 'D'
        elif p in ['DISTANCE_ESTIMATOR.md', 'PROJECT_REPORT.md', 'SGC_TRACKER_OVERLAY_SPEC.md', 'GPU_CUDA_VERIFICATION.md', 'JETSON_DEPLOYMENT_FILE_MAP.md', 'PRE_JETSON_BASELINE_REPORT.md', 'PRE_JETSON_VERIFICATION_REPORT.md']: c = 'D'
        else: c = 'D'
    elif p.startswith('.idea/') or p.startswith('.vtcode/'): c = 'F'
    elif p.startswith('.pytest_cache/') or p.startswith('__pycache__/') or '/__pycache__/' in p: c = 'H'
    elif p in ['.gitignore', 'requirements.txt']: c = 'E'
    elif p == 'benchmarks/distance/manifest.json': c = 'B'
    elif p == 'YOLO/yolo11n.pt': c = 'A'
    elif p == 'autonomous_drone_main.py': c = 'A'
    elif p.endswith('.pyc'): c = 'H'
    elif p in ['.gitignore', 'requirements.txt']: c = 'E'
    else: c = 'I'
    counts[c] = counts.get(c, 0) + 1

print('Classification counts:')
for cls in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
    print(f'  {cls}: {counts.get(cls, 0)}')
print(f'Total: {sum(counts.values())}')