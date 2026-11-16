# -*- coding: utf-8 -*-
"""
28_Extract_Corrected_RF_FinalGW_1000_Heads.py

RUN WITH ABAQUS PYTHON.

Extract the same 23 borehole hydraulic heads from all 1000 corrected-random-
field FinalGW ODBs.

Expected ODB directory:
matlab/inputs/Corrected_RF_FinalGW_1000/INP

Head formula:
H = node_z + POR / 9810

Outputs:
- Corrected_RF_Heads_1000x23.csv
- Borehole_Node_Mapping.csv
- Extraction_Status_1000.csv
- Failed_ODB_List.txt (only if failures occur)
"""

from odbAccess import openOdb
import os
import sys
import csv
import math
import traceback


# ============================================================================
# 1. Paths
# ============================================================================

ODB_DIR = os.environ.get('GHB_ODB_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'matlab', 'inputs', 'Corrected_RF_FinalGW_1000', 'INP'))

ROOT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abaqus_extraction')

if not os.path.isdir(ROOT_OUT):
    os.makedirs(ROOT_OUT)

HEADS_CSV = os.path.join(ROOT_OUT, 'Corrected_RF_Heads_1000x23.csv')
NODE_MAP_CSV = os.path.join(ROOT_OUT, 'Borehole_Node_Mapping.csv')
STATUS_CSV = os.path.join(ROOT_OUT, 'Extraction_Status_1000.csv')
FAILED_TXT = os.path.join(ROOT_OUT, 'Failed_ODB_List.txt')


# ============================================================================
# 2. Constants
# ============================================================================

N_PARAM = 50
N_REAL = 20
EXPECTED_ODBS = N_PARAM * N_REAL
GAMMA_W = 9810.0


# ============================================================================
# 3. Exact 23 target coordinates and final column order
# ============================================================================

BOREHOLES = {
    'zk107': (3320.895783, 1193.843810, 461.039511),
    'zk108': (3141.815783, 1157.263800, 426.289511),
    'zk109': (2895.535783, 1114.513810, 425.159511),
    'zk131': (2838.945783, 1182.523810, 419.499511),
    'zk140': (2861.595783, 1298.223810, 398.019511),
    'zk120': (2840.215783, 2085.103810, 407.869511),
    'zk102': (2940.805783, 2051.683810, 404.169511),
    'zk141': (2986.595783, 1996.693810, 404.859511),
    'zk103': (3047.915783, 1938.063810, 426.159511),
    'zk139': (3169.345783, 1901.143810, 416.719511),
    'zk138': (3214.505783, 1840.693810, 433.649511),
    'zk104': (3271.755783, 1784.973810, 449.769511),
    'zk137': (3330.475783, 1777.113810, 460.129511),
    'zk105': (3497.575783, 1597.663810, 472.299511),
    'zk133': (3032.845783, 1196.793810, 412.939511),
    'zk134': (3130.065783, 1199.803810, 404.709511),
    'zk106': (3543.165783, 1425.393810, 502.699511),
    'zk135': (3210.325783, 1263.323810, 403.779511),
    'zk132': (2939.555783, 1194.793810, 425.159511),
    'zk101': (2823.765783, 2163.383810, 421.989511),
    'zk410': (3349.685783, 1429.863810, 487.489511),
    'zk136': (3281.245783, 1315.773810, 406.919511),
    'zk110': (2799.965783, 1287.383810, 409.169511),
}

BOREHOLE_ORDER = [
    'zk101', 'zk102', 'zk103', 'zk104', 'zk105', 'zk106',
    'zk107', 'zk108', 'zk109', 'zk110', 'zk120', 'zk131',
    'zk132', 'zk133', 'zk134', 'zk135', 'zk136', 'zk137',
    'zk138', 'zk139', 'zk140', 'zk141', 'zk410'
]


# ============================================================================
# 4. CSV helper
# ============================================================================

def open_csv_write(path):
    if sys.version_info[0] < 3:
        return open(path, 'wb')
    return open(path, 'w', newline='', encoding='utf-8-sig')


# ============================================================================
# 5. ODB helpers
# ============================================================================

NODE_CACHE = {}


def get_instance(odb):
    names = list(odb.rootAssembly.instances.keys())

    for name in names:
        if 'DICENG' in name.upper():
            return odb.rootAssembly.instances[name]

    if len(names) == 0:
        raise RuntimeError('No assembly instance found.')

    return odb.rootAssembly.instances[names[0]]


def find_nearest(nodes, tx, ty, tz):
    best_index = -1
    best_node = None
    best_dist = 1.0e100

    for i, node in enumerate(nodes):
        x, y, z = node.coordinates

        d = math.sqrt(
            (x - tx) ** 2
            + (y - ty) ** 2
            + (z - tz) ** 2
        )

        if d < best_dist:
            best_index = i
            best_node = node
            best_dist = d

    return best_node, best_index, best_dist


def build_node_cache(first_odb_path):
    odb = openOdb(path=first_odb_path, readOnly=True)

    try:
        instance = get_instance(odb)
        nodes = instance.nodes

        for bh in BOREHOLE_ORDER:
            tx, ty, tz = BOREHOLES[bh]

            node, idx, dist = find_nearest(
                nodes,
                tx,
                ty,
                tz
            )

            NODE_CACHE[bh] = {
                'index': idx,
                'label': node.label,
                'target_x': tx,
                'target_y': ty,
                'target_z': tz,
                'node_x': node.coordinates[0],
                'node_y': node.coordinates[1],
                'node_z': node.coordinates[2],
                'distance_m': dist
            }

    finally:
        odb.close()


def validate_cached_node(nodes, bh):
    c = NODE_CACHE[bh]
    idx = c['index']

    if idx < 0 or idx >= len(nodes):
        raise RuntimeError('Cached node index out of range for %s.' % bh)

    node = nodes[idx]

    if node.label != c['label']:
        raise RuntimeError(
            'Mesh/node ordering changed for %s: expected node %s, got %s.'
            % (bh, str(c['label']), str(node.label))
        )

    return node


def extract_one_odb(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)

    try:
        step_names = list(odb.steps.keys())

        if len(step_names) == 0:
            raise RuntimeError('ODB has no analysis step.')

        step = odb.steps[step_names[-1]]

        if len(step.frames) == 0:
            raise RuntimeError('Final step has no output frame.')

        frame = step.frames[-1]

        if 'POR' not in frame.fieldOutputs.keys():
            raise RuntimeError('POR field output not found.')

        por_field = frame.fieldOutputs['POR']

        instance = get_instance(odb)
        nodes = instance.nodes

        heads = {}

        for bh in BOREHOLE_ORDER:
            node = validate_cached_node(nodes, bh)

            subset = por_field.getSubset(region=node)

            if len(subset.values) == 0:
                raise RuntimeError(
                    'No POR value for %s at node %s.'
                    % (bh, str(node.label))
                )

            por = subset.values[0].data
            node_z = node.coordinates[2]

            heads[bh] = node_z + por / GAMMA_W

        return heads

    finally:
        odb.close()


def is_finite_number(x):
    try:
        value = float(x)
    except Exception:
        return False

    if value != value:
        return False

    if value == float('inf') or value == float('-inf'):
        return False

    return True


# ============================================================================
# 6. Build expected ODB list and require all 1000
# ============================================================================

expected_jobs = []

for p in range(1, N_PARAM + 1):
    for r in range(1, N_REAL + 1):
        job_name = 'Job_P%03d_R%02d' % (p, r)
        odb_path = os.path.join(ODB_DIR, job_name + '.odb')

        expected_jobs.append(
            (p, r, job_name, odb_path)
        )


existing_count = sum(
    1 for item in expected_jobs
    if os.path.exists(item[3])
)


print('=' * 96)
print('STEP 28: EXTRACT CORRECTED-RF FINALGW 1000 x 23 HEADS')
print('=' * 96)
print('Expected ODBs : %d' % EXPECTED_ODBS)
print('Existing ODBs : %d' % existing_count)
print('ODB directory : %s' % ODB_DIR)


if existing_count != EXPECTED_ODBS:
    missing = [
        item[2]
        for item in expected_jobs
        if not os.path.exists(item[3])
    ]

    print('\nMissing ODB count: %d' % len(missing))

    for name in missing[:30]:
        print('  missing: %s.odb' % name)

    if len(missing) > 30:
        print('  ... and %d more' % (len(missing) - 30))

    raise RuntimeError(
        'Database incomplete: expected all 1000 ODBs before extraction.'
    )


# ============================================================================
# 7. Build fixed nearest-node map once
# ============================================================================

first_odb = os.path.join(ODB_DIR, 'Job_P001_R01.odb')

print('\nBuilding fixed 23-borehole node map from Job_P001_R01.odb ...')

build_node_cache(first_odb)

max_distance = max(
    NODE_CACHE[bh]['distance_m']
    for bh in BOREHOLE_ORDER
)

print(
    'Node mapping complete. Max nearest-node distance = %.6f m'
    % max_distance
)


# ============================================================================
# 8. Save node mapping
# ============================================================================

map_fields = [
    'Borehole',
    'Target_X', 'Target_Y', 'Target_Z',
    'Node_ID',
    'Node_X', 'Node_Y', 'Node_Z',
    'Distance_m'
]

f = open_csv_write(NODE_MAP_CSV)

try:
    writer = csv.DictWriter(f, fieldnames=map_fields)
    writer.writeheader()

    for bh in BOREHOLE_ORDER:
        c = NODE_CACHE[bh]

        writer.writerow({
            'Borehole': bh,
            'Target_X': c['target_x'],
            'Target_Y': c['target_y'],
            'Target_Z': c['target_z'],
            'Node_ID': c['label'],
            'Node_X': c['node_x'],
            'Node_Y': c['node_y'],
            'Node_Z': c['node_z'],
            'Distance_m': c['distance_m']
        })

finally:
    f.close()


# ============================================================================
# 9. Extract all 1000 ODBs
# ============================================================================

head_rows = []
status_rows = []
failed_rows = []

for i, item in enumerate(expected_jobs):
    p, r, job_name, odb_path = item

    print(
        '[%04d/%04d] %s'
        % (i + 1, EXPECTED_ODBS, job_name)
    )

    try:
        heads = extract_one_odb(odb_path)

        record = {
            'P_index': p,
            'Realization': r,
            'Job_Name': job_name,
            'odb_file': os.path.basename(odb_path)
        }

        for bh in BOREHOLE_ORDER:
            value = heads[bh]

            if not is_finite_number(value):
                raise RuntimeError(
                    'Non-finite head for %s.'
                    % bh
                )

            record[bh] = value

        head_rows.append(record)

        status_rows.append({
            'P_index': p,
            'Realization': r,
            'Job_Name': job_name,
            'odb_file': os.path.basename(odb_path),
            'Status': 'PASS',
            'Message': ''
        })

    except Exception as exc:
        message = '%s: %s' % (
            exc.__class__.__name__,
            str(exc)
        )

        status_rows.append({
            'P_index': p,
            'Realization': r,
            'Job_Name': job_name,
            'odb_file': os.path.basename(odb_path),
            'Status': 'FAIL',
            'Message': message
        })

        failed_rows.append(
            (job_name, message, traceback.format_exc())
        )

        print('    FAIL: %s' % message)


# ============================================================================
# 10. Save extraction status
# ============================================================================

status_fields = [
    'P_index',
    'Realization',
    'Job_Name',
    'odb_file',
    'Status',
    'Message'
]

f = open_csv_write(STATUS_CSV)

try:
    writer = csv.DictWriter(f, fieldnames=status_fields)
    writer.writeheader()

    for row in status_rows:
        writer.writerow(row)

finally:
    f.close()


# ============================================================================
# 11. Fail safely if any extraction failed
# ============================================================================

if len(failed_rows) > 0:
    if sys.version_info[0] < 3:
        f = open(FAILED_TXT, 'wb')
    else:
        f = open(FAILED_TXT, 'w', encoding='utf-8')

    try:
        for job_name, message, tb in failed_rows:
            text = (
                '%s\n%s\n%s\n%s\n'
                % ('=' * 80, job_name, message, tb)
            )

            if sys.version_info[0] < 3:
                f.write(text.encode('utf-8'))
            else:
                f.write(text)

    finally:
        f.close()

    print('\n' + '=' * 96)
    print('EXTRACTION INCOMPLETE')
    print('=' * 96)
    print('PASS : %d' % len(head_rows))
    print('FAIL : %d' % len(failed_rows))
    print('See:')
    print('  %s' % STATUS_CSV)
    print('  %s' % FAILED_TXT)

    raise RuntimeError(
        'At least one ODB failed extraction. Final 1000x23 CSV was not written.'
    )


# ============================================================================
# 12. Final validation and write 1000 x 23 matrix
# ============================================================================

if len(head_rows) != EXPECTED_ODBS:
    raise RuntimeError(
        'Expected 1000 successful rows, got %d.'
        % len(head_rows)
    )

head_fields = [
    'P_index',
    'Realization',
    'Job_Name',
    'odb_file'
] + BOREHOLE_ORDER

f = open_csv_write(HEADS_CSV)

try:
    writer = csv.DictWriter(f, fieldnames=head_fields)
    writer.writeheader()

    for row in head_rows:
        writer.writerow(row)

finally:
    f.close()


# ============================================================================
# 13. Global diagnostics
# ============================================================================

all_heads = []

for row in head_rows:
    for bh in BOREHOLE_ORDER:
        all_heads.append(float(row[bh]))

global_min = min(all_heads)
global_max = max(all_heads)
global_mean = sum(all_heads) / len(all_heads)


print('\n' + '=' * 96)
print('STEP 28 COMPLETE')
print('=' * 96)
print('Successful ODBs      : %d / %d' % (len(head_rows), EXPECTED_ODBS))
print('Head rows            : %d' % len(head_rows))
print('Heads per row        : %d' % len(BOREHOLE_ORDER))
print('Total head values    : %d' % len(all_heads))
print('Global head range    : [%.6f, %.6f] m' % (global_min, global_max))
print('Global mean head     : %.6f m' % global_mean)
print('Max node-map distance: %.6f m' % max_distance)

print('\nSaved:')
print('  %s' % HEADS_CSV)
print('  %s' % NODE_MAP_CSV)
print('  %s' % STATUS_CSV)

print('\nNEXT:')
print('Step 29 = rebuild the corrected Final ML dataset by merging these')
print('1000x23 heads with the frozen 50 macroparameter groups.')
