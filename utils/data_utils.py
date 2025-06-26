import pyarrow as pa
import pyarrow.ipc as ipc
import os

def save_to_arrow(inputs, gt_actions, filepath):
    schema = pa.schema([
        ('input_tensors', pa.list_(pa.int8())),
        ('gt_actions', pa.int8())
    ])

    input_tensors_col = pa.array(inputs, type=pa.list_(pa.int8()))
    gt_actions_col = pa.array(gt_actions)
    table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write(table)


def compute_metrics_diff(left, right):
    result = {}
    for metric in ['ISR', 'CSR', 'makespan']:
        if not (metric in left and metric in right):
            continue
        result[metric] = right[metric] - left[metric]
        # if metric == 'makespan':
        #     result[metric] *= -1
    return result
