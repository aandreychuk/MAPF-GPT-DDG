def filter_data(inputs, gt_actions):
    if len(inputs) == 0 or len(gt_actions) == 0:
        return None

    known_hashes = set()
    filtered_inputs = []
    filtered_gt_actions = []

    for input, gt_action in zip(inputs, gt_actions):
        input_tuple = tuple(input)
        input_hash = hash(input_tuple)

        if input_hash not in known_hashes:
            known_hashes.add(input_hash)
            filtered_inputs.append(input)
            filtered_gt_actions.append(gt_action)

    return {"inputs": filtered_inputs, "gt_actions": filtered_gt_actions}
