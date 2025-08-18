DATA_CONFIGS = {
    'prmbench': {
        'task_name': 'prmbench',
        'task_id': 'modified',  # Use modified process (default behavior)
        'overcomplete': True,
        'dataset_type': 'prmbench',
    },
    'prmbench_original': {
        'task_name': 'prmbench_original',
        'task_id': 'original',  # Use original question and process
        'overcomplete': True,
        'dataset_type': 'prmbench',
    },
    'prmbench_min_length_10': {
        'task_name': 'prmbench_min_length_10',
        'task_id': 'modified',  # Use modified process
        'overcomplete': True,
        'dataset_type': 'prmbench',
        'min_length': 10,
    },
    'prmbench_min_length_20': {
        'task_name': 'prmbench_min_length_20',
        'task_id': 'modified',  # Use modified process
        'overcomplete': True,
        'dataset_type': 'prmbench',
        'min_length': 20,
    },
    'prmbench_min_length_30': {
        'task_name': 'prmbench_min_length_30',
        'task_id': 'modified',  # Use modified process
        'overcomplete': True,
        'dataset_type': 'prmbench',
        'min_length': 30,
    },
    'prmbench_min_length_40': {
        'task_name': 'prmbench_min_length_40',
        'task_id': 'modified',  # Use modified process
        'overcomplete': True,
        'dataset_type': 'prmbench',
        'min_length': 40,  # Fixed: was missing min_length value
    },
}