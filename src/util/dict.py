def flatten_dict(obj, parent_key='', separator='_'):
    item = []

    for key, value in obj.items():
        aggregated_key = parent_key + separator + key if parent_key else key

        try:
            flatten = flatten_dict(
                value,
                aggregated_key,
                separator=separator
            ).items()

            item.extend(flatten)
        except Exception:
            item.append((aggregated_key, value))

    return dict(item)
