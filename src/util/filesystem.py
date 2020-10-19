import os


def files_in_dir_tree(path):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory!")

    return [
        os.path.join(root, name)
        for root, _, files in os.walk(path, topdown=False)
        for name in files
    ]
