import os
import argparse
import pandas as pd
import hashlib

def add_args(parser: argparse.ArgumentParser):
    pass


def generate_metadata(output_dir, root_folder, **kwargs):

    dir_set = [
        d
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ]

    file_paths = []
    labels = []
    split = []
    classe_data = []

    for direct in dir_set:
        dir_path = os.path.join(root_folder, direct)
        classes = [
            d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))
        ]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_path = os.path.join(dir_path, cls_name)
            for file in os.listdir(cls_path):
                file_paths.append(os.path.join(cls_path, file))
                labels.append(class_to_idx[cls_name])
                split.append(direct)
                classe_data.append(cls_name)

    sha256 = [hashlib.sha256(f.encode()).hexdigest() for f in file_paths]
    local_path = [os.path.relpath(f, output_dir) for f in file_paths]

    metadata = pd.DataFrame(
        {
            "sha256": sha256,
            "file_identifier": file_paths,
            "local_path": local_path,
            "label": labels,
            "class": classe_data,
            "split": split,
        }
    )

    metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def get_metadata(output_dir, root_folder, **kwargs):
    if not os.path.exists(os.path.join(output_dir, "metadata.csv")):
        generate_metadata(output_dir, root_folder)
    metadata = pd.read_csv(os.path.join(output_dir, "metadata.csv"))

    return metadata


def download(metadata, output_dir, **kwargs):
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    sha256s = metadata["sha256"].values
    local_path = metadata["local_path"].values
    for i in range(len(sha256s)):
        downloaded[sha256s[i]] = local_path[i]

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def foreach_instance(
    metadata, output_dir, func, max_workers=None, desc="Processing objects"
) -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile

    # load metadata
    metadata = metadata.to_dict("records")

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(metadata), desc=desc
        ) as pbar:

            def worker(metadatum):
                try:
                    local_path = metadatum["local_path"]
                    sha256 = metadatum["sha256"]
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error processing objects: {e}")

    return pd.DataFrame.from_records(records)
