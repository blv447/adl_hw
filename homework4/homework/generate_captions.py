import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    if ego_kart is None:
        return []

    ego_x, ego_y = ego_kart["center"]
    ego_distance = ego_kart["distance_down_track"]

    captions = []

    # 1. Ego car
    captions.append(f"{ego_kart['kart_name']} is the ego car.")

    # 2. Counting
    captions.append(f"There are {len(karts)} karts in the scene.")

    # 3. Track name
    track_name = extract_track_info(info_path)
    captions.append(f"The track is {track_name}.")

    # 4. Relative position (front/behind and left/right as separate captions per kart)
    for kart in karts:
        if kart["is_center_kart"]:
            continue

        kart_x, _ = kart["center"]
        left_right = "left" if kart_x < ego_x else "right"

        if ego_distance is not None and kart["distance_down_track"] is not None:
            front_behind = "in front of" if kart["distance_down_track"] > ego_distance else "behind"
        else:
            _, kart_y = kart["center"]
            front_behind = "in front of" if kart_y < ego_y else "behind"

        captions.append(f"{kart['kart_name']} is {front_behind} the ego car.")
        captions.append(f"{kart['kart_name']} is {left_right} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_dataset(
    split: str,
    output_name: str = None,
    img_width: int = 150,
    img_height: int = 100,
    max_views_per_frame: int = None,
):
    """
    Generate captions for every frame/view in a data split and write them out to
    data/train/<output_name>_captions.json as (image_file, caption) pairs.

    Args:
        split: Name of the folder under data/ containing *_info.json + *_im.jpg files.
        output_name: Prefix for the output file name (default: split name).
                     Written to data/train/{output_name}_captions.json.
        img_width / img_height: Must match the resized image size used at train/eval time.
        max_views_per_frame: Optionally cap how many views (camera indices) are used per
                              frame, useful for controlling dataset size.
    """
    data_dir = Path(__file__).parent.parent / "data"
    split_dir = data_dir / split
    output_name = output_name or split

    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {split_dir}")

    all_pairs = []
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        image_files = sorted(split_dir.glob(f"{base_name}_*_im.jpg"))

        view_indices = [extract_frame_info(str(img))[1] for img in image_files]
        if max_views_per_frame is not None:
            view_indices = view_indices[:max_views_per_frame]

        for view_index in view_indices:
            captions = generate_caption(str(info_file), view_index, img_width, img_height)
            image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            for caption in captions:
                all_pairs.append({"image_file": image_file, "caption": caption})

    output_path = data_dir / "train" / f"{output_name}_captions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Wrote {len(all_pairs)} (image, caption) pairs to {output_path}")


"""
Usage Example: Visualize a caption for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

Generate the full training captions file from data/train/*_info.json:
   python generate_captions.py generate_dataset --split train
"""


def main():
    fire.Fire({"check": check_caption, "generate_dataset": generate_dataset})


if __name__ == "__main__":
    main()