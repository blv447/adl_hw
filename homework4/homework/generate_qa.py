import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def _load_info(info_path: str) -> dict:
    with open(info_path) as f:
        return json.load(f)


def _kart_name_lookup(info: dict, track_id: int) -> str:
    """
    Resolve a kart's display/identifier name from the info dict, given its track_id.

    The exact schema of info.json can vary a bit (list of names indexed by track_id,
    list of dicts, or a dict keyed by track_id), so we handle the common shapes here.
    If you find your actual info.json uses a different key, update KART_NAME_KEYS below.
    """
    karts_field = info.get("karts") or info.get("kart_names") or info.get("kart_list")

    if karts_field is None:
        return f"kart_{track_id}"

    # list of plain strings, indexed by track_id
    if isinstance(karts_field, list) and (len(karts_field) == 0 or isinstance(karts_field[0], str)):
        if 0 <= track_id < len(karts_field):
            return karts_field[track_id]
        return f"kart_{track_id}"

    # list of dicts, e.g. [{"instance_id": 0, "kart_name": "tux"}, ...]
    if isinstance(karts_field, list):
        for entry in karts_field:
            if isinstance(entry, dict):
                entry_id = entry.get("instance_id", entry.get("track_id", entry.get("id")))
                if entry_id == track_id:
                    return entry.get("kart_name", entry.get("name", f"kart_{track_id}"))
        return f"kart_{track_id}"

    # dict keyed by track_id (possibly as a string key)
    if isinstance(karts_field, dict):
        return karts_field.get(track_id, karts_field.get(str(track_id), f"kart_{track_id}"))

    return f"kart_{track_id}"


def _distance_down_track_lookup(info: dict, track_id: int) -> float | None:
    """
    Resolve a kart's race-progress distance from info["distance_down_track"], a flat list
    indexed by track_id (parallel to info["karts"]). This is the authoritative measure of
    whether a kart is ahead of ("front") or behind ("back") another kart on the track --
    far more robust than screen-space Y position, which is sensitive to camera angle,
    curves, and hills.
    """
    distances = info.get("distance_down_track")
    if not isinstance(distances, list) or not (0 <= track_id < len(distances)):
        return None
    return distances[track_id]


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
        - distance_down_track: The kart's race-progress distance (None if unavailable)
    """
    info = _load_info(info_path)

    if view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    kart_objects = []
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        # Only interested in karts, not track boundaries / other elements
        if class_id != 1:
            continue

        x1_s = x1 * scale_x
        y1_s = y1 * scale_y
        x2_s = x2 * scale_x
        y2_s = y2 * scale_y

        # Skip boxes that are too small to be meaningful
        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
            continue

        # Skip karts that are fully out of the image bounds
        if x2_s < 0 or x1_s > img_width or y2_s < 0 or y1_s > img_height:
            continue

        center_x = (x1_s + x2_s) / 2
        center_y = (y1_s + y2_s) / 2

        kart_objects.append(
            {
                "instance_id": track_id,
                "kart_name": _kart_name_lookup(info, track_id),
                "center": (center_x, center_y),
                "is_center_kart": False,
                "distance_down_track": _distance_down_track_lookup(info, track_id),
            }
        )

    # The ego kart is the one whose bounding-box center is closest to the image center,
    # since each view is a chase-camera following that particular kart.
    if kart_objects:
        img_center_x, img_center_y = img_width / 2, img_height / 2
        ego = min(
            kart_objects,
            key=lambda k: (k["center"][0] - img_center_x) ** 2 + (k["center"][1] - img_center_y) ** 2,
        )
        ego["is_center_kart"] = True

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    info = _load_info(info_path)
    track_name = info.get("track") or info.get("track_name")
    if track_name is None:
        raise KeyError(
            f"Could not find a track name in {info_path}. "
            f"Available top-level keys: {list(info.keys())}"
        )
    return track_name


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    if ego_kart is None:
        return []

    ego_x, ego_y = ego_kart["center"]
    ego_distance = ego_kart["distance_down_track"]
    qa_pairs = []

    # 1. Ego car question
    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego_kart["kart_name"]})

    # 2. Total karts question
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts))})

    # 3. Track information questions
    track_name = extract_track_info(info_path)
    qa_pairs.append({"question": "What track is this?", "answer": track_name})

    # 4. Relative position questions for each kart (skip the ego kart itself)
    num_left = num_right = num_front = num_behind = 0
    for kart in karts:
        if kart["is_center_kart"]:
            continue

        kart_x, _ = kart["center"]

        # Smaller x -> left of ego.
        left_right = "left" if kart_x < ego_x else "right"

        # Prefer race-progress distance (robust to camera angle/curves/hills) for front/back.
        # Falls back to screen-space Y if distance_down_track wasn't available.
        # NOTE: the answer vocabulary is "front"/"back" (not "behind"), even though the
        # *question* text says "in front of or behind" per the assignment's template.
        if ego_distance is not None and kart["distance_down_track"] is not None:
            front_behind = "front" if kart["distance_down_track"] > ego_distance else "back"
        else:
            _, kart_y = kart["center"]
            front_behind = "front" if kart_y < ego_y else "back"

        if left_right == "left":
            num_left += 1
        else:
            num_right += 1
        if front_behind == "front":
            num_front += 1
        else:
            num_behind += 1

        qa_pairs.append(
            {
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": left_right,
            }
        )
        qa_pairs.append(
            {
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": front_behind,
            }
        )
        qa_pairs.append(
            {
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": f"{front_behind} and {left_right}",
            }
        )

    # 5. Counting questions
    qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(num_left)})
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(num_right)})
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(num_front)})
    qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(num_behind)})

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def inspect_info(info_file: str):
    """
    Debug helper: print the top-level keys (and a peek at a couple of them) of an info.json
    file. Handy for double-checking the real field names (e.g. "karts" vs "kart_names")
    before relying on the assumptions baked into extract_kart_objects / extract_track_info.
    """
    info = _load_info(info_file)
    print(f"Top-level keys: {list(info.keys())}")
    for key in ("track", "track_name", "karts", "kart_names", "kart_list", "distance_down_track", "velocity"):
        if key in info:
            value = info[key]
            if isinstance(value, list):
                preview = value[:5]
                print(f"  {key!r}: type=list, len={len(value)}, preview={preview}")
                if value and isinstance(value[0], list):
                    print(f"    (nested) first element: type=list, len={len(value[0])}, preview={value[0][:5]}")
            else:
                print(f"  {key!r}: {value}")
    if "detections" in info:
        print(f"  'detections': {len(info['detections'])} views")
        if info["detections"]:
            print(f"    view 0 has {len(info['detections'][0])} detections, e.g. {info['detections'][0][:3]}")


def generate_dataset(
    split: str,
    output_name: str = None,
    img_width: int = 150,
    img_height: int = 100,
    max_views_per_frame: int = None,
):
    """
    Generate QA pairs for every frame/view in a data split (e.g. "train") and write them
    out to data/train/<output_name>_qa_pairs.json.

    Args:
        split: Name of the folder under data/ containing *_info.json + *_im.jpg files
               (e.g. "train"). This is also where the info files are read from.
        output_name: Prefix for the output file name (default: split name).
                     Written to data/train/{output_name}_qa_pairs.json.
        img_width / img_height: Must match the resized image size used at train/eval time.
        max_views_per_frame: Optionally cap how many views (camera indices) are used per
                              frame, useful for controlling dataset size.
    """
    data_dir = Path(__file__).parent.parent / "data"
    split_dir = data_dir / split
    output_name = output_name or split

    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {split_dir}")

    all_qa_pairs = []
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        image_files = sorted(split_dir.glob(f"{base_name}_*_im.jpg"))

        view_indices = [extract_frame_info(str(img))[1] for img in image_files]
        if max_views_per_frame is not None:
            view_indices = view_indices[:max_views_per_frame]

        for view_index in view_indices:
            qa_pairs = generate_qa_pairs(str(info_file), view_index, img_width, img_height)
            image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            for qa in qa_pairs:
                all_qa_pairs.append(
                    {
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "image_file": image_file,
                    }
                )

    output_path = data_dir / "train" / f"{output_name}_qa_pairs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Wrote {len(all_qa_pairs)} QA pairs to {output_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

Inspect the raw schema of an info.json file:
   python generate_qa.py inspect_info --info_file ../data/valid/00000_info.json

Generate the full training QA-pairs file from data/train/*_info.json:
   python generate_qa.py generate_dataset --split train
"""


def main():
    fire.Fire(
        {
            "check": check_qa_pairs,
            "inspect_info": inspect_info,
            "generate_dataset": generate_dataset,
        }
    )


if __name__ == "__main__":
    main()