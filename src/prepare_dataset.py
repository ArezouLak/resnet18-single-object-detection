from pathlib import Path
import cv2 as cv


def prepare_dataset(annotation_dir, image_dir):
    """Load a single-object localization dataset and normalize xyxy boxes to [0,1]."""
    annotation_dir = Path(annotation_dir)
    image_dir = Path(image_dir)
    images, labels, bboxes, image_paths = [], [], [], []
    for csv_file in sorted(annotation_dir.glob('*.csv')):
        with csv_file.open('r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) != 6:
                    continue
                image_name, xmin, ymin, xmax, ymax, label = parts
                image_path = image_dir / label / image_name
                image = cv.imread(str(image_path))
                if image is None:
                    print(f'Could not read: {image_path}')
                    continue
                h, w = image.shape[:2]
                bbox = [float(xmin)/w, float(ymin)/h, float(xmax)/w, float(ymax)/h]
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (224,224))
                images.append(image); labels.append(label); bboxes.append(bbox); image_paths.append(str(image_path))
    return images, labels, bboxes, image_paths
