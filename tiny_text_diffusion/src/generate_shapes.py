
import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw

SEED = 42
NUM_IMAGES = 500
IMAGE_SIZE = 32

def generate_dataset(output_dir, num_images=NUM_IMAGES):
    random.seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"

    colors = [
        (220, 50, 50),
        (40, 110, 220),
        (40, 170, 80),
        (240, 180, 30),
        (160, 70, 190),
    ]
    shapes = ["circle", "square", "triangle"]

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "shape", "color"])

        for i in range(num_images):
            shape = random.choice(shapes)
            color = random.choice(colors)
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
            draw = ImageDraw.Draw(image)

            x1 = random.randint(4, 10)
            y1 = random.randint(4, 10)
            x2 = random.randint(22, 28)
            y2 = random.randint(22, 28)

            if shape == "circle":
                draw.ellipse((x1, y1, x2, y2), fill=color)
            elif shape == "square":
                side = min(x2 - x1, y2 - y1)
                draw.rectangle((x1, y1, x1 + side, y1 + side), fill=color)
            else:
                draw.polygon(
                    [(16, y1), (x1, y2), (x2, y2)],
                    fill=color
                )

            filename = f"shape_{i:05d}.png"
            image.save(output_dir / filename)
            writer.writerow([filename, shape, color])

    print(f"Generated {num_images} images in {output_dir}")
    print(f"Labels saved to {labels_path}")

if __name__ == "__main__":
    generate_dataset(
        Path(__file__).resolve().parents[1] / "dataset" / "synthetic_shapes"
    )
