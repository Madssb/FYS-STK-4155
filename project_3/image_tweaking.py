from PIL import Image
from pathlib import Path
from PIL import ImageEnhance

dir = Path.cwd()
image_paths = sorted(dir.rglob("**/*.jpg"))
image_path = image_paths[0]
with Image.open(image_path) as image:
    image.show()
    enhancer = ImageEnhance.Contrast(image)
    enhancer.enhance(10).show()
