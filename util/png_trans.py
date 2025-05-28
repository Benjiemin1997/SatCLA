import os
from PIL import Image

def convert_tif_to_jpg(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                tif_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_dir)
                jpg_dir = os.path.join(output_dir, relative_path)
                jpg_filename = os.path.splitext(filename)[0] + ".jpg"
                jpg_path = os.path.join(jpg_dir, jpg_filename)

                if not os.path.exists(jpg_dir):
                    os.makedirs(jpg_dir)

                with Image.open(tif_path) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(jpg_path, "JPEG", quality=95)
                print(f"Converted {tif_path} to {jpg_path}")

if __name__ == "__main__":
    input_directory = ""
    output_directory = ""
    convert_tif_to_jpg(input_directory, output_directory)