import imageio
import os

image_folder = 'images'
gif_name = 'video.gif'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
sorted_images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))

# Check if there are images to process
if not sorted_images:
    print("No images found in the specified folder.")
    exit()

# Process and add each image to the GIF
frames = []
for image in sorted_images:
    image_path = os.path.join(image_folder, image)
    img = imageio.imread(image_path)
    if img is None:
        print(f"Unable to read the image: {image_path}")
        continue  # Skip to the next image if unable to read

    frames.append(img)

print(f"Processed images!")

# Save the frames as a GIF
imageio.mimsave(gif_name, frames, duration=5)  # Adjust fps as necessary

print("GIF creation completed successfully.")