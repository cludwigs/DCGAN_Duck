"""
@Author: Cédric Ludwigs
@Description: This script is used to prepare a dataset of duck images for training a DCGAN model. 
It just open the images and resize them to 128x128 pixels.
"""

# Import necessary libraries
import PIL.Image
import os 

# Define global variables 
INPUT_DIR = 'data/duck_original'  # Directory containing the original duck images
OUTPUT_DIR = 'data/duck_resized/duck'  # Directory to save the resized images

width = 128

def resize_images(input_dir, output_dir, size=(width, width), show_progress=True):
    """
    Resize all images in the input directory to the specified size and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing the original images.
        output_dir (str): Path to the directory where resized images will be saved.
        size (tuple): Desired size for the resized images (width, height).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
            # Open the image file
            img_path = os.path.join(input_dir, filename)
            with PIL.Image.open(img_path) as img:
                # Resize the image to the specified size
                img_resized = img.resize(size)
                # Save the resized image to the output directory
                img_resized.save(os.path.join(output_dir, filename))
                
                # create a flipped version of the image and save it
                img_flipped = img_resized.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                flipped_filename = f"flipped_{filename}"
                img_flipped.save(os.path.join(output_dir, flipped_filename))
                
                
                
                img.close()
                img_resized.close()
                img_flipped.close()
                
                

                if show_progress:
                    # Print a message indicating the image has been resized
                    print(f"Resized image saved to: {os.path.join(output_dir, filename)}")
if __name__ == "__main__":
    resize_images(INPUT_DIR, OUTPUT_DIR, show_progress=False)
    