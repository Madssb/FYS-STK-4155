import matplotlib.pyplot as plt
from initialize_features import read_image
from pathlib import Path
from utilities import my_figsize

if __name__ == "__main__":
    dir = Path.cwd()
    image_paths = sorted(dir.rglob("**/*.jpg"))
    image_path = image_paths[0]
    
    img_16x16 = read_image(image_path, image_size=(16, 16))
    img_32x32 = read_image(image_path, image_size=(32, 32))
    img_64x64 = read_image(image_path, image_size=(64, 64))
    img_128x128 = read_image(image_path, image_size=(128, 128))
    
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=my_figsize(column=False, subplots=(2,2)))
    
    # Plot the images in the subplots
    axs[0, 0].imshow(img_16x16, cmap='gray')
    axs[0, 1].imshow(img_32x32, cmap='gray')
    axs[1, 0].imshow(img_64x64, cmap='gray')
    axs[1, 1].imshow(img_128x128, cmap='gray')
    
    # Remove axes and add figure text with resolutions
    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    
    axs[0, 0].text(0.5, -0.1, '16x16', transform=axs[0, 0].transAxes, ha='center')
    axs[0, 1].text(0.5, -0.1, '32x32', transform=axs[0, 1].transAxes, ha='center')
    axs[1, 0].text(0.5, -0.1, '64x64', transform=axs[1, 0].transAxes, ha='center')
    axs[1, 1].text(0.5, -0.1, '128x128', transform=axs[1, 1].transAxes, ha='center')
    fig.tight_layout()
    plt.show()
