import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage

# Function to create an OffsetImage
def getImage(identifier, zoom=0.1):
    # Map identifiers to image paths, with a default image for unknown types
    image_map = {
        "en": "./icons/en.png",
    }
    default_image_path = "./icons/unk.png"  # Catch-all image

    # Use the mapped image or default image if the identifier is not found
    img_path = image_map.get(identifier, default_image_path)
    img = plt.imread(img_path)  # Load image
    return OffsetImage(img, zoom=zoom)  # Return OffsetImage object