from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

def customIcon(x, y, ax, icon):
    image = OffsetImage(icon, zoom=0.05)  # Adjust the zoom level as needed
    for (xi, yi) in zip(x, y):
        ab = AnnotationBbox(image, (xi, yi), frameon=False)
        ax.add_artist(ab)