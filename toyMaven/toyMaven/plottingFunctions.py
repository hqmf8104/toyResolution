import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


def initialize_plot(groundX, groundY):
    """
    Initialize the figure and axis for plotting.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, groundX)  # Set x-axis limits
    ax.set_ylim(0, groundY)  # Set y-axis limits
    ax.invert_yaxis()
    ax.grid(True)

    # position window in top left
    # Retrieve the current figure manager
    manager = plt.get_current_fig_manager()

    # Set the window position to top-left corner (0, 0)
    manager.window.wm_geometry("+0+0")

    return fig, ax

def get_image(identifier, image_map, default_image_path, zoom=0.05):
    """
    Fetch the image associated with an identifier.
    """
    img_path = image_map.get(identifier, default_image_path)  # Fallback to default image
    img = plt.imread(img_path)  # Load image
    return OffsetImage(img, zoom=zoom)  # Return OffsetImage object

def update_plot(ax, agedtrackedEntities, image_map, default_image_path):
    """
    Update the plot with the latest entity data.

    Parameters:
    - ax: Matplotlib axis object.
    - agedtrackedEntities: Dictionary of entities to plot.
    - image_map: Dictionary mapping identifiers to image paths.
    - default_image_path: Path to the default image for unknown types.
    """
    # Remove each artist from the axis
    for artist in ax.get_children():
        # Check for removable artist types
        if isinstance(artist, (AnnotationBbox)):
            artist.remove()

    # Remove previous text annotations to avoid duplication
    for artist in ax.texts:
        artist.remove()

    # Loop through entities to update their positions or add them if new
    for key, ent in agedtrackedEntities.items():
        # Extract details from the entity
        x, y = ent[0]  # Coordinates
        identifier = ent[3]  # Type/identifier
        annotationText = f"{ent[0]}"  # Annotation text

        # add shapes to plot
        ab = AnnotationBbox(get_image(identifier, image_map, default_image_path), (x, y), frameon=False)
        ax.add_artist(ab)  # Add to the plot

        # Add text annotation near the plot point
        ax.text(
            x, y, annotationText, fontsize=8, ha="left", va="bottom"
        )  # Customize font, color, alignment, etc.


    plt.pause(0.1)  # Pause for smooth updates
    return ax
