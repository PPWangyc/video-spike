import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
import imageio
import cv2
import os
from matplotlib.animation import PillowWriter

def plot_embeddings_anim(embeddings, title=None, fps=30, outfile="embeddings_animation.mp4"):
    """
    Create an animation of embeddings over time and save as an MP4 without using ffmpeg directly.
    Steps:
    1. Animate using matplotlib and save as GIF (PillowWriter).
    2. Read GIF frames with imageio.
    3. Save frames as MP4 using OpenCV.
    4. Remove the temporary GIF.

    Parameters
    ----------
    embeddings : np.ndarray
        A 2D array of shape [Time, D], where each column is a feature dimension.
    title : str, optional
        Title for the figure.
    fps : int, optional
        Frames per second for the output video.
    outfile : str, optional
        The filename for the output mp4 video.
    """

    sns.set_theme(style="whitegrid")

    Time, D = embeddings.shape

    # Create subplots
    fig, axes = plt.subplots(nrows=D, ncols=1, figsize=(10, 2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    colors = sns.color_palette("husl", D)
    lines = []
    for i in range(D):
        line, = axes[i].plot([], [], color=colors[i], label=f"Dimension {i}")
        lines.append(line)
        axes[i].set_ylabel(f"D {i}")
        if i == 0 and title is not None:
            axes[i].set_title(title, fontsize=14)
        axes[i].legend(loc="upper right")
        # set y-limits
        axes[i].set_ylim(min(embeddings[:, i]), max(embeddings[:, i]))
        axes[i].set_xlim(0, Time)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(i):
        x = np.arange(i + 1)
        for d, line in enumerate(lines):
            line.set_data(x, embeddings[:i+1, d])
        return lines
    anim = animation.FuncAnimation(
        fig, 
        update, 
        init_func=init,
        frames=Time, 
        blit=True,
    )
    writer = PillowWriter(fps=fps)
    # Save to a temporary GIF
    temp_gif = outfile
    anim.save(temp_gif, 
              writer=writer)
    plt.close(fig)

    # Read the GIF frames using imageio
    # frames = imageio.mimread(temp_gif, memtest=False)
    # height, width, c = frames[0].shape
    # imageio.mimsave(outfile, frames, fps=fps)
    # # Remove the temporary GIF file
    # os.remove(temp_gif)

    # print(f"Animation saved as {outfile}")

def plot_embeddings(embeddings, title=None):
    """
    Plots embeddings over time, with one subplot per dimension.
    
    Parameters
    ----------
    embeddings : numpy.ndarray
        A 2D array of shape [Time, D], where each column is a feature dimension.
    title : str, optional
        A title for the entire figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    axes : array of matplotlib.axes.Axes
        The array of Axes objects.
    """
    # Optional: set a style for nicer aesthetics
    sns.set_theme(style="whitegrid")

    # Extract the shape of embeddings
    Time, D = embeddings.shape

    # Create subplots
    fig, axes = plt.subplots(nrows=D, ncols=1, figsize=(10, 2 * D), sharex=True)
    
    # If D == 1, `axes` is a single Axes object, convert to list for consistency
    if D == 1:
        axes = [axes]

    # Generate a distinct color for each dimension
    colors = sns.color_palette("husl", D)

    for i in range(D):
        ax = axes[i]
        ax.plot(
            embeddings[:, i], 
            color=colors[i],
            label=f"Dimension {i}")
        ax.set_ylabel(f"D {i}")
        if i == 0 and title is not None:
            ax.set_title(title)
        ax.legend(loc="upper right")

    # Set a common X-axis label
    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    
    # Don't show the plot, just return the figure so it can be saved remotely
    return fig, axes

def save_numpy_video_to_gif(video_array, filename="output.gif", fps=10):
    """
    Save a 4D numpy array of shape [T, 1, H, W] as an animated GIF.
    
    Parameters
    ----------
    video_array : np.ndarray
        A 4D numpy array of shape [T, 1, H, W]. 
        T is number of frames, and we have a single channel (C=1).
    filename : str
        The name of the GIF file to create.
    fps : int
        Frames per second for the GIF.
    """

    # Extract dimensions
    T, C, H, W = video_array.shape
    if C != 1:
        raise ValueError("Expected a single-channel video array of shape [T, 1, H, W].")

    # Convert frames to a list of 2D arrays
    # If the data isn't already uint8, you might need to scale or convert it.
    # For example, if it's floating-point in [0,1], multiply by 255:
    # frames = [(video_array[t, 0] * 255).astype(np.uint8) for t in range(T)]
    # If it's already in the right format, just directly use it:
    frames = [video_array[t, 0].astype(np.uint8) for t in range(T)]

    # Use imageio to save as a GIF and keep looping
    imageio.mimsave(filename, frames, fps=fps, loop=0)

def convertNumpy_video_to_gif(video_array, outfile="video.gif", fps=30, title=None):
    """
    Convert a numpy video array (T, 1, H, W) into an animated GIF.
    
    Parameters
    ----------
    video_array : np.ndarray
        A 4D numpy array of shape [T, 1, H, W], where T is the number of frames, 
        and we have a single channel (C=1).
    outfile : str, optional
        Filename for the output GIF.
    fps : int, optional
        Frames per second for the GIF.
    title : str, optional
        Title for the figure.
    """
    # Validate input shape
    if video_array.ndim != 4 or video_array.shape[1] != 1:
        raise ValueError("video_array must have shape [T, 1, H, W].")

    T, C, H, W = video_array.shape
    # For convenience, we know C=1, so we can ignore that dimension when displaying.

    # Normalize the frames if not already uint8
    if video_array.dtype != np.uint8:
        # no normalization if the array is float32
        if video_array.dtype == np.float32:
            video_array = float32_to_uint8(video_array)
        else:
            v_min, v_max = video_array.min(), video_array.max()
            if v_min == v_max:
                # Avoid division by zero if all frames are identical
                video_array = np.zeros_like(video_array, dtype=np.uint8)
            else:
                video_array = ((video_array - v_min) / (v_max - v_min) * 255).astype(np.uint8)

    # Setup the figure
    fig, ax = plt.subplots(figsize=(5, 5))
    im_display = ax.imshow(video_array[0, 0, :, :], cmap='gray', vmin=0, vmax=255, animated=True)
    ax.axis('off')

    if title is not None:
        ax.set_title(title)

    def init():
        im_display.set_data(video_array[0, 0, :, :])
        return [im_display]

    def update(i):
        im_display.set_data(video_array[i, 0, :, :])
        return [im_display]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=T,
        blit=True
    )

    writer = PillowWriter(fps=fps)
    anim.save(outfile, writer=writer)
    plt.close(fig)
    print(f"Saved GIF to {outfile}")

def float32_to_uint8(video_array,norm=False):
    """
    Convert a float32 video array into a uint8 representation, scaling values from [min, max] to [0, 255].
    
    Parameters
    ----------
    video_array : np.ndarray
        A numpy array of float32 values. The array can have any shape, but typically represents video data,
        e.g., [T, C, H, W] or [T, H, W, C]. The values can be any float range.
        
    Returns
    -------
    video_uint8 : np.ndarray
        The converted uint8 array with values scaled into the [0, 255] range.
        If the array is constant (min == max), it will return an array of all zeros.
    """
    if not isinstance(video_array, np.ndarray):
        raise TypeError("video_array must be a numpy array.")

    if video_array.dtype != np.float32:
        raise ValueError("video_array must be of dtype float32.")

    v_min, v_max = video_array.min(), video_array.max()
    if v_min == v_max:
        # If all frames are identical, just return zeros
        video_uint8 = np.zeros_like(video_array, dtype=np.uint8)
    else:
        # Scale the array values to [0, 255]
        if norm:
            video_uint8 = (255 * (video_array - v_min) / (v_max - v_min)).astype(np.uint8)
        else:
            print('here is the right one')
            video_uint8 = (video_array * 255).astype(np.uint8)

    return video_uint8

# Example usage:
# if __name__ == "__main__":
#     Time, D = 100, 5
#     dummy_embeddings = np.random.randn(Time, D)
#     fig, axes = plot_embeddings(dummy_embeddings, title="Embeddings Over Time")
#     fig.savefig("embeddings_plot.png", dpi=300)
#     plot_embeddings_anim(dummy_embeddings, title="Embedding Animation", fps=30, outfile="embeddings_animation.gif")
#     # Create a dummy video: 50 frames, single channel, 64x64
#     T, C, H, W = 50, 1, 64, 64
#     dummy_video = np.random.rand(T, C, H, W)
#     convertNumpy_video_to_gif(dummy_video, outfile="random_video.gif", fps=30, title="Random Video")