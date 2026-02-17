"""Side-by-side 3D volume rendering for comparison.

Render uncertainty and density (or any two scalar fields) side-by-side
in a single image for direct visual comparison.
"""

import numpy as np
from typing import Optional, Tuple
import warnings


def render_side_by_side_pyvista(
    volume: np.ndarray,
    scalar_left: np.ndarray,
    scalar_right: np.ndarray,
    save_path: Optional[str] = None,
    camera_elevation: float = 30.0,
    camera_azimuth: float = 45.0,
    zoom: float = 1.0,
    # Left panel (uncertainty)
    left_cmap: str = 'Reds',
    left_clim: Optional[Tuple[float, float]] = None,
    left_opacity: float = 1.0,
    left_opacity_mode: str = 'adaptive',
    left_title: str = 'Uncertainty',
    # Right panel (density)
    right_cmap: str = 'Greens',
    right_clim: Optional[Tuple[float, float]] = None,
    right_opacity: float = 0.8,
    right_opacity_mode: str = 'linear',
    right_title: str = 'Density',
    # Common settings
    volume_opacity: float = 0.3,
    background_color: str = 'white',
    show_axes: bool = True,
    show_scalar_bar: bool = True,
    show_volume_scalar_bar: bool = False,
    scalar_bar_vertical: bool = True,
    scalar_bar_font_size: int = 12,
    scalar_bar_font_family: str = 'arial',
    scalar_bar_outline: bool = True,
    # Units for color bars
    volume_unit: str = '',
    left_unit: str = '',
    right_unit: str = '',
    show_bounds: bool = False,
    bounds_color: str = 'black',
    bounds_width: float = 2.0,
    # Slice visualization
    show_slice: bool = False,
    slice_z: Optional[int] = None,
    slice_frame_color: str = 'darkblue',
    slice_frame_width: float = 3.0,
    window_size: Tuple[int, int] = (2560, 1080),
    return_image: bool = False,
) -> Optional[np.ndarray]:
    """Render two scalar fields side-by-side using PyVista.

    Args:
        volume: (D, H, W) input volume (beads, shared by both panels)
        scalar_left: (D, H, W) scalar field for left panel (e.g., uncertainty)
        scalar_right: (D, H, W) scalar field for right panel (e.g., density)
        save_path: output image path (required if return_image=False)
        camera_elevation: camera elevation angle (degrees)
        camera_azimuth: camera azimuth angle (degrees)
        zoom: zoom factor
        left_cmap: colormap for left panel
        left_clim: (vmin, vmax) color range for left panel
        left_opacity: opacity scaling for left panel (0-1)
        left_opacity_mode: 'adaptive' or 'linear' for left panel
        left_title: title for left panel
        right_cmap: colormap for right panel
        right_clim: (vmin, vmax) color range for right panel
        right_opacity: opacity scaling for right panel (0-1)
        right_opacity_mode: 'adaptive' or 'linear' for right panel
        right_title: title for right panel
        volume_opacity: opacity for input volume (beads) in both panels
        background_color: 'white', 'black', or RGB tuple
        show_axes: show coordinate axes
        show_scalar_bar: show color bars for uncertainty/density
        show_volume_scalar_bar: show color bar for input volume (default: False)
        scalar_bar_vertical: if True, color bar is vertical (default: True)
        scalar_bar_font_size: font size for color bar labels (default: 12)
        scalar_bar_font_family: font family ('arial', 'times', etc., default: 'arial')
        scalar_bar_outline: if True, draw outline around color bars (default: True)
        volume_unit: unit string for volume scalar bar (e.g., 'a.u.')
        left_unit: unit string for left panel (e.g., 'voxels', 'µm')
        right_unit: unit string for right panel
        show_bounds: if True, show bounding box around volume (default: False)
        bounds_color: color of bounding box lines (default: 'black')
        bounds_width: line width of bounding box (default: 2.0)
        show_slice: if True, show 2D slices below 3D volumes (default: False)
        slice_z: z-index for slice plane (None = middle, default: None)
        slice_frame_color: color of slice plane frame (default: 'darkblue')
        slice_frame_width: line width of slice frame (default: 3.0)
        window_size: (width, height) output resolution
        return_image: if True, return numpy array instead of saving

    Returns:
        np.ndarray (H, W, 3) if return_image=True, else None

    Raises:
        ImportError: if PyVista not installed
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for side-by-side rendering. "
            "Install with: pip install pyvista"
        )

    # Create plotter with 1 row, 2 columns
    plotter = pv.Plotter(
        shape=(1, 2),
        off_screen=True,
        window_size=window_size,
        border=False
    )
    plotter.background_color = background_color

    # Compute camera position (shared by both panels)
    D, H, W = volume.shape
    center = (D / 2, H / 2, W / 2)
    base_distance = max(D, H, W) * 2.0
    distance = base_distance / zoom

    elev_rad = np.radians(camera_elevation)
    azim_rad = np.radians(camera_azimuth)

    cam_x = center[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
    cam_y = center[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
    cam_z = center[2] + distance * np.sin(elev_rad)

    camera_position = [
        (cam_x, cam_y, cam_z),
        center,
        (0, 0, 1)
    ]

    # === Left Panel (Uncertainty) ===
    plotter.subplot(0, 0)

    # Create grid for left panel
    grid_left = pv.ImageData()
    grid_left.dimensions = np.array(volume.shape) + 1
    grid_left.spacing = (1, 1, 1)
    grid_left.cell_data['input'] = volume.flatten(order='F')
    grid_left.cell_data['scalar'] = scalar_left.flatten(order='F')

    # Add input volume layer (only if volume_opacity > 0)
    if volume_opacity > 0 and show_volume_scalar_bar:
        # Configure volume scalar bar (if enabled)
        volume_title = f'Input Volume{" (" + volume_unit + ")" if volume_unit else ""}'
        volume_scalar_bar_args = {
            'title': volume_title,
            'vertical': scalar_bar_vertical,
            'title_font_size': scalar_bar_font_size,
            'label_font_size': scalar_bar_font_size - 2,
            'font_family': scalar_bar_font_family,
            'fmt': '%.2f',
            'outline': scalar_bar_outline,
        }
        if scalar_bar_vertical:
            volume_scalar_bar_args.update({
                'position_x': 0.05,
                'position_y': 0.1,
                'width': 0.08,
                'height': 0.8,
            })
        else:
            volume_scalar_bar_args.update({
                'position_x': 0.1,
                'position_y': 0.85,
                'width': 0.8,
                'height': 0.08,
            })
        vol_actor_left = plotter.add_volume(
            grid_left, scalars='input', cmap='gray',
            opacity='linear', opacity_unit_distance=2.0, shade=True,
            show_scalar_bar=True,
            scalar_bar_args=volume_scalar_bar_args
        )
        vol_actor_left.prop.opacity_unit_distance = 2.0 / max(volume_opacity, 0.01)
    elif volume_opacity > 0:
        # Add volume without scalar bar
        vol_actor_left = plotter.add_volume(
            grid_left, scalars='input', cmap='gray',
            opacity='linear', opacity_unit_distance=2.0, shade=True
        )
        vol_actor_left.prop.opacity_unit_distance = 2.0 / max(volume_opacity, 0.01)
    # else: volume_opacity <= 0, don't render input volume at all

    # Add scalar layer
    opacity_preset_left = 'sigmoid_10' if left_opacity_mode == 'adaptive' else 'linear'

    # Configure scalar bar for uncertainty
    if show_scalar_bar:
        left_title_with_unit = f'{left_title}{" (" + left_unit + ")" if left_unit else ""}'
        scalar_bar_args_left = {
            'title': left_title_with_unit,
            'vertical': scalar_bar_vertical,
            'title_font_size': scalar_bar_font_size,
            'label_font_size': scalar_bar_font_size - 2,
            'font_family': scalar_bar_font_family,
            'fmt': '%.2f',
            'outline': scalar_bar_outline,
        }
        if scalar_bar_vertical:
            # Vertical: right side of panel
            scalar_bar_args_left.update({
                'position_x': 0.85,
                'position_y': 0.1,
                'width': 0.08,
                'height': 0.8,
            })
        else:
            # Horizontal: bottom of panel
            scalar_bar_args_left.update({
                'position_x': 0.1,
                'position_y': 0.05,
                'width': 0.8,
                'height': 0.08,
            })
    else:
        scalar_bar_args_left = {}

    scalar_actor_left = plotter.add_volume(
        grid_left, scalars='scalar', cmap=left_cmap,
        opacity=opacity_preset_left,
        opacity_unit_distance=3.0,
        clim=left_clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=scalar_bar_args_left
    )
    scalar_actor_left.prop.opacity_unit_distance = 3.0 / max(left_opacity, 0.01)

    # Set camera
    plotter.camera_position = camera_position

    # Add slice frame if requested
    if show_slice:
        # Determine slice z position (default: middle)
        slice_z_idx = slice_z if slice_z is not None else D // 2
        slice_z_idx = np.clip(slice_z_idx, 0, D - 1)

        # Create rectangle at z=slice_z_idx (on volume surface)
        # Rectangle corners in (x, y, z) = (d, h, w) coordinates
        rect_points = np.array([
            [slice_z_idx, 0, 0],
            [slice_z_idx, H - 1, 0],
            [slice_z_idx, H - 1, W - 1],
            [slice_z_idx, 0, W - 1],
            [slice_z_idx, 0, 0],  # Close the loop
        ], dtype=float)

        # Create a continuous polyline (not line segments)
        import pyvista as pv
        rect_mesh = pv.lines_from_points(rect_points)
        plotter.add_mesh(rect_mesh, color=slice_frame_color, line_width=slice_frame_width)

    # Add bounding box if requested
    if show_bounds:
        plotter.add_bounding_box(
            color=bounds_color,
            line_width=bounds_width,
            opacity=1.0
        )

    # Add axes
    if show_axes:
        plotter.add_axes(
            xlabel='X', ylabel='Y', zlabel='Z',
            color='black' if background_color == 'white' else 'white',
            line_width=2
        )

    # Add title
    plotter.add_text(
        left_title, position='upper_edge', font_size=14,
        color='black' if background_color == 'white' else 'white'
    )

    # === Right Panel (Density) ===
    plotter.subplot(0, 1)

    # Create grid for right panel
    grid_right = pv.ImageData()
    grid_right.dimensions = np.array(volume.shape) + 1
    grid_right.spacing = (1, 1, 1)
    grid_right.cell_data['input'] = volume.flatten(order='F')
    grid_right.cell_data['scalar'] = scalar_right.flatten(order='F')

    # Add input volume layer (only if volume_opacity > 0)
    if volume_opacity > 0 and show_volume_scalar_bar:
        # Configure volume scalar bar (if enabled)
        volume_title = f'Input Volume{" (" + volume_unit + ")" if volume_unit else ""}'
        volume_scalar_bar_args_right = {
            'title': volume_title,
            'vertical': scalar_bar_vertical,
            'title_font_size': scalar_bar_font_size,
            'label_font_size': scalar_bar_font_size - 2,
            'font_family': scalar_bar_font_family,
            'fmt': '%.2f',
            'outline': scalar_bar_outline,
        }
        if scalar_bar_vertical:
            volume_scalar_bar_args_right.update({
                'position_x': 0.05,
                'position_y': 0.1,
                'width': 0.08,
                'height': 0.8,
            })
        else:
            volume_scalar_bar_args_right.update({
                'position_x': 0.1,
                'position_y': 0.85,
                'width': 0.8,
                'height': 0.08,
            })
        vol_actor_right = plotter.add_volume(
            grid_right, scalars='input', cmap='gray',
            opacity='linear', opacity_unit_distance=2.0, shade=True,
            show_scalar_bar=True,
            scalar_bar_args=volume_scalar_bar_args_right
        )
        vol_actor_right.prop.opacity_unit_distance = 2.0 / max(volume_opacity, 0.01)
    elif volume_opacity > 0:
        # Add volume without scalar bar
        vol_actor_right = plotter.add_volume(
            grid_right, scalars='input', cmap='gray',
            opacity='linear', opacity_unit_distance=2.0, shade=True
        )
        vol_actor_right.prop.opacity_unit_distance = 2.0 / max(volume_opacity, 0.01)
    # else: volume_opacity <= 0, don't render input volume at all

    # Add scalar layer
    opacity_preset_right = 'sigmoid_10' if right_opacity_mode == 'adaptive' else 'linear'

    # Configure scalar bar for density
    if show_scalar_bar:
        right_title_with_unit = f'{right_title}{" (" + right_unit + ")" if right_unit else ""}'
        scalar_bar_args_right = {
            'title': right_title_with_unit,
            'vertical': scalar_bar_vertical,
            'title_font_size': scalar_bar_font_size,
            'label_font_size': scalar_bar_font_size - 2,
            'font_family': scalar_bar_font_family,
            'fmt': '%.2f',
            'outline': scalar_bar_outline,
        }
        if scalar_bar_vertical:
            # Vertical: right side of panel
            scalar_bar_args_right.update({
                'position_x': 0.85,
                'position_y': 0.1,
                'width': 0.08,
                'height': 0.8,
            })
        else:
            # Horizontal: bottom of panel
            scalar_bar_args_right.update({
                'position_x': 0.1,
                'position_y': 0.05,
                'width': 0.8,
                'height': 0.08,
            })
    else:
        scalar_bar_args_right = {}

    scalar_actor_right = plotter.add_volume(
        grid_right, scalars='scalar', cmap=right_cmap,
        opacity=opacity_preset_right,
        opacity_unit_distance=3.0,
        clim=right_clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=scalar_bar_args_right
    )
    scalar_actor_right.prop.opacity_unit_distance = 3.0 / max(right_opacity, 0.01)

    # Set camera (same as left panel)
    plotter.camera_position = camera_position

    # Add slice frame if requested
    if show_slice:
        # Determine slice z position (same as left panel)
        slice_z_idx = slice_z if slice_z is not None else D // 2
        slice_z_idx = np.clip(slice_z_idx, 0, D - 1)

        # Create rectangle at z=slice_z_idx
        rect_points = np.array([
            [slice_z_idx, 0, 0],
            [slice_z_idx, H - 1, 0],
            [slice_z_idx, H - 1, W - 1],
            [slice_z_idx, 0, W - 1],
            [slice_z_idx, 0, 0],  # Close the loop
        ], dtype=float)

        # Create a continuous polyline (not line segments)
        import pyvista as pv
        rect_mesh = pv.lines_from_points(rect_points)
        plotter.add_mesh(rect_mesh, color=slice_frame_color, line_width=slice_frame_width)

    # Add bounding box if requested
    if show_bounds:
        plotter.add_bounding_box(
            color=bounds_color,
            line_width=bounds_width,
            opacity=1.0
        )

    # Add axes
    if show_axes:
        plotter.add_axes(
            xlabel='X', ylabel='Y', zlabel='Z',
            color='black' if background_color == 'white' else 'white',
            line_width=2
        )

    # Add title
    plotter.add_text(
        right_title, position='upper_edge', font_size=14,
        color='black' if background_color == 'white' else 'white'
    )

    # Render 3D part
    image_3d = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()

    # If show_slice is True, create 2D slice visualization and combine
    if show_slice:
        # Determine slice z position
        slice_z_idx = slice_z if slice_z is not None else D // 2
        slice_z_idx = np.clip(slice_z_idx, 0, D - 1)

        # Extract 2D slices
        slice_left_2d = scalar_left[slice_z_idx, :, :]
        slice_right_2d = scalar_right[slice_z_idx, :, :]

        # Create 2D slice visualization using matplotlib
        import matplotlib.pyplot as plt

        # Calculate figure size to match 3D width
        fig_width = window_size[0] / 100  # Convert pixels to inches (assuming 100 dpi)
        fig_height = window_size[1] / 100 / 2  # Half height for 2D slices
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor=background_color, dpi=100)

        # Left panel: 2D uncertainty
        ax_left = fig.add_subplot(1, 2, 1)
        im_left = ax_left.imshow(
            slice_left_2d, cmap=left_cmap,
            vmin=left_clim[0] if left_clim else None,
            vmax=left_clim[1] if left_clim else None,
            origin='lower', aspect='equal'
        )
        ax_left.set_title(f'{left_title} (z={slice_z_idx})',
                          fontsize=scalar_bar_font_size,
                          family=scalar_bar_font_family,
                          color='black' if background_color == 'white' else 'white')
        ax_left.set_xlabel('W', fontsize=scalar_bar_font_size - 2,
                           color='black' if background_color == 'white' else 'white')
        ax_left.set_ylabel('H', fontsize=scalar_bar_font_size - 2,
                           color='black' if background_color == 'white' else 'white')
        ax_left.tick_params(colors='black' if background_color == 'white' else 'white')

        if show_scalar_bar:
            cbar_left = plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
            left_title_with_unit = f'{left_title}{" (" + left_unit + ")" if left_unit else ""}'
            cbar_left.set_label(left_title_with_unit, fontsize=scalar_bar_font_size - 2,
                                family=scalar_bar_font_family,
                                color='black' if background_color == 'white' else 'white')
            cbar_left.ax.tick_params(labelsize=scalar_bar_font_size - 4,
                                      colors='black' if background_color == 'white' else 'white')

        # Right panel: 2D density
        ax_right = fig.add_subplot(1, 2, 2)
        im_right = ax_right.imshow(
            slice_right_2d, cmap=right_cmap,
            vmin=right_clim[0] if right_clim else None,
            vmax=right_clim[1] if right_clim else None,
            origin='lower', aspect='equal'
        )
        ax_right.set_title(f'{right_title} (z={slice_z_idx})',
                           fontsize=scalar_bar_font_size,
                           family=scalar_bar_font_family,
                           color='black' if background_color == 'white' else 'white')
        ax_right.set_xlabel('W', fontsize=scalar_bar_font_size - 2,
                            color='black' if background_color == 'white' else 'white')
        ax_right.set_ylabel('H', fontsize=scalar_bar_font_size - 2,
                            color='black' if background_color == 'white' else 'white')
        ax_right.tick_params(colors='black' if background_color == 'white' else 'white')

        if show_scalar_bar:
            cbar_right = plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
            right_title_with_unit = f'{right_title}{" (" + right_unit + ")" if right_unit else ""}'
            cbar_right.set_label(right_title_with_unit, fontsize=scalar_bar_font_size - 2,
                                 family=scalar_bar_font_family,
                                 color='black' if background_color == 'white' else 'white')
            cbar_right.ax.tick_params(labelsize=scalar_bar_font_size - 4,
                                       colors='black' if background_color == 'white' else 'white')

        plt.tight_layout()

        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        image_2d = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_2d = image_2d.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_2d = image_2d[:, :, :3]  # Drop alpha channel
        plt.close(fig)

        # Combine 3D and 2D images vertically
        # Resize 2D image width to match 3D if needed
        if image_2d.shape[1] != image_3d.shape[1]:
            from PIL import Image
            pil_image_2d = Image.fromarray(image_2d)
            pil_image_2d = pil_image_2d.resize((image_3d.shape[1], image_2d.shape[0]), Image.Resampling.LANCZOS)
            image_2d = np.array(pil_image_2d)

        # Vertical stack
        combined_image = np.vstack([image_3d, image_2d])

        # Save or return
        if return_image:
            return combined_image
        else:
            if save_path is None:
                raise ValueError("Either save_path or return_image=True must be specified")
            from PIL import Image
            Image.fromarray(combined_image).save(save_path)
            return None
    else:
        # No slice, just return 3D image
        if return_image:
            return image_3d
        else:
            if save_path is None:
                raise ValueError("Either save_path or return_image=True must be specified")
            from PIL import Image
            Image.fromarray(image_3d).save(save_path)
            return None


def render_side_by_side_matplotlib(
    volume: np.ndarray,
    scalar_left: np.ndarray,
    scalar_right: np.ndarray,
    left_cmap: str = 'Reds',
    right_cmap: str = 'Greens',
    left_title: str = 'Uncertainty',
    right_title: str = 'Density',
    view_elev: float = 30.0,
    view_azim: float = 45.0,
    threshold_percentile: float = 70.0,
    figsize: Tuple[int, int] = (16, 7),
    background_color: str = 'white',
):
    """Render two scalar fields side-by-side using Matplotlib.

    Args:
        volume: (D, H, W) input volume
        scalar_left: (D, H, W) left panel scalar
        scalar_right: (D, H, W) right panel scalar
        left_cmap: colormap for left panel
        right_cmap: colormap for right panel
        left_title: title for left panel
        right_title: title for right panel
        view_elev: camera elevation
        view_azim: camera azimuth
        threshold_percentile: voxel threshold percentile
        figsize: figure size (width, height)
        background_color: 'white' or 'black'

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=figsize, facecolor=background_color)

    # Normalize scalars
    scalar_left_norm = (scalar_left - scalar_left.min()) / (scalar_left.max() - scalar_left.min() + 1e-8)
    scalar_right_norm = (scalar_right - scalar_right.min()) / (scalar_right.max() - scalar_right.min() + 1e-8)

    # Get colormaps
    cmap_left_obj = cm.get_cmap(left_cmap)
    cmap_right_obj = cm.get_cmap(right_cmap)

    # === Left Panel ===
    ax_left = fig.add_subplot(1, 2, 1, projection='3d', facecolor=background_color)

    # Threshold and create mask
    threshold_value = np.percentile(scalar_left_norm, threshold_percentile)
    mask_left = scalar_left_norm > threshold_value

    # Fallback if no voxels
    if not mask_left.any():
        for p in [50, 30, 10, 0]:
            threshold_value = np.percentile(scalar_left_norm, p)
            mask_left = scalar_left_norm > threshold_value
            if mask_left.any():
                break

    # Create colors
    colors_left = np.zeros(scalar_left.shape + (4,))
    if mask_left.any():
        colors_left[mask_left] = cmap_left_obj(scalar_left_norm[mask_left])
        colors_left[mask_left, 3] = 0.7  # Alpha
        ax_left.voxels(mask_left, facecolors=colors_left, edgecolors=None)

    ax_left.view_init(elev=view_elev, azim=view_azim)
    ax_left.set_xlabel('X', fontsize=10)
    ax_left.set_ylabel('Y', fontsize=10)
    ax_left.set_zlabel('Z', fontsize=10)
    ax_left.set_title(left_title, fontsize=14, pad=10)

    # === Right Panel ===
    ax_right = fig.add_subplot(1, 2, 2, projection='3d', facecolor=background_color)

    # Threshold and create mask
    threshold_value = np.percentile(scalar_right_norm, threshold_percentile)
    mask_right = scalar_right_norm > threshold_value

    # Fallback if no voxels
    if not mask_right.any():
        for p in [50, 30, 10, 0]:
            threshold_value = np.percentile(scalar_right_norm, p)
            mask_right = scalar_right_norm > threshold_value
            if mask_right.any():
                break

    # Create colors
    colors_right = np.zeros(scalar_right.shape + (4,))
    if mask_right.any():
        colors_right[mask_right] = cmap_right_obj(scalar_right_norm[mask_right])
        colors_right[mask_right, 3] = 0.7  # Alpha
        ax_right.voxels(mask_right, facecolors=colors_right, edgecolors=None)

    ax_right.view_init(elev=view_elev, azim=view_azim)
    ax_right.set_xlabel('X', fontsize=10)
    ax_right.set_ylabel('Y', fontsize=10)
    ax_right.set_zlabel('Z', fontsize=10)
    ax_right.set_title(right_title, fontsize=14, pad=10)

    plt.tight_layout()
    return fig


def auto_render_side_by_side(
    volume: np.ndarray,
    scalar_left: np.ndarray,
    scalar_right: np.ndarray,
    backend: str = 'auto',
    **kwargs
):
    """Automatically render side-by-side comparison with best backend.

    Args:
        volume: (D, H, W) input volume
        scalar_left: (D, H, W) left panel scalar (e.g., uncertainty)
        scalar_right: (D, H, W) right panel scalar (e.g., density)
        backend: 'auto', 'pyvista', or 'matplotlib'
        **kwargs: backend-specific parameters

    PyVista kwargs:
        camera_elevation, camera_azimuth, zoom,
        left_cmap, left_clim, left_opacity, left_opacity_mode, left_title,
        right_cmap, right_clim, right_opacity, right_opacity_mode, right_title,
        volume_opacity, background_color, show_axes, show_scalar_bar,
        show_volume_scalar_bar, scalar_bar_vertical, scalar_bar_font_size,
        scalar_bar_font_family, scalar_bar_outline,
        volume_unit, left_unit, right_unit,
        show_bounds, bounds_color, bounds_width,
        show_slice, slice_z, slice_frame_color, slice_frame_width,
        window_size, save_path, return_image

    Matplotlib kwargs:
        view_elev, view_azim, left_cmap, right_cmap,
        left_title, right_title, threshold_percentile, figsize

    Returns:
        - matplotlib Figure if backend='matplotlib'
        - numpy array if backend='pyvista' and return_image=True
        - None if saved to file
    """
    if backend == 'auto':
        try:
            import pyvista as pv
            _ = pv.ImageData
            backend = 'pyvista'
        except (ImportError, AttributeError):
            backend = 'matplotlib'
            warnings.warn(
                "PyVista not available, using matplotlib backend. "
                "For better quality: pip install pyvista",
                UserWarning
            )

    if backend == 'pyvista':
        pyvista_params = {
            'save_path', 'camera_elevation', 'camera_azimuth', 'zoom',
            'left_cmap', 'left_clim', 'left_opacity', 'left_opacity_mode', 'left_title',
            'right_cmap', 'right_clim', 'right_opacity', 'right_opacity_mode', 'right_title',
            'volume_opacity', 'background_color', 'show_axes', 'show_scalar_bar',
            'show_volume_scalar_bar', 'scalar_bar_vertical', 'scalar_bar_font_size',
            'scalar_bar_font_family', 'scalar_bar_outline',
            'volume_unit', 'left_unit', 'right_unit',
            'show_bounds', 'bounds_color', 'bounds_width',
            'show_slice', 'slice_z', 'slice_frame_color', 'slice_frame_width',
            'window_size', 'return_image'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in pyvista_params}
        return render_side_by_side_pyvista(volume, scalar_left, scalar_right, **filtered_kwargs)
    else:  # matplotlib
        mpl_params = {
            'view_elev', 'view_azim', 'left_cmap', 'right_cmap',
            'left_title', 'right_title', 'threshold_percentile', 'figsize', 'background_color'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in mpl_params}
        return render_side_by_side_matplotlib(volume, scalar_left, scalar_right, **filtered_kwargs)
