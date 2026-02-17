"""Triple-panel 3D volume rendering for uncertainty, density, and error comparison.

Extends the side-by-side rendering to include a third panel for error visualization.
"""

import numpy as np
from typing import Optional, Tuple


def render_triple_volume_pyvista(
    volume: np.ndarray,
    scalar1: np.ndarray,
    scalar2: np.ndarray,
    scalar3: np.ndarray,
    save_path: Optional[str] = None,
    camera_elevation: float = 30.0,
    camera_azimuth: float = 45.0,
    zoom: float = 1.0,
    # Panel 1 (uncertainty)
    scalar1_cmap: str = 'Reds',
    scalar1_clim: Optional[Tuple[float, float]] = None,
    scalar1_opacity: float = 1.0,
    scalar1_opacity_mode: str = 'adaptive',
    scalar1_title: str = 'Uncertainty',
    scalar1_unit: str = '',
    # Panel 2 (density)
    scalar2_cmap: str = 'Greens',
    scalar2_clim: Optional[Tuple[float, float]] = None,
    scalar2_opacity: float = 0.8,
    scalar2_opacity_mode: str = 'linear',
    scalar2_title: str = 'Density',
    scalar2_unit: str = '',
    # Panel 3 (error)
    scalar3_cmap: str = 'hot',
    scalar3_clim: Optional[Tuple[float, float]] = None,
    scalar3_opacity: float = 1.0,
    scalar3_opacity_mode: str = 'adaptive',
    scalar3_title: str = 'Error',
    scalar3_unit: str = '',
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
    volume_unit: str = '',
    show_bounds: bool = False,
    bounds_color: str = 'black',
    bounds_width: float = 2.0,
    # Slice visualization
    show_slice: bool = False,
    slice_z: Optional[int] = None,
    slice_frame_color: str = 'darkblue',
    slice_frame_width: float = 3.0,
    window_size: Tuple[int, int] = (5760, 2120),
    return_image: bool = False,
) -> Optional[np.ndarray]:
    """Render three scalar fields in a row using PyVista (1x3 or 2x3 layout).

    Args:
        volume: (D, H, W) input volume (beads, shared by all panels)
        scalar1: (D, H, W) scalar field for panel 1 (e.g., uncertainty)
        scalar2: (D, H, W) scalar field for panel 2 (e.g., density)
        scalar3: (D, H, W) scalar field for panel 3 (e.g., error)
        save_path: output image path (required if return_image=False)
        camera_elevation: camera elevation angle (degrees)
        camera_azimuth: camera azimuth angle (degrees)
        zoom: zoom factor
        scalar1_cmap: colormap for panel 1
        scalar1_clim: (vmin, vmax) color range for panel 1
        scalar1_opacity: opacity scaling for panel 1 (0-1)
        scalar1_opacity_mode: 'adaptive' or 'linear' for panel 1
        scalar1_title: title for panel 1
        scalar1_unit: unit string for panel 1
        scalar2_cmap: colormap for panel 2
        scalar2_clim: (vmin, vmax) color range for panel 2
        scalar2_opacity: opacity scaling for panel 2 (0-1)
        scalar2_opacity_mode: 'adaptive' or 'linear' for panel 2
        scalar2_title: title for panel 2
        scalar2_unit: unit string for panel 2
        scalar3_cmap: colormap for panel 3
        scalar3_clim: (vmin, vmax) color range for panel 3
        scalar3_opacity: opacity scaling for panel 3 (0-1)
        scalar3_opacity_mode: 'adaptive' or 'linear' for panel 3
        scalar3_title: title for panel 3
        scalar3_unit: unit string for panel 3
        volume_opacity: opacity for input volume (beads) in all panels
        background_color: 'white', 'black', or RGB tuple
        show_axes: show coordinate axes
        show_scalar_bar: show color bars for scalar fields
        show_volume_scalar_bar: show color bar for input volume
        scalar_bar_vertical: if True, color bar is vertical
        scalar_bar_font_size: font size for color bar labels
        scalar_bar_font_family: font family ('arial', 'times', etc.)
        scalar_bar_outline: if True, draw outline around color bars
        volume_unit: unit string for volume scalar bar
        show_bounds: if True, show bounding box around volume
        bounds_color: color of bounding box lines
        bounds_width: line width of bounding box
        show_slice: if True, show 2D slices below 3D volumes
        slice_z: z-index for slice plane (None = middle)
        slice_frame_color: color of slice plane frame
        slice_frame_width: line width of slice frame
        window_size: (width, height) output resolution
        return_image: if True, return numpy array instead of saving

    Returns:
        np.ndarray (H, W, 3) if return_image=True, else None
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista is required for triple volume rendering. Install with: pip install pyvista")

    # Determine layout: 2x3 if show_slice, otherwise 1x3
    if show_slice:
        shape = (2, 3)  # 2 rows x 3 columns
        adjusted_window_size = (window_size[0], int(window_size[1] * 1.5))
    else:
        shape = (1, 3)  # 1 row x 3 columns
        adjusted_window_size = window_size

    plotter = pv.Plotter(shape=shape, window_size=adjusted_window_size, off_screen=True)
    plotter.set_background(background_color)

    # Prepare grid base
    grid_base = pv.ImageData(dimensions=volume.shape)
    D, H, W = volume.shape

    # Calculate camera position (same as side_by_side_render.py)
    center = np.array([(D - 1) / 2, (H - 1) / 2, (W - 1) / 2])
    max_dim = max(D, H, W)
    distance = max_dim * 2.5 / zoom

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

    # Scalar bar configuration
    scalar_bar_args = {
        'title_font_size': scalar_bar_font_size,
        'label_font_size': scalar_bar_font_size - 2,
        'font_family': scalar_bar_font_family,
        'vertical': scalar_bar_vertical,
        'outline': scalar_bar_outline,
        'color': 'black' if background_color == 'white' else 'white',
    }

    volume_scalar_bar_args = scalar_bar_args.copy()
    if volume_unit:
        volume_scalar_bar_args['title'] = f'Input ({volume_unit})'
    else:
        volume_scalar_bar_args['title'] = 'Input'

    # Function to add volume to a subplot
    def add_volume_panel(row, col, scalar_data, cmap, opacity, opacity_mode, title, unit, clim=None):
        plotter.subplot(row, col)

        grid = grid_base.copy()
        grid['scalar'] = scalar_data.ravel(order='F')
        if volume_opacity > 0:
            grid['input'] = volume.ravel(order='F')

        # Add scalar volume
        # Use PyVista's built-in opacity presets
        opacity_preset = 'sigmoid_10' if opacity_mode == 'adaptive' else 'linear'

        scalar_title = f'{title} ({unit})' if unit else title
        actor_scalar = plotter.add_volume(
            grid, scalars='scalar', cmap=cmap,
            opacity=opacity_preset,
            clim=clim,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={**scalar_bar_args, 'title': scalar_title}
        )
        actor_scalar.prop.opacity_unit_distance = 3.0 / max(opacity, 0.01)

        # Add input volume overlay (if opacity > 0)
        if volume_opacity > 0:
            if show_volume_scalar_bar:
                vol_actor = plotter.add_volume(
                    grid, scalars='input', cmap='gray',
                    opacity='linear', opacity_unit_distance=2.0, shade=True,
                    show_scalar_bar=True,
                    scalar_bar_args=volume_scalar_bar_args
                )
            else:
                vol_actor = plotter.add_volume(
                    grid, scalars='input', cmap='gray',
                    opacity='linear', opacity_unit_distance=2.0, shade=True
                )
            vol_actor.prop.opacity_unit_distance = 2.0 / max(volume_opacity, 0.01)

        # Set camera (same for all panels)
        plotter.camera_position = camera_position

        # Add slice frame if requested (frame in 3D volume showing slice location)
        if show_slice:
            slice_z_idx = slice_z if slice_z is not None else D // 2
            slice_z_idx = np.clip(slice_z_idx, 0, D - 1)

            # Create rectangle at z=slice_z_idx
            rect_points = np.array([
                [slice_z_idx, 0, 0],
                [slice_z_idx, H - 1, 0],
                [slice_z_idx, H - 1, W - 1],
                [slice_z_idx, 0, W - 1],
                [slice_z_idx, 0, 0],
            ], dtype=float)

            rect_mesh = pv.lines_from_points(rect_points)
            plotter.add_mesh(rect_mesh, color=slice_frame_color, line_width=slice_frame_width)

        # Add bounding box if requested
        if show_bounds:
            plotter.add_bounding_box(color=bounds_color, line_width=bounds_width)

        # Add axes if requested
        if show_axes:
            plotter.add_axes(
                xlabel='X', ylabel='Y', zlabel='Z',
                color='black' if background_color == 'white' else 'white',
                line_width=2
            )

    # Add 3D volume panels (row 0)
    add_volume_panel(0, 0, scalar1, scalar1_cmap, scalar1_opacity, scalar1_opacity_mode,
                     scalar1_title, scalar1_unit, scalar1_clim)
    add_volume_panel(0, 1, scalar2, scalar2_cmap, scalar2_opacity, scalar2_opacity_mode,
                     scalar2_title, scalar2_unit, scalar2_clim)
    add_volume_panel(0, 2, scalar3, scalar3_cmap, scalar3_opacity, scalar3_opacity_mode,
                     scalar3_title, scalar3_unit, scalar3_clim)

    # Add 2D slice panels (row 1) if enabled
    if show_slice:
        slice_z_idx = slice_z if slice_z is not None else D // 2
        slice_z_idx = np.clip(slice_z_idx, 0, D - 1)

        # Function to add 2D slice subplot
        def add_2d_slice(row, col, scalar_data, cmap, title, unit, clim=None):
            plotter.subplot(row, col)

            # Extract 2D slice
            slice_2d = scalar_data[slice_z_idx, :, :]

            # Create 2D mesh
            grid_2d = pv.ImageData(dimensions=(W, H, 1))
            grid_2d['scalar'] = slice_2d.ravel(order='F')

            # Add mesh with colormap
            slice_title = f'{title} (z={slice_z_idx})'
            if unit:
                slice_title += f' ({unit})'

            plotter.add_mesh(
                grid_2d,
                scalars='scalar',
                cmap=cmap,
                clim=clim,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={**scalar_bar_args, 'title': slice_title}
            )

            # Set 2D view (looking down at XY plane)
            plotter.view_xy()

            # Add bounding box for 2D slice
            if show_bounds:
                plotter.add_bounding_box(color=bounds_color, line_width=bounds_width)

        # Add 3 2D slice panels
        add_2d_slice(1, 0, scalar1, scalar1_cmap, scalar1_title, scalar1_unit, scalar1_clim)
        add_2d_slice(1, 1, scalar2, scalar2_cmap, scalar2_title, scalar2_unit, scalar2_clim)
        add_2d_slice(1, 2, scalar3, scalar3_cmap, scalar3_title, scalar3_unit, scalar3_clim)

    # Render
    if return_image:
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return image
    else:
        if save_path is None:
            raise ValueError("save_path must be specified when return_image=False")
        plotter.screenshot(save_path)
        plotter.close()
        return None
