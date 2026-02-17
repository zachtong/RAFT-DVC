"""3D volume rendering utilities.

Provides multiple backends for 3D visualization:
- Matplotlib (lightweight, always available)
- PyVista (high-quality, optional dependency)

Design principles:
- All functions accept numpy arrays only
- No direct dependency on PyTorch or TensorFlow
- Graceful degradation if optional deps missing
- Return matplotlib Figure or save to file
- Lazy imports: dependencies only loaded when functions are called
"""

import numpy as np
from typing import Optional, Tuple, Union
import warnings


def compute_feature_density(
    volume: np.ndarray,
    threshold: float = 0.1,
    sigma: float = 3.0,
    min_density: float = 0.0,
    max_density: float = 1.0,
) -> np.ndarray:
    """Compute feature density map from input volume.

    This is a deterministic, training-free metric of local feature
    presence. High density indicates reliable regions for flow estimation.

    Args:
        volume: (D, H, W) input volume (normalized to [0, 1] range)
        threshold: intensity threshold for "feature" (default: 0.1)
        sigma: Gaussian blur sigma for density spread (default: 3.0 voxels)
        min_density: minimum output density value (default: 0.0)
        max_density: maximum output density value (default: 1.0)

    Returns:
        density_map: (D, H, W) feature density in [min_density, max_density]

    Algorithm:
        1. Binary mask: volume > threshold
        2. Gaussian blur (separable 3D convolution)
        3. Normalize to [0, 1]
        4. Scale to [min_density, max_density]
    """
    from scipy.ndimage import gaussian_filter

    # Step 1: Binary feature mask
    feature_mask = (volume > threshold).astype(np.float32)

    # Step 2: Gaussian blur (separable for efficiency)
    density = gaussian_filter(
        feature_mask, sigma=sigma, mode='constant', cval=0.0
    )

    # Step 3: Normalize to [0, 1]
    if density.max() > 0:
        density = density / density.max()

    # Step 4: Scale to [min_density, max_density]
    density = density * (max_density - min_density) + min_density

    return density


def render_uncertainty_volume_mpl(
    volume: np.ndarray,
    uncertainty: np.ndarray,
    feature_density: Optional[np.ndarray] = None,
    threshold_percentile: float = 70.0,
    view_elev: float = 30.0,
    view_azim: float = 45.0,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Uncertainty Volume (3D)',
):
    """Render 3D uncertainty volume using Matplotlib voxels.

    Lightweight backend, always available, but slower for large volumes.
    Recommended for volumes <= 64^3.

    Args:
        volume: (D, H, W) input volume
        uncertainty: (D, H, W) uncertainty map
        feature_density: (D, H, W) optional density map (green overlay)
        threshold_percentile: only show voxels above this (0-100)
        view_elev: camera elevation angle in degrees
        view_azim: camera azimuth angle in degrees
        figsize: figure size (width, height) in inches
        title: figure title

    Returns:
        fig: matplotlib Figure object

    Note:
        For volumes > 64^3, this may be slow. Use
        render_uncertainty_volume_pyvista for better performance.
    """
    # Lazy import
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Downsample warning if too large
    max_dim = max(volume.shape)
    if max_dim > 64:
        warnings.warn(
            f"Volume dimension {max_dim} > 64 may be slow with "
            "matplotlib. Consider using "
            "render_uncertainty_volume_pyvista() or downsampling.",
            UserWarning
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Normalize uncertainty to [0, 1]
    unc_norm = (uncertainty - uncertainty.min()) / \
               (uncertainty.max() - uncertainty.min() + 1e-8)

    # Threshold: only show high-uncertainty regions
    # Auto-adjust threshold if it would result in no voxels
    threshold_value = np.percentile(unc_norm, threshold_percentile)
    high_unc_mask = unc_norm > threshold_value

    # If no voxels after thresholding, lower threshold adaptively
    if not high_unc_mask.any():
        # Try progressively lower thresholds
        for fallback_percentile in [50, 30, 10, 0]:
            threshold_value = np.percentile(unc_norm, fallback_percentile)
            high_unc_mask = unc_norm > threshold_value
            if high_unc_mask.any():
                warnings.warn(
                    f"threshold_percentile={threshold_percentile} resulted in "
                    f"no voxels, using {fallback_percentile}th percentile instead",
                    UserWarning
                )
                break

    # If still no voxels (uniform uncertainty), show all above median
    if not high_unc_mask.any():
        high_unc_mask = unc_norm >= np.median(unc_norm)
        warnings.warn(
            "Uncertainty is very uniform, showing all above-median voxels",
            UserWarning
        )

    # RGBA colors: red for uncertainty
    colors = np.zeros((*uncertainty.shape, 4), dtype=np.float32)
    colors[high_unc_mask, 0] = 1.0  # Red channel
    # Alpha (scaled for visibility)
    colors[high_unc_mask, 3] = unc_norm[high_unc_mask] * 0.8

    # Optional: overlay feature density in green
    if feature_density is not None:
        density_norm = (feature_density - feature_density.min()) / \
                       (feature_density.max() - feature_density.min() + 1e-8)
        high_density_mask = density_norm > 0.5
        # Green overlay (lower alpha to not obscure red)
        colors[high_density_mask, 1] = density_norm[high_density_mask] * 0.5
        colors[high_density_mask, 3] = np.maximum(
            colors[high_density_mask, 3],
            density_norm[high_density_mask] * 0.3
        )

    # Voxel rendering (only if we have voxels to render)
    if high_unc_mask.any():
        # Note: edgecolors=None to avoid edge rendering (more efficient)
        ax.voxels(high_unc_mask, facecolors=colors, edgecolors=None)
    else:
        # Fallback: show wireframe box
        ax.text(
            0.5, 0.5, 0.5, 'No voxels to render\n(uniform uncertainty)',
            transform=ax.transAxes, ha='center', va='center', fontsize=12
        )

    # Camera setup
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='High Uncertainty')
    ]
    if feature_density is not None:
        legend_elements.append(
            Patch(facecolor='green', alpha=0.5, label='High Density')
        )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    return fig


def render_uncertainty_volume_pyvista(
    volume: np.ndarray,
    uncertainty: np.ndarray,
    feature_density: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    camera_position: Optional[Tuple] = None,
    camera_elevation: Optional[float] = None,
    camera_azimuth: Optional[float] = None,
    zoom: float = 1.0,
    window_size: Tuple[int, int] = (1920, 1080),
    opacity_mode: str = 'adaptive',
    return_image: bool = False,
    uncertainty_cmap: str = 'Reds',
    density_cmap: str = 'Greens',
    uncertainty_clim: Optional[Tuple[float, float]] = None,
    density_clim: Optional[Tuple[float, float]] = None,
    volume_opacity: float = 0.3,
    uncertainty_opacity: float = 1.0,
    density_opacity: float = 0.8,
    show_scalar_bar: bool = False,
    scalar_bar_args: Optional[dict] = None,
    background_color: str = 'white',
    show_axes: bool = True,
    title: str = 'Uncertainty Volume Rendering',
) -> Optional[np.ndarray]:
    """Render high-quality 3D uncertainty volume using PyVista.

    Professional volume rendering with extensive customization options.
    Requires: pip install pyvista

    Args:
        volume: (D, H, W) input volume (beads)
        uncertainty: (D, H, W) uncertainty map (or any scalar field to visualize)
        feature_density: (D, H, W) optional density map
        save_path: if provided, save image to this path
        camera_position: optional custom camera position (advanced)
                        [(camera_xyz), (focal_xyz), (up_vector)]
                        If provided, overrides elevation/azimuth/zoom
        camera_elevation: camera elevation angle in degrees (0-90)
        camera_azimuth: camera azimuth angle in degrees (0-360)
        zoom: zoom factor (>1 zooms in, <1 zooms out)
        window_size: (width, height) in pixels
        opacity_mode: 'adaptive' (sigmoid) or 'linear' for uncertainty layer
        return_image: if True, return numpy array (H, W, 3)
        uncertainty_cmap: colormap for uncertainty layer (e.g., 'Reds', 'hot', 'viridis')
        density_cmap: colormap for density layer (e.g., 'Greens', 'Blues')
        uncertainty_clim: (vmin, vmax) color range for uncertainty, None = auto
        density_clim: (vmin, vmax) color range for density, None = auto
        volume_opacity: overall opacity for input volume layer (0-1)
        uncertainty_opacity: opacity scaling for uncertainty layer (0-1)
        density_opacity: opacity scaling for density layer (0-1)
        show_scalar_bar: whether to show color bar
        scalar_bar_args: dict of scalar bar kwargs (e.g., {'title': 'Uncertainty'})
        background_color: background color ('white', 'black', or RGB tuple)
        show_axes: whether to show coordinate axes
        title: title text (empty string to hide)

    Returns:
        image_array: (H, W, 3) RGB image if return_image=True, else None

    Raises:
        ImportError: if pyvista is not installed
    """
    # Lazy import
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for high-quality volume rendering. "
            "Install with: pip install pyvista"
        )

    # Create PyVista ImageData grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1  # +1 for cell data
    grid.spacing = (1, 1, 1)

    # Add scalar fields (flatten in Fortran order for VTK compatibility)
    grid.cell_data['input'] = volume.flatten(order='F')
    grid.cell_data['uncertainty'] = uncertainty.flatten(order='F')
    if feature_density is not None:
        grid.cell_data['density'] = feature_density.flatten(order='F')

    # Create plotter with background color
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = background_color

    # Layer 1: Input volume (gray, semi-transparent)
    # Opacity scaled by volume_opacity parameter
    volume_actor = plotter.add_volume(
        grid, scalars='input', cmap='gray',
        opacity='linear',
        opacity_unit_distance=2.0,
        shade=True
    )
    # Scale overall opacity
    volume_actor.prop.opacity_unit_distance = 2.0 / max(volume_opacity, 0.01)

    # Layer 2: Uncertainty (customizable colormap and color range)
    if opacity_mode == 'adaptive':
        opacity_unc = 'sigmoid_10'  # Emphasize high values
    else:
        opacity_unc = 'linear'

    unc_actor = plotter.add_volume(
        grid, scalars='uncertainty', cmap=uncertainty_cmap,
        opacity=opacity_unc,
        opacity_unit_distance=3.0,
        clim=uncertainty_clim,  # Color range
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=scalar_bar_args if scalar_bar_args else {}
    )
    # Scale overall opacity
    unc_actor.prop.opacity_unit_distance = 3.0 / max(uncertainty_opacity, 0.01)

    # Layer 3: Feature density (optional, customizable colormap)
    if feature_density is not None:
        density_actor = plotter.add_volume(
            grid, scalars='density', cmap=density_cmap,
            opacity='linear',
            opacity_unit_distance=3.0,
            clim=density_clim,  # Color range
        )
        # Scale overall opacity
        density_actor.prop.opacity_unit_distance = 3.0 / max(density_opacity, 0.01)

    # Camera setup
    D, H, W = volume.shape
    center = (D / 2, H / 2, W / 2)

    if camera_position is not None:
        # Use explicit camera position (advanced mode)
        plotter.camera_position = camera_position
    else:
        # Use elevation/azimuth angles (simpler mode)
        if camera_elevation is None:
            camera_elevation = 30.0  # Default elevation
        if camera_azimuth is None:
            camera_azimuth = 45.0  # Default azimuth

        # Convert spherical coordinates to Cartesian
        # Distance from center (before zoom)
        base_distance = max(D, H, W) * 2.0
        distance = base_distance / zoom  # Apply zoom

        # Convert degrees to radians
        elev_rad = np.radians(camera_elevation)
        azim_rad = np.radians(camera_azimuth)

        # Spherical to Cartesian (physics convention: elevation from XY plane)
        cam_x = center[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
        cam_y = center[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
        cam_z = center[2] + distance * np.sin(elev_rad)

        plotter.camera_position = [
            (cam_x, cam_y, cam_z),  # camera position
            center,                  # focal point
            (0, 0, 1)               # up vector
        ]

    # Add axes (optional)
    if show_axes:
        plotter.add_axes(
            xlabel='X', ylabel='Y', zlabel='Z',
            color='black' if background_color == 'white' else 'white',
            line_width=2
        )

    # Add title (if not empty)
    if title:
        plotter.add_text(
            title,
            position='upper_edge',
            font_size=14,
            color='black' if background_color == 'white' else 'white'
        )

    # Render
    if return_image:
        image = plotter.screenshot(
            return_img=True, transparent_background=False
        )
        plotter.close()
        return image
    else:
        if save_path is None:
            raise ValueError(
                "Either save_path or return_image=True must be specified"
            )
        plotter.screenshot(save_path, transparent_background=False)
        plotter.close()
        return None


def auto_render_uncertainty_volume(
    volume: np.ndarray,
    uncertainty: np.ndarray,
    feature_density: Optional[np.ndarray] = None,
    backend: str = 'auto',
    **kwargs
) -> Union[object, np.ndarray, None]:
    """Automatically select best available backend for 3D rendering.

    Args:
        volume: (D, H, W) input volume
        uncertainty: (D, H, W) uncertainty map
        feature_density: (D, H, W) optional density map
        backend: 'auto', 'matplotlib', or 'pyvista'
        **kwargs: backend-specific parameters

    Common kwargs:
        - title (str): Figure title
        - save_path (str): Path to save image (pyvista only)
        - return_image (bool): Return image array (pyvista only)

    Matplotlib-specific kwargs:
        - view_elev (float): Camera elevation angle
        - view_azim (float): Camera azimuth angle
        - threshold_percentile (float): Voxel threshold percentile
        - figsize (tuple): Figure size

    PyVista-specific kwargs (extensive customization):
        Camera control:
        - camera_position (tuple): Explicit [(cam_xyz), (focal_xyz), (up)] (advanced)
        - camera_elevation (float): Elevation angle in degrees (0-90)
        - camera_azimuth (float): Azimuth angle in degrees (0-360)
        - zoom (float): Zoom factor (>1 zooms in, <1 zooms out)

        Colormaps and ranges:
        - uncertainty_cmap (str): Uncertainty colormap ('Reds', 'hot', 'viridis', etc.)
        - density_cmap (str): Density colormap ('Greens', 'Blues', etc.)
        - uncertainty_clim (tuple): (vmin, vmax) for uncertainty color range
        - density_clim (tuple): (vmin, vmax) for density color range

        Opacity control:
        - opacity_mode (str): 'adaptive' (sigmoid) or 'linear' for uncertainty layer
        - volume_opacity (float): Input volume layer opacity (0-1)
        - uncertainty_opacity (float): Uncertainty layer opacity scaling (0-1)
        - density_opacity (float): Density layer opacity scaling (0-1)

        Visual settings:
        - show_scalar_bar (bool): Show color bar
        - scalar_bar_args (dict): Scalar bar customization (e.g., {'title': 'Value'})
        - background_color (str): 'white', 'black', or RGB tuple
        - show_axes (bool): Show coordinate axes
        - window_size (tuple): Output resolution (width, height)

    Returns:
        - matplotlib Figure if backend='matplotlib'
        - numpy image array if backend='pyvista' and return_image=True
        - None if saved to file

    Backend selection (if 'auto'):
        1. Try PyVista (best quality)
        2. Fallback to Matplotlib (always available)
    """
    if backend == 'auto':
        try:
            import pyvista as pv
            # Check if key components are available (detect broken installations)
            _ = pv.ImageData
            backend = 'pyvista'
        except (ImportError, AttributeError) as e:
            backend = 'matplotlib'
            error_type = type(e).__name__
            if error_type == 'ImportError':
                msg = ("PyVista not installed, using matplotlib backend. "
                       "For better quality: pip install pyvista")
            else:
                msg = ("PyVista installation is broken, using matplotlib backend. "
                       "Fix with: pip uninstall pyvista && pip install pyvista")
            warnings.warn(msg, UserWarning)

    # Filter kwargs for each backend
    if backend == 'pyvista':
        # PyVista-specific parameters (all supported params)
        pyvista_params = {
            'save_path', 'camera_position', 'camera_elevation', 'camera_azimuth',
            'zoom', 'window_size', 'opacity_mode', 'return_image',
            'uncertainty_cmap', 'density_cmap',
            'uncertainty_clim', 'density_clim',
            'volume_opacity', 'uncertainty_opacity', 'density_opacity',
            'show_scalar_bar', 'scalar_bar_args',
            'background_color', 'show_axes', 'title'
        }
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in pyvista_params
        }
        return render_uncertainty_volume_pyvista(
            volume, uncertainty, feature_density, **filtered_kwargs
        )
    elif backend == 'matplotlib':
        # Matplotlib-specific parameters
        mpl_params = {
            'threshold_percentile', 'view_elev', 'view_azim',
            'figsize', 'title'
        }
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in mpl_params
        }
        return render_uncertainty_volume_mpl(
            volume, uncertainty, feature_density, **filtered_kwargs
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Use 'auto', 'matplotlib', or 'pyvista'"
        )
