"""
Imaging simulation: Point Spread Function (PSF) and noise modeling.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class ImagingSimulator:
    """Simulate confocal microscopy imaging effects."""

    def __init__(self, config):
        """
        Args:
            config: Configuration dict for imaging parameters
        """
        self.config = config

    def apply_imaging_effects(self, volume, apply_photobleaching=False):
        """
        Apply full imaging pipeline: PSF + noise + background.

        Args:
            volume: (D, H, W) clean volume
            apply_photobleaching: If True, reduce intensity (for vol1)

        Returns:
            imaged_volume: (D, H, W) volume with imaging effects
            metadata: dict with applied parameters
        """
        # 1. Apply PSF (blurring)
        psf_sigma = np.random.uniform(self.config['psf_sigma_min'],
                                      self.config['psf_sigma_max'])
        volume_blurred = self._apply_psf(volume, psf_sigma)

        # 2. Apply photobleaching (optional, for vol1)
        if apply_photobleaching and self.config.get('photobleaching_enabled', False):
            bleach_factor = np.random.uniform(
                self.config['photobleaching_factor_min'],
                self.config['photobleaching_factor_max']
            )
            volume_blurred *= bleach_factor
        else:
            bleach_factor = 1.0

        # 3. Add background
        background = self._generate_background(volume.shape)
        volume_with_bg = volume_blurred + background

        # 4. Add noise
        snr_db = np.random.uniform(self.config['snr_db_min'],
                                   self.config['snr_db_max'])
        volume_noisy = self._add_noise(volume_with_bg, snr_db)

        # 5. Clip and normalize to [0, 1]
        volume_final = np.clip(volume_noisy, 0, 1)

        metadata = {
            'psf_sigma': float(psf_sigma),
            'snr_db': float(snr_db),
            'photobleaching_factor': float(bleach_factor),
            'background_mean': float(self.config['background_mean'])
        }

        return volume_final, metadata

    def _apply_psf(self, volume, sigma):
        """
        Apply Point Spread Function (simplified as Gaussian blur).

        Args:
            volume: (D, H, W) input volume
            sigma: PSF width (in voxels)

        Returns:
            blurred: (D, H, W) blurred volume
        """
        if self.config['psf_type'] == 'gaussian':
            blurred = gaussian_filter(volume, sigma=sigma, mode='nearest')
        else:
            # Could implement more realistic PSF (Airy disk) here
            blurred = gaussian_filter(volume, sigma=sigma, mode='nearest')

        return blurred

    def _generate_background(self, shape):
        """
        Generate background intensity (smooth + random).

        Args:
            shape: (D, H, W)

        Returns:
            background: (D, H, W) background volume
        """
        # Random smooth background
        bg_mean = self.config['background_mean']
        bg_std = self.config['background_std']

        # Low-frequency background variation
        background_lowfreq = np.random.randn(*shape) * bg_std
        background_lowfreq = gaussian_filter(background_lowfreq, sigma=10)

        # Add uniform offset
        background = background_lowfreq + bg_mean

        return background.astype(np.float32)

    def _add_noise(self, volume, snr_db):
        """
        Add realistic noise: Poisson (photon counting) + Gaussian (readout).

        Args:
            volume: (D, H, W) input volume (signal)
            snr_db: Signal-to-noise ratio in dB

        Returns:
            noisy: (D, H, W) noisy volume
        """
        if self.config['noise_type'] == 'poisson_gaussian':
            # Convert SNR from dB to linear scale
            snr_linear = 10 ** (snr_db / 10)

            # Signal power
            signal_power = np.mean(volume ** 2)

            # Noise power
            noise_power = signal_power / snr_linear

            # Poisson noise (signal-dependent)
            # Scale signal to reasonable photon count range
            photon_scale = 1000  # arbitrary scaling factor
            volume_scaled = volume * photon_scale
            volume_poisson = np.random.poisson(volume_scaled) / photon_scale

            # Gaussian readout noise (signal-independent)
            gaussian_noise_std = np.sqrt(noise_power * 0.3)  # 30% Gaussian contribution
            gaussian_noise = np.random.randn(*volume.shape) * gaussian_noise_std

            # Combined
            noisy = volume_poisson + gaussian_noise

        elif self.config['noise_type'] == 'gaussian':
            # Simple Gaussian noise (for debugging)
            snr_linear = 10 ** (snr_db / 10)
            signal_power = np.mean(volume ** 2)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power)

            noise = np.random.randn(*volume.shape) * noise_std
            noisy = volume + noise

        else:
            noisy = volume

        return noisy.astype(np.float32)


def compute_snr(signal, noise):
    """
    Compute actual SNR of a volume.

    Args:
        signal: (D, H, W) clean signal
        noise: (D, H, W) noise (or noisy - signal)

    Returns:
        snr_db: SNR in decibels
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power > 0:
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
    else:
        snr_db = np.inf

    return float(snr_db)


def compute_imaging_statistics(volume_clean, volume_noisy):
    """
    Compute statistics comparing clean and noisy volumes.

    Args:
        volume_clean: (D, H, W) clean volume
        volume_noisy: (D, H, W) noisy volume

    Returns:
        stats: dict with statistics
    """
    # Signal statistics
    signal_mean = float(np.mean(volume_clean))
    signal_std = float(np.std(volume_clean))

    # Noise statistics
    noise = volume_noisy - volume_clean
    noise_std = float(np.std(noise))

    # Actual SNR
    snr_db = compute_snr(volume_clean, noise)

    # Intensity range
    intensity_min = float(np.min(volume_noisy))
    intensity_max = float(np.max(volume_noisy))

    stats = {
        'signal_mean': signal_mean,
        'signal_std': signal_std,
        'noise_std': noise_std,
        'snr_db_actual': snr_db,
        'intensity_min': intensity_min,
        'intensity_max': intensity_max
    }

    return stats
