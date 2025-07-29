from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DetectorConfig:    
    # General detection parameters
    sigma_threshold: float = 5.0  # Detection threshold in sigma
    min_anomaly_pixels: int = 1   # Minimum pixels for valid anomaly
    
    # # Cosmic ray parameters
    # cosmic_ray_min_intensity: float = 50.0
    # cosmic_ray_max_spatial_extent: int = 20  # pixels
    # cosmic_ray_persistence_threshold: float = 0.8  # 80% of remaining frames
    cosmic_ray_min_intensity: float = 30.0  # Lower threshold
    cosmic_ray_max_spatial_extent: int = 20
    cosmic_ray_persistence_threshold: float = 0.8
    cosmic_ray_max_transitions: int = 15  # More permissive
    cosmic_ray_min_step: float = 20.0  # Reasonable step
    
    # Snowball parameters
    snowball_min_intensity: float = 30.0
    snowball_max_intensity: float = 500.0
    snowball_min_radius: int = 3
    snowball_max_radius: int = 15
    snowball_circularity_threshold: float = 0.7
    snowball_expansion_rate: float = 0.1  # pixels per frame
    
    # Telegraph noise parameters
    rtn_min_transitions: int = 2
    rtn_amplitude_range: tuple = (10.0, 300.0)  # e-
    rtn_state_stability_threshold: float = 0.7
    rtn_frequency_range: tuple = (0.001, 0.5)  # Hz
    
    # Hot pixel parameters
    hot_pixel_persistence_threshold: float = 0.95  # 95% of all frames
    hot_pixel_cross_exposure_threshold: float = 0.8  # 80% of exposures
    hot_pixel_intensity_variation: float = 0.2  # 20% variation allowed
    
    # Processing parameters
    reference_pixel_window_x: int = 64
    reference_pixel_window_y: int = 4
    temporal_analysis_chunk_size: int = 50  # frames
    
    # Visualization parameters
    plot_dpi: int = 150
    colormap: str = 'viridis'
    anomaly_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.anomaly_colors is None:
            self.anomaly_colors = {
                'cosmic_ray': 'red',
                'snowball': 'blue',
                'telegraph_noise': 'green',
                'hot_pixel': 'yellow',
                'unknown': 'gray'
            }

    min_exposures_for_classification: int = 3

@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""
    
    # File handling
    data_root: str = "/Volumes/jwst/ilongo/raw_data/18220_Euclid_SCA"
    output_root: str = "/projects/JWST_planets/ilongo/new_script/processed_data"
    cache_enabled: bool = True
    
    # Performance
    n_workers: int = 4
    batch_size: int = 10
    use_gpu: bool = False
    
    # Data parameters
    detector_shape: tuple = (2048, 2048)
    frames_per_exposure: int = 450

    # Debug options
    verbose: bool = True
    save_intermediate: bool = False
    test_mode: bool = False
    test_frames: int = 50