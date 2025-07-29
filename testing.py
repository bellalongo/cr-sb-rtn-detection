#!/usr/bin/env python3
"""
H2RG Cosmic Ray Detection Script

Detects cosmic ray hits in H2RG infrared detector data across two exposures.
Cosmic rays should appear in one exposure but not persist in the next exposure
at the same location.

Features:
- Multi-exposure cosmic ray validation
- Reference pixel correction
- Temporal persistence analysis
- Cross-exposure verification
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from astropy.io import fits
from scipy import ndimage
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CosmicRayConfig:
    """Configuration parameters for cosmic ray detection."""
    # Detection thresholds
    sigma_threshold: float = 5.0
    min_intensity: float = 50.0
    min_anomaly_pixels: int = 3
    max_spatial_extent: int = 20
    
    # Persistence requirements
    persistence_threshold: float = 0.8  # Fraction of remaining frames
    early_frame_threshold: int = 5      # Ignore events in first N frames
    
    # Cross-exposure validation
    validation_region_size: int = 3     # Region around cosmic ray to check
    
    # Reference pixel correction
    reference_pixel_window: int = 4

class CosmicRayDetector:
    """H2RG Cosmic Ray Detection Class."""
    
    def __init__(self, config: CosmicRayConfig = None, verbose: bool = True):
        """Initialize the cosmic ray detector."""
        self.config = config or CosmicRayConfig()
        self.verbose = verbose
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(level=log_level, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def detect_cosmic_rays(self, exposure1_file: str, exposure2_file: str, 
                          output_dir: str = "results") -> Dict:
        """
        Detect cosmic rays across two exposures.
        
        Args:
            exposure1_file: Path to first exposure FITS file
            exposure2_file: Path to second exposure FITS file
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing cosmic ray detection results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting cosmic ray detection across two exposures")
        self.logger.info(f"Exposure 1: {exposure1_file}")
        self.logger.info(f"Exposure 2: {exposure2_file}")

        from os.path import exists

        print(exists(exposure1_file))
        print(exists(exposure2_file))
        
        # Load and preprocess both exposures
        frames1 = self._load_fits_data(exposure1_file)
        frames2 = self._load_fits_data(exposure2_file)
        
        # Apply reference pixel correction
        corrected_frames1 = self._reference_pixel_correction(frames1)
        corrected_frames2 = self._reference_pixel_correction(frames2)
        
        # Combine frames for 900-frame analysis as requested
        combined_frames = np.concatenate([corrected_frames1, corrected_frames2], axis=0)
        self.logger.info(f"Combined frame stack shape: {combined_frames.shape}")
        
        # Perform temporal analysis on first exposure
        temporal_data1 = self._temporal_analysis(corrected_frames1)
        
        # Detect cosmic ray candidates in first exposure
        cosmic_ray_candidates = self._detect_cosmic_ray_candidates(temporal_data1, corrected_frames1)
        
        # Validate candidates against second exposure
        validated_cosmic_rays = self._validate_against_second_exposure(
            cosmic_ray_candidates, corrected_frames2
        )
        
        # Compile results
        results = {
            'exposure1_file': exposure1_file,
            'exposure2_file': exposure2_file,
            'total_candidates': len(cosmic_ray_candidates),
            'validated_cosmic_rays': len(validated_cosmic_rays),
            'validation_rate': len(validated_cosmic_rays) / len(cosmic_ray_candidates) if cosmic_ray_candidates else 0,
            'cosmic_rays': validated_cosmic_rays,
            'processing_timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        # Save results and create visualizations
        self._save_results(results, output_path)
        self._create_visualizations(results, temporal_data1, corrected_frames1, output_path)
        
        self.logger.info(f"Detection complete: {len(cosmic_ray_candidates)} candidates -> "
                        f"{len(validated_cosmic_rays)} validated cosmic rays")
        
        return results
    
    def _load_fits_data(self, file_path: str) -> np.ndarray:
        """Load FITS data."""
        file_path = str(file_path)
        print(file_path)
        self.logger.info(f"Loading {file_path}")
        with fits.open(file_path) as hdul:
            print(hdul)
            data = hdul[0].data.astype(np.float32)
        
        self.logger.info(f"Loaded data shape: {data.shape}")
        return data
    
    def _reference_pixel_correction(self, frames: np.ndarray) -> np.ndarray:
        """Apply reference pixel correction using edge pixels."""
        self.logger.info("Applying reference pixel correction")
        
        corrected = frames.copy()
        n_frames, height, width = frames.shape
        window = self.config.reference_pixel_window
        
        for i in range(n_frames):
            frame = frames[i]
            
            # Calculate median reference values from edges
            ref_top = np.median(frame[:window, :])
            ref_bottom = np.median(frame[-window:, :])
            ref_left = np.median(frame[:, :window])
            ref_right = np.median(frame[:, -window:])
            
            # Average reference correction
            ref_correction = (ref_top + ref_bottom + ref_left + ref_right) / 4
            
            # Apply correction
            corrected[i] = frame - ref_correction
            
        return corrected
    
    def _temporal_analysis(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform temporal analysis on frame sequence."""
        self.logger.info("Performing temporal analysis")
        
        n_frames, height, width = frames.shape
        
        # Calculate frame differences
        diff_frames = np.diff(frames, axis=0)
        
        # Calculate noise statistics
        noise_std = np.std(diff_frames, axis=0)
        
        # Initialize analysis arrays
        first_appearance = np.full((height, width), -1, dtype=np.int32)
        persistence_count = np.zeros((height, width), dtype=np.int32)
        max_intensity = np.zeros((height, width), dtype=np.float32)
        
        # Detection threshold based on noise
        threshold = self.config.sigma_threshold * noise_std
        
        # Analyze each pixel's temporal behavior
        for y in range(height):
            for x in range(width):
                diff_series = diff_frames[:, y, x]
                
                # Find significant events
                significant_events = np.abs(diff_series) > threshold[y, x]
                event_frames = np.where(significant_events)[0]
                
                if len(event_frames) > 0:
                    first_appearance[y, x] = event_frames[0] + 1  # +1 because diff is offset
                    persistence_count[y, x] = n_frames - first_appearance[y, x]
                    max_intensity[y, x] = np.max(np.abs(diff_series[significant_events]))
        
        return {
            'first_appearance': first_appearance,
            'persistence_count': persistence_count,
            'max_intensity': max_intensity,
            'frames': frames,
            'diff_frames': diff_frames,
            'noise_std': noise_std,
            'threshold': threshold
        }
    
    def _detect_cosmic_ray_candidates(self, temporal_data: Dict, frames: np.ndarray) -> List[Dict]:
        """Detect cosmic ray candidates in single exposure."""
        self.logger.info("Detecting cosmic ray candidates")

        print(temporal_data)
        
        first_appearance = temporal_data['first_appearance']
        persistence = temporal_data['persistence_count']
        max_intensity = temporal_data['max_intensity']
        n_frames = frames.shape[0]
        
        # Cosmic ray criteria
        # 1. Appears suddenly (not in first few frames)
        sudden_appearance = first_appearance > self.config.early_frame_threshold
        
        # 2. High persistence (continues for most remaining frames)
        min_persistence = (n_frames - first_appearance) * self.config.persistence_threshold
        high_persistence = persistence >= min_persistence
        
        # 3. High intensity
        high_intensity = max_intensity > self.config.min_intensity
        
        # Combine all criteria
        cosmic_ray_mask = sudden_appearance & high_persistence & high_intensity
        
        # Find connected components
        labeled, num_features = ndimage.label(cosmic_ray_mask)
        
        candidates = []
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            pixel_count = np.sum(component_mask)
            
            # Filter by size
            if pixel_count < self.config.min_anomaly_pixels:
                continue
            if pixel_count > self.config.max_spatial_extent:
                continue
            
            # Get component properties
            y_coords, x_coords = np.where(component_mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Extract temporal properties
            first_frame = int(np.min(first_appearance[component_mask]))
            mean_intensity = np.mean(max_intensity[component_mask])
            max_component_intensity = np.max(max_intensity[component_mask])
            mean_persistence = np.mean(persistence[component_mask])
            
            # Calculate temporal profile
            temporal_profile = []
            for frame_idx in range(first_frame, n_frames):
                frame_intensity = np.mean(frames[frame_idx, y_coords, x_coords])
                temporal_profile.append(frame_intensity)
            
            candidate = {
                'id': i,
                'centroid': (centroid_y, centroid_x),
                'pixel_coords': list(zip(y_coords, x_coords)),
                'pixel_count': pixel_count,
                'first_frame': first_frame,
                'mean_intensity': mean_intensity,
                'max_intensity': max_component_intensity,
                'persistence_frames': int(mean_persistence),
                'temporal_profile': temporal_profile,
                'validated': False
            }
            
            candidates.append(candidate)
        
        self.logger.info(f"Found {len(candidates)} cosmic ray candidates")
        return candidates
    
    def _validate_against_second_exposure(self, candidates: List[Dict], 
                                        frames2: np.ndarray) -> List[Dict]:
        """
        Validate cosmic ray candidates against second exposure.
        True cosmic rays should not appear in the same location in the second exposure.
        """
        self.logger.info("Validating cosmic ray candidates against second exposure")
        
        # Perform temporal analysis on second exposure
        temporal_data2 = self._temporal_analysis(frames2)
        
        validated_cosmic_rays = []
        region_size = self.config.validation_region_size
        
        for candidate in candidates:
            centroid = candidate['centroid']
            cy, cx = int(centroid[0]), int(centroid[1])
            
            # Define region around cosmic ray location
            y_start = max(0, cy - region_size)
            y_end = min(temporal_data2['first_appearance'].shape[0], cy + region_size + 1)
            x_start = max(0, cx - region_size)
            x_end = min(temporal_data2['first_appearance'].shape[1], cx + region_size + 1)
            
            # Check for significant events in this region in second exposure
            region_first_app = temporal_data2['first_appearance'][y_start:y_end, x_start:x_end]
            region_max_int = temporal_data2['max_intensity'][y_start:y_end, x_start:x_end]
            
            # Look for significant anomalies in the same region
            significant_events = (region_first_app > 0) & (region_max_int > self.config.min_intensity)
            
            # If no significant events in second exposure, this is likely a true cosmic ray
            if not np.any(significant_events):
                candidate['validated'] = True
                validated_cosmic_rays.append(candidate)
                self.logger.debug(f"Validated cosmic ray at {centroid}")
            else:
                self.logger.debug(f"Rejected candidate at {centroid} - appears in second exposure")
        
        self.logger.info(f"Validated {len(validated_cosmic_rays)} out of {len(candidates)} candidates")
        return validated_cosmic_rays
    
    def _save_results(self, results: Dict, output_path: Path):
        """Save results to HDF5 file."""
        results_file = output_path / "cosmic_ray_detection_results.h5"
        
        self.logger.info(f"Saving results to {results_file}")
        
        with h5py.File(results_file, 'w') as f:
            # Save summary
            summary_grp = f.create_group('summary')
            summary_grp.attrs['exposure1_file'] = results['exposure1_file']
            summary_grp.attrs['exposure2_file'] = results['exposure2_file']
            summary_grp.attrs['total_candidates'] = results['total_candidates']
            summary_grp.attrs['validated_cosmic_rays'] = results['validated_cosmic_rays']
            summary_grp.attrs['validation_rate'] = results['validation_rate']
            summary_grp.attrs['processing_timestamp'] = results['processing_timestamp']
            
            # Save configuration
            config_grp = f.create_group('configuration')
            for key, value in results['config'].items():
                config_grp.attrs[key] = value
            
            # Save cosmic ray details
            if results['cosmic_rays']:
                cr_grp = f.create_group('cosmic_rays')
                for i, cr in enumerate(results['cosmic_rays']):
                    cr_subgrp = cr_grp.create_group(f'cosmic_ray_{i}')
                    cr_subgrp.attrs['id'] = cr['id']
                    cr_subgrp.attrs['centroid'] = cr['centroid']
                    cr_subgrp.attrs['pixel_count'] = cr['pixel_count']
                    cr_subgrp.attrs['first_frame'] = cr['first_frame']
                    cr_subgrp.attrs['mean_intensity'] = cr['mean_intensity']
                    cr_subgrp.attrs['max_intensity'] = cr['max_intensity']
                    cr_subgrp.attrs['persistence_frames'] = cr['persistence_frames']
                    cr_subgrp.create_dataset('pixel_coords', data=np.array(cr['pixel_coords']))
                    cr_subgrp.create_dataset('temporal_profile', data=np.array(cr['temporal_profile']))
    
    def _create_visualizations(self, results: Dict, temporal_data: Dict, 
                             frames: np.ndarray, output_path: Path):
        """Create visualization plots."""
        self.logger.info("Creating visualizations")
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('H2RG Cosmic Ray Detection Analysis', fontsize=16)
        
        # 1. First appearance map
        ax = axes[0, 0]
        im1 = ax.imshow(temporal_data['first_appearance'], cmap='viridis', aspect='auto')
        ax.set_title('First Appearance Frame')
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        plt.colorbar(im1, ax=ax, label='Frame Number')
        
        # Overlay cosmic ray locations
        if results['cosmic_rays']:
            cr_y = [cr['centroid'][0] for cr in results['cosmic_rays']]
            cr_x = [cr['centroid'][1] for cr in results['cosmic_rays']]
            ax.scatter(cr_x, cr_y, c='red', marker='o', s=50, alpha=0.8, label='Cosmic Rays')
            ax.legend()
        
        # 2. Max intensity map
        ax = axes[0, 1]
        im2 = ax.imshow(temporal_data['max_intensity'], cmap='hot', aspect='auto')
        ax.set_title('Maximum Intensity')
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        plt.colorbar(im2, ax=ax, label='Intensity (DN)')
        
        # 3. Persistence count map
        ax = axes[0, 2]
        im3 = ax.imshow(temporal_data['persistence_count'], cmap='plasma', aspect='auto')
        ax.set_title('Persistence Count')
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        plt.colorbar(im3, ax=ax, label='Number of Frames')
        
        # 4. Cosmic ray properties scatter plot
        ax = axes[1, 0]
        if results['cosmic_rays']:
            intensities = [cr['max_intensity'] for cr in results['cosmic_rays']]
            first_frames = [cr['first_frame'] for cr in results['cosmic_rays']]
            
            ax.scatter(first_frames, intensities, alpha=0.7, s=60)
            ax.set_xlabel('First Appearance Frame')
            ax.set_ylabel('Max Intensity (DN)')
            ax.set_title(f'Cosmic Ray Properties (n={len(results["cosmic_rays"])})')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No cosmic rays detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Cosmic Ray Properties')
        
        # 5. Temporal evolution example
        ax = axes[1, 1]
        if results['cosmic_rays']:
            # Show temporal profile of first cosmic ray
            cr = results['cosmic_rays'][0]
            frames_axis = range(cr['first_frame'], cr['first_frame'] + len(cr['temporal_profile']))
            ax.plot(frames_axis, cr['temporal_profile'], 'b-', linewidth=2, marker='o', markersize=4)
            ax.axvline(cr['first_frame'], color='red', linestyle='--', alpha=0.7, label='First Appearance')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Mean Intensity (DN)')
            ax.set_title(f'Example Cosmic Ray Temporal Profile\n(ID: {cr["id"]}, Location: {cr["centroid"]})')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No cosmic rays to show', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Temporal Profile Example')
        
        # 6. Detection statistics
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        stats_data = [
            ['Total Candidates', results['total_candidates']],
            ['Validated Cosmic Rays', results['validated_cosmic_rays']],
            ['Validation Rate', f"{results['validation_rate']*100:.1f}%"],
            ['Sigma Threshold', self.config.sigma_threshold],
            ['Min Intensity', f"{self.config.min_intensity} DN"],
            ['Persistence Threshold', f"{self.config.persistence_threshold*100:.0f}%"]
        ]
        
        if results['cosmic_rays']:
            avg_intensity = np.mean([cr['max_intensity'] for cr in results['cosmic_rays']])
            avg_size = np.mean([cr['pixel_count'] for cr in results['cosmic_rays']])
            stats_data.extend([
                ['Avg Intensity', f"{avg_intensity:.1f} DN"],
                ['Avg Size', f"{avg_size:.1f} pixels"]
            ])
        
        table = ax.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Detection Statistics')
        
        plt.tight_layout()
        plt.savefig(output_path / 'cosmic_ray_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary plot
        self._plot_summary(results, output_path)
    
    def _plot_summary(self, results: Dict, output_path: Path):
        """Create summary plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Cosmic Ray Detection Summary', fontsize=14)
        
        # Validation pie chart
        ax = axes[0]
        if results['total_candidates'] > 0:
            validated = results['validated_cosmic_rays']
            rejected = results['total_candidates'] - validated
            
            labels = ['Validated Cosmic Rays', 'Rejected Candidates']
            sizes = [validated, rejected]
            colors = ['lightcoral', 'lightblue']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Validation Results\n(Total: {results["total_candidates"]} candidates)')
        else:
            ax.text(0.5, 0.5, 'No candidates detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('No Cosmic Ray Candidates')
        
        # Intensity distribution
        ax = axes[1]
        if results['cosmic_rays']:
            intensities = [cr['max_intensity'] for cr in results['cosmic_rays']]
            ax.hist(intensities, bins=min(10, len(intensities)), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Max Intensity (DN)')
            ax.set_ylabel('Count')
            ax.set_title(f'Cosmic Ray Intensity Distribution\n(n={len(results["cosmic_rays"])})')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No cosmic rays detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Intensity Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'cosmic_ray_summary.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='H2RG Cosmic Ray Detection')
    parser.add_argument('exposure1', help='First exposure FITS file')
    parser.add_argument('exposure2', help='Second exposure FITS file')
    parser.add_argument('-o', '--output', default='results', help='Output directory')
    parser.add_argument('--sigma', type=float, default=5.0, help='Detection threshold (sigma)')
    parser.add_argument('--min-intensity', type=float, default=50.0, 
                       help='Minimum cosmic ray intensity (DN)')
    parser.add_argument('--persistence', type=float, default=0.8, 
                       help='Persistence threshold (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure detector
    config = CosmicRayConfig(
        sigma_threshold=args.sigma,
        min_intensity=args.min_intensity,
        persistence_threshold=args.persistence
    )
    
    # Initialize detector
    detector = CosmicRayDetector(config=config, verbose=args.verbose)
    
    # Detect cosmic rays
    print(f"Detecting cosmic rays between:")
    print(f"  Exposure 1: {args.exposure1}")
    print(f"  Exposure 2: {args.exposure2}")
    
    results = detector.detect_cosmic_rays(args.exposure1, args.exposure2, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("COSMIC RAY DETECTION SUMMARY")
    print("="*60)
    print(f"Total candidates found: {results['total_candidates']}")
    print(f"Validated cosmic rays: {results['validated_cosmic_rays']}")
    print(f"Validation rate: {results['validation_rate']*100:.1f}%")
    
    if results['cosmic_rays']:
        avg_intensity = np.mean([cr['max_intensity'] for cr in results['cosmic_rays']])
        avg_size = np.mean([cr['pixel_count'] for cr in results['cosmic_rays']])
        print(f"Average intensity: {avg_intensity:.1f} DN")
        print(f"Average size: {avg_size:.1f} pixels")
    
    print(f"\nResults saved to: {args.output}/")
    print("="*60)


if __name__ == "__main__":
    main()