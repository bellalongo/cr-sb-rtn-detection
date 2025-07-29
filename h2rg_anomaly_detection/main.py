import argparse
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
import h5py
from astropy.io import fits
import os
import sys
from datetime import datetime

from .config import DetectorConfig, ProcessingConfig
from .preprocessing import ReferencePixelCorrector, FrameDifferencer, TemporalAnalyzer
from .classifier import AnomalyClassifier
from .visualization import PlotManager

class H2RGAnomalyDetector:
    """Main class for H2RG anomaly detection pipeline."""
    
    def __init__(self, detector_config: DetectorConfig = None, 
                 processing_config: ProcessingConfig = None):
        """Initialize the anomaly detector."""
        self.detector_config = detector_config or DetectorConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.reference_corrector = ReferencePixelCorrector(
            x_window=self.detector_config.reference_pixel_window_x,
            y_window=self.detector_config.reference_pixel_window_y
        )
        self.frame_differencer = FrameDifferencer(self.reference_corrector)
        self.temporal_analyzer = TemporalAnalyzer(
            sigma_threshold=self.detector_config.sigma_threshold
        )
        self.classifier = AnomalyClassifier(self.detector_config)
        self.plot_manager = PlotManager(self.detector_config)
        
        self.logger.info("H2RG Anomaly Detector initialized")
    
    def _setup_logging(self):
        """Configure logging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO if self.processing_config.verbose else logging.WARNING,
            format=log_format
        )
        self.logger = logging.getLogger(__name__)
    
    def process_exposures(self, file_paths: List[str], 
                         output_dir: Optional[str] = None) -> Dict:
        """Process multiple exposures for anomaly detection."""
        self.logger.info(f"Processing {len(file_paths)} exposures")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(self.processing_config.output_root)
        
        # Check if we have cached results
        temporal_data_list = []
        diff_stack_list = []
        
        for i, file_path in enumerate(file_paths):
            self.logger.info(f"Processing exposure {i+1}/{len(file_paths)}: {file_path}")
            
            # Check for cached preprocessed data
            file_name = Path(file_path).stem
            cache_dir = output_path / 'cache' / file_name
            
            if self.processing_config.cache_enabled and cache_dir.exists():
                # Load from cache
                self.logger.info(f"Loading cached data for {file_name}")
                temporal_data, diff_stack = self._load_cached_data(cache_dir)
            else:
                # Process the exposure
                temporal_data, diff_stack = self._process_single_exposure(file_path)
                
                # Save to cache if enabled
                if self.processing_config.cache_enabled:
                    self._save_cached_data(cache_dir, temporal_data, diff_stack)
            
            temporal_data_list.append(temporal_data)
            diff_stack_list.append(diff_stack)
        
        # Classify anomalies
        if len(file_paths) == 1:
            # Single exposure classification
            results = self.classifier.classify_single_exposure(
                temporal_data_list[0], diff_stack_list[0]
            )
            results['file_path'] = file_paths[0]
            
            # Create visualization
            if output_dir:
                plot_path = output_path / f"{Path(file_paths[0]).stem}_anomalies.png"
                self.plot_manager.plot_exposure_summary(
                    results, diff_stack_list[0], temporal_data_list[0], 
                    save_path=str(plot_path)
                )
        else:
            # Multiple exposure classification
            results = self.classifier.classify_multiple_exposures(
                temporal_data_list, diff_stack_list
            )
            results['file_paths'] = file_paths
            
            # Create visualizations
            if output_dir:
                # Individual exposure plots
                for i, exposure_result in enumerate(results['exposures']):
                    plot_path = output_path / f"exposure_{i}_anomalies.png"
                    self.plot_manager.plot_exposure_summary(
                        exposure_result, diff_stack_list[i], temporal_data_list[i],
                        save_path=str(plot_path)
                    )
                
                # Summary plot
                summary_path = output_path / "multi_exposure_summary.png"
                self.plot_manager.plot_multi_exposure_summary(
                    results, save_path=str(summary_path)
                )
        
        # Save results
        if output_dir:
            results_path = output_path / "anomaly_detection_results.h5"
            self._save_results(results_path, results)
        
        return results
    
    def _process_single_exposure(self, file_path: str) -> tuple:
        """Process a single exposure file."""
        # Compute frame differences
        diff_data = self.frame_differencer.compute_differences(
            file_path, apply_correction=True
        )
        
        # Analyze temporal patterns
        temporal_data = self.temporal_analyzer.analyze_temporal_patterns(
            diff_data['differences']
        )
        
        return temporal_data, diff_data['differences']
    
    def _load_cached_data(self, cache_dir: Path) -> tuple:
        """Load cached preprocessed data."""
        temporal_file = cache_dir / 'temporal_data.h5'
        diff_file = cache_dir / 'diff_stack.h5'
        
        with h5py.File(temporal_file, 'r') as f:
            temporal_data = {
                'first_appearance': f['first_appearance'][:],
                'persistence_count': f['persistence_count'][:],
                'max_intensity': f['max_intensity'][:],
                'intensity_variance': f['intensity_variance'][:],
                'transition_count': f['transition_count'][:],
                'threshold_used': f.attrs['threshold_used'],
                'background_stats': dict(f['background_stats'].attrs),
                'temporal_evolution': []
            }
            
            # Load temporal evolution
            if 'temporal_evolution' in f:
                for i in range(len(f['temporal_evolution'])):
                    temporal_data['temporal_evolution'].append(
                        dict(f['temporal_evolution'][str(i)].attrs)
                    )
        
        with h5py.File(diff_file, 'r') as f:
            diff_stack = f['differences'][:]
        
        return temporal_data, diff_stack
    
    def _save_cached_data(self, cache_dir: Path, temporal_data: Dict, 
                         diff_stack: np.ndarray):
        """Save preprocessed data to cache."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save temporal data
        temporal_file = cache_dir / 'temporal_data.h5'
        with h5py.File(temporal_file, 'w') as f:
            f.create_dataset('first_appearance', data=temporal_data['first_appearance'])
            f.create_dataset('persistence_count', data=temporal_data['persistence_count'])
            f.create_dataset('max_intensity', data=temporal_data['max_intensity'])
            f.create_dataset('intensity_variance', data=temporal_data.get('intensity_variance', np.zeros_like(temporal_data['max_intensity'])))
            f.create_dataset('transition_count', data=temporal_data.get('transition_count', np.zeros_like(temporal_data['max_intensity'])))
            f.attrs['threshold_used'] = temporal_data['threshold_used']
            
            # Save background stats
            bg_grp = f.create_group('background_stats')
            for key, value in temporal_data['background_stats'].items():
                bg_grp.attrs[key] = value
            
            # Save temporal evolution
            if 'temporal_evolution' in temporal_data:
                evo_grp = f.create_group('temporal_evolution')
                for i, frame_data in enumerate(temporal_data['temporal_evolution']):
                    frame_grp = evo_grp.create_group(str(i))
                    for key, value in frame_data.items():
                        frame_grp.attrs[key] = value
        
        # Save difference stack
        diff_file = cache_dir / 'diff_stack.h5'
        with h5py.File(diff_file, 'w') as f:
            f.create_dataset('differences', data=diff_stack, 
                           compression='lzf', chunks=(1, 512, 512))
    
    def _save_results(self, results_path: Path, results: Dict):
        """Save detection results to HDF5 file."""
        with h5py.File(results_path, 'w') as f:
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['detector_config'] = str(self.detector_config)
            
            # Save summary statistics
            if 'summary' in results:
                summary_grp = f.create_group('summary')
                for key, value in results['summary'].items():
                    summary_grp.attrs[key] = value
            
            # Save individual anomalies
            if 'cosmic_rays' in results:
                self._save_anomaly_list(f, 'cosmic_rays', results['cosmic_rays'])
            if 'snowballs' in results:
                self._save_anomaly_list(f, 'snowballs', results['snowballs'])
            if 'telegraph_noise' in results:
                self._save_anomaly_list(f, 'telegraph_noise', results['telegraph_noise'])
            if 'hot_pixels' in results:
                self._save_anomaly_list(f, 'hot_pixels', results.get('hot_pixels', []))
            
            # For multi-exposure results
            if 'exposures' in results:
                exp_grp = f.create_group('exposures')
                for i, exp_result in enumerate(results['exposures']):
                    exp_subgrp = exp_grp.create_group(f'exposure_{i}')
                    if 'summary' in exp_result:
                        for key, value in exp_result['summary'].items():
                            exp_subgrp.attrs[key] = value
        
        self.logger.info(f"Results saved to {results_path}")
    
    def _save_anomaly_list(self, h5file, name: str, anomaly_list: List[Dict]):
        """Save a list of anomalies to HDF5 group."""
        if not anomaly_list:
            return
        
        grp = h5file.create_group(name)
        grp.attrs['count'] = len(anomaly_list)
        
        for i, anomaly in enumerate(anomaly_list):
            anomaly_grp = grp.create_group(f'anomaly_{i}')
            
            for key, value in anomaly.items():
                if key == 'mask':
                    # Skip large mask arrays
                    continue
                elif key in ['position', 'centroid']:
                    anomaly_grp.attrs[key] = value
                elif key == 'pixel_coords':
                    if value:
                        anomaly_grp.create_dataset('pixel_coords', 
                                                 data=np.array(value))
                elif key == 'time_series':
                    if isinstance(value, np.ndarray):
                        anomaly_grp.create_dataset('time_series', data=value)
                elif isinstance(value, (int, float, str)):
                    anomaly_grp.attrs[key] = value


def main():
    """Command-line interface for H2RG anomaly detection."""
    parser = argparse.ArgumentParser(
        description='H2RG Infrared Detector Anomaly Detection'
    )
    
    parser.add_argument(
        'files', nargs='+', 
        help='FITS files to process (1 or more exposures)'
    )
    
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output directory for results and plots'
    )
    
    parser.add_argument(
        '--sigma', type=float, default=5.0,
        help='Detection threshold in sigma (default: 5.0)'
    )
    
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching of preprocessed data'
    )
    
    parser.add_argument(
        '--test-mode', action='store_true',
        help='Run in test mode with limited frames'
    )
    
    parser.add_argument(
        '--test-frames', type=int, default=50,
        help='Number of frames to process in test mode (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Configure detector
    detector_config = DetectorConfig(sigma_threshold=args.sigma)
    processing_config = ProcessingConfig(
        cache_enabled=not args.no_cache,
        test_mode=args.test_mode,
        test_frames=args.test_frames
    )
    
    # Initialize detector
    detector = H2RGAnomalyDetector(detector_config, processing_config)
    
    # Process files
    results = detector.process_exposures(args.files, args.output)
    
    # Print summary
    if 'summary' in results:
        print("\nDetection Summary:")
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())