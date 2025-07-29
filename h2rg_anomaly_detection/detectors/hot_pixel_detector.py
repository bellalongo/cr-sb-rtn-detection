import numpy as np
from typing import Dict, List, Tuple
from .base_detector import BaseDetector

class HotPixelDetector(BaseDetector):
    """Detects hot pixels that persist across exposures."""
    
    def detect(self, temporal_data_list: List[Dict], 
              diff_stack_list: List[np.ndarray]) -> Dict:
        """Detect hot pixels across multiple exposures."""
        if len(temporal_data_list) < self.config.min_exposures_for_classification:
            self.logger.warning(f"Need at least {self.config.min_exposures_for_classification} exposures for hot pixel detection")
            return {'candidates': [], 'mask': None, 'num_candidates': 0}
        
        # Combine persistence information across exposures
        height, width = temporal_data_list[0]['first_appearance'].shape
        
        # Count how many exposures each pixel appears in
        exposure_count = np.zeros((height, width), dtype=int)
        total_persistence = np.zeros((height, width), dtype=float)
        intensity_values = []
        
        for i, temporal_data in enumerate(temporal_data_list):
            # Check if pixel appears from the beginning
            appears_early = temporal_data['first_appearance'] <= 5  # Within first 5 frames
            high_persistence = temporal_data['persistence_count'] > \
                             diff_stack_list[i].shape[0] * self.config.hot_pixel_persistence_threshold
            
            hot_in_exposure = appears_early & high_persistence
            exposure_count[hot_in_exposure] += 1
            
            # Track persistence fraction
            persistence_fraction = temporal_data['persistence_count'] / diff_stack_list[i].shape[0]
            total_persistence += persistence_fraction
            
            # Collect intensity values
            intensity_values.append(temporal_data['max_intensity'])
        
        # Average persistence across exposures
        avg_persistence = total_persistence / len(temporal_data_list)
        
        # Hot pixels appear in most exposures
        min_exposures = int(len(temporal_data_list) * self.config.hot_pixel_cross_exposure_threshold)
        hot_pixel_mask = (exposure_count >= min_exposures) & \
                        (avg_persistence >= self.config.hot_pixel_persistence_threshold)
        
        # Find individual hot pixels
        y_coords, x_coords = np.where(hot_pixel_mask)
        
        candidates = []
        for y, x in zip(y_coords, x_coords):
            # Calculate intensity statistics across exposures
            pixel_intensities = [intensity_values[i][y, x] for i in range(len(intensity_values))]
            mean_intensity = np.mean(pixel_intensities)
            std_intensity = np.std(pixel_intensities)
            cv_intensity = std_intensity / mean_intensity if mean_intensity > 0 else 0
            
            candidate = {
                'type': 'hot_pixel_candidate',
                'position': (y, x),
                'exposure_count': int(exposure_count[y, x]),
                'avg_persistence': avg_persistence[y, x],
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'cv_intensity': cv_intensity,
                'total_exposures': len(temporal_data_list)
            }
            
            candidates.append(candidate)
        
        self.logger.info(f"Found {len(candidates)} hot pixel candidates")
        
        return {
            'candidates': candidates,
            'mask': hot_pixel_mask,
            'num_candidates': len(candidates)
        }
    
    def classify(self, candidates: Dict) -> List[Dict]:
        """Classify hot pixel candidates."""
        classified = []
        
        for candidate in candidates['candidates']:
            # Check intensity variation
            if candidate['cv_intensity'] <= self.config.hot_pixel_intensity_variation:
                # Calculate confidence based on exposure count
                confidence = candidate['exposure_count'] / candidate['total_exposures']
                
                classified_event = {
                    'type': 'hot_pixel',
                    'confidence': confidence,
                    **candidate
                }
                
                classified.append(classified_event)
        
        return classified
