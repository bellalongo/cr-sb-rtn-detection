import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from ..detectors import (
    CosmicRayDetector, SnowballDetector, 
    TelegraphNoiseDetector, HotPixelDetector
)

class AnomalyClassifier:
    """Orchestrates anomaly detection and classification."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.cosmic_ray_detector = CosmicRayDetector(config)
        self.snowball_detector = SnowballDetector(config)
        self.rtn_detector = TelegraphNoiseDetector(config)
        self.hot_pixel_detector = HotPixelDetector(config)
    
    def classify_single_exposure(self, temporal_data: Dict, 
                               diff_stack: np.ndarray) -> Dict:
        """Classify anomalies in a single exposure."""
        self.logger.info("Classifying anomalies in single exposure")
        
        results = {
            'cosmic_rays': [],
            'snowballs': [],
            'telegraph_noise': [],
            'unknown': []
        }
        
        # Detect cosmic rays
        cr_detection = self.cosmic_ray_detector.detect(temporal_data, diff_stack)
        cr_classified = self.cosmic_ray_detector.classify(cr_detection)
        results['cosmic_rays'] = cr_classified

        # # Detect telegraph noise
        # rtn_detection = self.rtn_detector.detect(temporal_data, diff_stack)
        # rtn_classified = self.rtn_detector.classify(rtn_detection)
        # results['telegraph_noise'] = rtn_classified

        return results
        
        # # Detect snowballs
        # sb_detection = self.snowball_detector.detect(temporal_data, diff_stack)
        # sb_classified = self.snowball_detector.classify(sb_detection)
        # results['snowballs'] = sb_classified
        
        # # Detect telegraph noise
        # rtn_detection = self.rtn_detector.detect(temporal_data, diff_stack)
        # rtn_classified = self.rtn_detector.classify(rtn_detection)
        # results['telegraph_noise'] = rtn_classified
        
        # # Create combined mask to find unclassified anomalies
        # classified_mask = np.zeros_like(temporal_data['first_appearance'], dtype=bool)
        
        # for event in cr_classified:
        #     if 'mask' in event:
        #         classified_mask |= event['mask']
        
        # for event in sb_classified:
        #     if 'mask' in event:
        #         classified_mask |= event['mask']
        
        # for event in rtn_classified:
        #     y, x = event['position']
        #     classified_mask[y, x] = True
        
        # # Find unclassified anomalies
        # anomaly_mask = temporal_data['first_appearance'] >= 0
        # unclassified_mask = anomaly_mask & ~classified_mask
        
        # # Create unknown events
        # y_coords, x_coords = np.where(unclassified_mask)
        # for y, x in zip(y_coords, x_coords):
        #     unknown_event = {
        #         'type': 'unknown',
        #         'position': (y, x),
        #         'first_frame': int(temporal_data['first_appearance'][y, x]),
        #         'persistence': int(temporal_data['persistence_count'][y, x]),
        #         'max_intensity': float(temporal_data['max_intensity'][y, x]),
        #         'confidence': 0.5
        #     }
        #     results['unknown'].append(unknown_event)
        
        # # Summary statistics
        # results['summary'] = {
        #     'total_anomalies': int(np.sum(anomaly_mask)),
        #     'cosmic_rays': len(results['cosmic_rays']),
        #     'snowballs': len(results['snowballs']),
        #     'telegraph_noise': len(results['telegraph_noise']),
        #     'unknown': len(results['unknown']),
        #     'classification_rate': 1 - (len(results['unknown']) / max(1, np.sum(anomaly_mask)))
        # }
        
        # self.logger.info(f"Classification summary: {results['summary']}")
        
        # return results
    
    def classify_multiple_exposures(self, temporal_data_list: List[Dict],
                                  diff_stack_list: List[np.ndarray]) -> Dict:
        """Classify anomalies across multiple exposures."""
        self.logger.info(f"Classifying anomalies across {len(temporal_data_list)} exposures")
        
        # First classify each exposure individually
        individual_results = []
        for i, (temporal_data, diff_stack) in enumerate(zip(temporal_data_list, diff_stack_list)):
            self.logger.info(f"Processing exposure {i+1}/{len(temporal_data_list)}")
            exposure_results = self.classify_single_exposure(temporal_data, diff_stack)
            exposure_results['exposure_id'] = i
            individual_results.append(exposure_results)
        
        # Detect hot pixels across exposures
        hot_pixel_detection = self.hot_pixel_detector.detect(
            temporal_data_list, diff_stack_list
        )
        hot_pixels_classified = self.hot_pixel_detector.classify(hot_pixel_detection)
        
        # Refine classifications based on cross-exposure analysis
        refined_results = self._refine_classifications(
            individual_results, hot_pixels_classified
        )
        
        # Combine results
        combined_results = {
            'exposures': refined_results,
            'hot_pixels': hot_pixels_classified,
            'summary': self._calculate_summary(refined_results, hot_pixels_classified)
        }
        
        return combined_results
    
    def _refine_classifications(self, individual_results: List[Dict],
                              hot_pixels: List[Dict]) -> List[Dict]:
        """Refine classifications based on cross-exposure information."""
        # Create hot pixel position set for quick lookup
        hot_pixel_positions = {(hp['position'][0], hp['position'][1]) 
                             for hp in hot_pixels}
        
        refined_results = []
        
        for exposure_result in individual_results:
            refined = {
                'exposure_id': exposure_result['exposure_id'],
                'cosmic_rays': [],
                'snowballs': [],
                'telegraph_noise': [],
                'hot_pixels': [],
                'unknown': []
            }
            
            # Check cosmic rays - should not appear in other exposures
            for cr in exposure_result['cosmic_rays']:
                if 'centroid' in cr:
                    y, x = int(cr['centroid'][0]), int(cr['centroid'][1])
                    if (y, x) not in hot_pixel_positions:
                        refined['cosmic_rays'].append(cr)
                    else:
                        # Reclassify as hot pixel
                        cr['type'] = 'hot_pixel'
                        cr['reclassified_from'] = 'cosmic_ray'
                        refined['hot_pixels'].append(cr)
            
            # Keep snowballs as is
            refined['snowballs'] = exposure_result['snowballs']
            
            # Check telegraph noise
            for rtn in exposure_result['telegraph_noise']:
                y, x = rtn['position']
                if (y, x) not in hot_pixel_positions:
                    refined['telegraph_noise'].append(rtn)
                else:
                    # Could be hot pixel with telegraph behavior
                    rtn['type'] = 'hot_pixel_with_rtn'
                    refined['hot_pixels'].append(rtn)
            
            # Check unknowns
            for unk in exposure_result['unknown']:
                y, x = unk['position']
                if (y, x) in hot_pixel_positions:
                    unk['type'] = 'hot_pixel'
                    unk['reclassified_from'] = 'unknown'
                    refined['hot_pixels'].append(unk)
                else:
                    refined['unknown'].append(unk)
                    
            total_anomalies = (len(refined['cosmic_rays']) +
                               len(refined['snowballs']) +
                               len(refined['telegraph_noise']) +
                               len(refined['hot_pixels']) +
                               len(refined['unknown']))

            refined['summary'] = {
                'cosmic_rays': len(refined['cosmic_rays']),
                'snowballs': len(refined['snowballs']),
                'telegraph_noise': len(refined['telegraph_noise']),
                'hot_pixels': len(refined['hot_pixels']),
                'unknown': len(refined['unknown']),
                'total_anomalies': total_anomalies,
                'classification_rate': 1 - (len(refined['unknown']) / max(1, total_anomalies))
            }
            
            refined_results.append(refined)
        
        return refined_results
    
    def _calculate_summary(self, refined_results: List[Dict],
                         hot_pixels: List[Dict]) -> Dict:
        """Calculate overall summary statistics."""
        total_cosmic_rays = sum(r['summary']['cosmic_rays'] for r in refined_results)
        total_snowballs = sum(r['summary']['snowballs'] for r in refined_results)
        total_rtn = sum(r['summary']['telegraph_noise'] for r in refined_results)
        total_unknown = sum(r['summary']['unknown'] for r in refined_results)
        
        return {
            'total_exposures': len(refined_results),
            'total_cosmic_rays': total_cosmic_rays,
            'total_snowballs': total_snowballs,
            'total_telegraph_noise': total_rtn,
            'total_hot_pixels': len(hot_pixels),
            'total_unknown': total_unknown,
            'avg_cosmic_rays_per_exposure': total_cosmic_rays / len(refined_results),
            'avg_snowballs_per_exposure': total_snowballs / len(refined_results),
            'avg_rtn_per_exposure': total_rtn / len(refined_results)
        }
