# import numpy as np
# from typing import Dict, List, Tuple
# from scipy import ndimage
# from skimage.measure import regionprops
# from .base_detector import BaseDetector

# class SnowballDetector(BaseDetector):
#     """Detects snowball events in H2RG data."""
    
#     def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
#         """Detect snowball candidates."""
#         first_appearance = temporal_data['first_appearance']
#         persistence = temporal_data['persistence_count']
#         max_intensity = temporal_data['max_intensity']
        
#         # Snowballs appear suddenly with moderate to high intensity
#         sudden_appearance = first_appearance > 0
#         intensity_range = (max_intensity > self.config.snowball_min_intensity) & \
#                          (max_intensity < self.config.snowball_max_intensity)
        
#         # Initial mask
#         snowball_mask = sudden_appearance & intensity_range
        
#         # Find connected components
#         labeled, num_features = ndimage.label(snowball_mask)
        
#         candidates = []
#         for i in range(1, num_features + 1):
#             component_mask = labeled == i
#             if np.sum(component_mask) < self.config.min_anomaly_pixels:
#                 continue
            
#             # Check circularity using regionprops
#             props = regionprops(component_mask.astype(int))[0]
            
#             # Calculate circularity
#             area = props.area
#             perimeter = props.perimeter
#             circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
#             # Check size constraints
#             equivalent_diameter = props.equivalent_diameter
#             if not (self.config.snowball_min_radius * 2 <= equivalent_diameter <= 
#                    self.config.snowball_max_radius * 2):
#                 continue
            
#             # Check circularity threshold
#             if circularity < self.config.snowball_circularity_threshold:
#                 continue
            
#             # Analyze temporal expansion
#             expansion_rate = self._analyze_expansion(
#                 component_mask, first_appearance, diff_stack
#             )
            
#             candidate = {
#                 'type': 'snowball_candidate',
#                 'centroid': props.centroid,
#                 'first_frame': int(np.min(first_appearance[component_mask])),
#                 'mean_intensity': np.mean(max_intensity[component_mask]),
#                 'max_intensity': np.max(max_intensity[component_mask]),
#                 'circularity': circularity,
#                 'equivalent_diameter': equivalent_diameter,
#                 'expansion_rate': expansion_rate,
#                 'area': area,
#                 'bbox': props.bbox,
#                 'mask': component_mask
#             }
            
#             candidates.append(candidate)
        
#         self.logger.info(f"Found {len(candidates)} snowball candidates")
        
#         return {
#             'candidates': candidates,
#             'mask': snowball_mask,
#             'num_candidates': len(candidates)
#         }
    
#     def _analyze_expansion(self, mask: np.ndarray, first_appearance: np.ndarray,
#                           diff_stack: np.ndarray) -> float:
#         """Analyze radial expansion of snowball over time."""
#         y_coords, x_coords = np.where(mask)
#         centroid_y = np.mean(y_coords)
#         centroid_x = np.mean(x_coords)
        
#         first_frame = int(np.min(first_appearance[mask]))
        
#         # Track radius over time
#         radii = []
#         for frame_offset in range(0, min(10, diff_stack.shape[0] - first_frame)):
#             frame_idx = first_frame + frame_offset
#             frame_data = diff_stack[frame_idx]
            
#             # Get region around centroid
#             region = self._get_region_around_pixel(
#                 int(centroid_y), int(centroid_x), frame_data, 
#                 radius=self.config.snowball_max_radius * 2
#             )
            
#             # Find pixels above threshold
#             threshold = np.percentile(frame_data, 95)
#             above_threshold = region > threshold
            
#             if np.any(above_threshold):
#                 # Calculate mean radius
#                 y_above, x_above = np.where(above_threshold)
#                 distances = np.sqrt((y_above - region.shape[0]//2)**2 + 
#                                   (x_above - region.shape[1]//2)**2)
#                 mean_radius = np.mean(distances)
#                 radii.append(mean_radius)
        
#         # Calculate expansion rate (pixels per frame)
#         if len(radii) > 1:
#             expansion_rate = np.polyfit(range(len(radii)), radii, 1)[0]
#         else:
#             expansion_rate = 0
        
#         return expansion_rate
    
#     def classify(self, candidates: Dict) -> List[Dict]:
#         """Classify snowball candidates."""
#         classified = []
        
#         for candidate in candidates['candidates']:
#             # Check all snowball criteria
#             if (candidate['circularity'] >= self.config.snowball_circularity_threshold and
#                 self.config.snowball_min_radius * 2 <= candidate['equivalent_diameter'] <= 
#                 self.config.snowball_max_radius * 2 and
#                 candidate['expansion_rate'] >= 0):  # Should expand or stay same size
                
#                 confidence = min(candidate['circularity'], 0.95)
                
#                 classified_event = {
#                     'type': 'snowball',
#                     'confidence': confidence,
#                     **candidate
#                 }
                
#                 classified.append(classified_event)
        
#         return classified

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, find_contours
from skimage.draw import ellipse
# import cv2
import logging

class SnowballDetector:
    """
    Detects snowball events in H2RG near-infrared data following JWST methodology.
    
    Based on JWST-STScI-008545: Detection and Flagging of Showers and Snowballs in JWST
    by Michael Regan, 2024.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
        """
        Detect snowball candidates using JWST methodology with working fallback.
        """
        self.logger.info("Starting snowball detection...")
        
        # Extract temporal data
        first_appearance = temporal_data['first_appearance']
        max_intensity = temporal_data['max_intensity']
        saturation_map = temporal_data.get('saturation_map', np.zeros_like(first_appearance))
        jump_map = temporal_data.get('jump_map', np.zeros_like(first_appearance))
        
        # DEBUG: Print what we have
        print(f"DEBUG: first_appearance range: {np.min(first_appearance)} to {np.max(first_appearance)}")
        print(f"DEBUG: max_intensity range: {np.min(max_intensity)} to {np.max(max_intensity)}")
        print(f"DEBUG: saturation_map sum: {np.sum(saturation_map)}")
        print(f"DEBUG: jump_map sum: {np.sum(jump_map)}")
        print(f"DEBUG: High intensity pixels (>1000): {np.sum(max_intensity > 1000)}")
        print(f"DEBUG: Sudden appearance pixels: {np.sum(first_appearance > 0)}")
        
        candidates = []
        newly_saturated = {}
        newly_jumped = {}
        
        # Try JWST methodology only if we have the required data
        if np.sum(saturation_map) > 0 or np.sum(jump_map) > 0:
            print("DEBUG: Trying JWST method...")
            
            # Step 1: Find newly saturated pixels 
            newly_saturated = self._find_newly_saturated_pixels(
                saturation_map, first_appearance, diff_stack
            )
            print(f"DEBUG: Found {len(newly_saturated)} saturated regions")
            
            # Step 2: Find newly flagged jump pixels
            newly_jumped = self._find_newly_jumped_pixels(
                jump_map, first_appearance, diff_stack
            )
            print(f"DEBUG: Found {len(newly_jumped)} jump regions")
            
            # Step 3: Associate jump regions with saturated cores
            if newly_saturated or newly_jumped:
                snowball_candidates = self._associate_jumps_with_saturation(
                    newly_saturated, newly_jumped, diff_stack
                )
                
                # Step 4: Analyze each candidate
                for candidate in snowball_candidates:
                    if self._validate_snowball_characteristics(candidate, diff_stack):
                        candidates.append(candidate)
        
        # ALWAYS try simple detection as well (not just fallback)
        print("DEBUG: Trying simple detection...")
        simple_candidates = self._simple_snowball_detection(temporal_data, diff_stack)
        print(f"DEBUG: Simple method found {len(simple_candidates)} candidates")
        
        # Add simple candidates to the list
        candidates.extend(simple_candidates)
        
        self.logger.info(f"Found {len(candidates)} total snowball candidates")
        
        return {
            'candidates': candidates,
            'num_candidates': len(candidates),
            'newly_saturated': newly_saturated,
            'newly_jumped': newly_jumped
        }

    def _simple_snowball_detection(self, temporal_data: Dict, diff_stack: np.ndarray) -> List[Dict]:
        """Simple snowball detection with proper format for plotting."""
        from scipy import ndimage
        from skimage.measure import regionprops
        
        first_appearance = temporal_data['first_appearance']
        max_intensity = temporal_data['max_intensity']
        
        print("DEBUG: Starting simple snowball detection...")
        
        # Find high-intensity regions that appear suddenly
        sudden_appearance = first_appearance > 0
        high_intensity = max_intensity > 1000
        
        candidate_mask = sudden_appearance & high_intensity
        print(f"DEBUG: Simple detection mask has {np.sum(candidate_mask)} pixels")
        
        if not np.any(candidate_mask):
            return []
        
        # Find connected components
        labeled, num_features = ndimage.label(candidate_mask)
        print(f"DEBUG: Found {num_features} connected components")
        
        candidates = []
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            area = np.sum(component_mask)
            
            # Size filter
            if not (5 <= area <= 5000):
                continue
            
            try:
                props = regionprops(component_mask.astype(int))[0]
            except IndexError:
                continue
            
            # Calculate circularity
            if props.perimeter > 0:
                circularity = 4 * np.pi * props.area / (props.perimeter ** 2)
            else:
                circularity = 0
            
            # Lenient filters
            if circularity < 0.1 or props.eccentricity > 0.95:
                continue
            
            # Get first frame for this component
            first_frame = int(np.min(first_appearance[component_mask]))
            
            # Create candidate with ALL the fields the plotting code expects
            candidate = {
                'type': 'snowball_candidate',
                'detection_method': 'simple',
                
                # Position data (multiple formats for compatibility)
                'centroid': props.centroid,  # Main position for plotting
                'position': props.centroid,  # Alternative position format
                
                # Timing data
                'frame': first_frame,
                'first_frame': first_frame,  # Expected by plotting code
                
                # Intensity data
                'max_intensity': float(np.max(max_intensity[component_mask])),
                'mean_intensity': float(np.mean(max_intensity[component_mask])),
                
                # Shape data
                'area': area,
                'circularity': circularity,
                'eccentricity': props.eccentricity,
                'equivalent_diameter': props.equivalent_diameter,
                'bbox': props.bbox,
                'mask': component_mask,
                
                # Time series data (extract from diff_stack)
                'time_series': diff_stack[:, int(props.centroid[0]), int(props.centroid[1])],
                
                # Jump data (for compatibility with JWST method)
                'jump_data': {
                    'mask': component_mask,
                    'area': area,
                    'centroid': props.centroid,
                    'ellipse': {
                        'center': props.centroid,
                        'major_axis_length': props.major_axis_length,
                        'minor_axis_length': props.minor_axis_length,
                        'orientation': props.orientation,
                        'eccentricity': props.eccentricity,
                        'area': area
                    }
                }
            }
            
            candidates.append(candidate)
            print(f"DEBUG: ✅ FOUND SNOWBALL: area={area}, circularity={circularity:.3f}, "
                f"intensity={candidate['max_intensity']:.1f}, "
                f"centroid=({props.centroid[0]:.1f}, {props.centroid[1]:.1f})")
        
        return candidates
    
    # def detect(self, temporal_data: Dict, diff_stack: np.ndarray) -> Dict:
    #     """
    #     Detect snowball candidates using JWST methodology.
        
    #     Key characteristics from paper:
    #     1. Snowballs always have a saturated core
    #     2. They show exponential decay with radius
    #     3. They create charge spilling rings
    #     4. They have diffuse halos extending beyond jump detection
    #     """
    #     self.logger.info("Starting snowball detection...")

    #     # Extract temporal data
    #     first_appearance = temporal_data['first_appearance']
    #     max_intensity = temporal_data['max_intensity']
    #     saturation_map = temporal_data.get('saturation_map', np.zeros_like(first_appearance))
    #     jump_map = temporal_data.get('jump_map', np.zeros_like(first_appearance))
        
    #     # DEBUG: Print what we have
    #     print(f"DEBUG: first_appearance range: {np.min(first_appearance)} to {np.max(first_appearance)}")
    #     print(f"DEBUG: max_intensity range: {np.min(max_intensity)} to {np.max(max_intensity)}")
    #     print(f"DEBUG: saturation_map sum: {np.sum(saturation_map)}")
    #     print(f"DEBUG: jump_map sum: {np.sum(jump_map)}")
    #     print(f"DEBUG: High intensity pixels (>1000): {np.sum(max_intensity > 1000)}")
    #     print(f"DEBUG: Sudden appearance pixels: {np.sum(first_appearance > 0)}")
        
    #     # Step 1: Find newly saturated pixels (core requirement)
    #     newly_saturated = self._find_newly_saturated_pixels(
    #         saturation_map, first_appearance, diff_stack
    #     )
    #     print(f"DEBUG: Found {len(newly_saturated)} saturated regions")
        
    #     # Step 2: Find newly flagged jump pixels around saturated cores
    #     newly_jumped = self._find_newly_jumped_pixels(
    #         jump_map, first_appearance, diff_stack
    #     )
    #     print(f"DEBUG: Found {len(newly_jumped)} jump regions")
        
    #     # # Extract temporal data
    #     # first_appearance = temporal_data['first_appearance']
    #     # max_intensity = temporal_data['max_intensity']
    #     # saturation_map = temporal_data.get('saturation_map', np.zeros_like(first_appearance))
    #     # jump_map = temporal_data.get('jump_map', np.zeros_like(first_appearance))
        
    #     # # Step 1: Find newly saturated pixels (core requirement)
    #     # # Snowballs must have saturated cores that appear suddenly
    #     # newly_saturated = self._find_newly_saturated_pixels(
    #     #     saturation_map, first_appearance, diff_stack
    #     # )
        
    #     # # Step 2: Find newly flagged jump pixels around saturated cores
    #     # newly_jumped = self._find_newly_jumped_pixels(
    #     #     jump_map, first_appearance, diff_stack
    #     # )
        
    #     # Step 3: Associate jump regions with saturated cores
    #     snowball_candidates = self._associate_jumps_with_saturation(
    #         newly_saturated, newly_jumped, diff_stack
    #     )
        
    #     # Step 4: Analyze each candidate for snowball characteristics
    #     validated_candidates = []
    #     for candidate in snowball_candidates:
    #         if self._validate_snowball_characteristics(candidate, diff_stack):
    #             validated_candidates.append(candidate)
        
    #     self.logger.info(f"Found {len(validated_candidates)} snowball candidates")
        
    #     return {
    #         'candidates': validated_candidates,
    #         'num_candidates': len(validated_candidates),
    #         'newly_saturated': newly_saturated,
    #         'newly_jumped': newly_jumped
    #     }
    
    def _find_newly_saturated_pixels(self, saturation_map: np.ndarray, 
                                   first_appearance: np.ndarray,
                                   diff_stack: np.ndarray) -> Dict:
        """Find pixels that become saturated during the observation."""
        newly_saturated_regions = {}
        
        # Group saturated pixels by when they first appear
        for frame_idx in range(1, diff_stack.shape[0]):  # Skip first frame
            # Find pixels that become saturated in this frame
            saturated_this_frame = (first_appearance == frame_idx) & (saturation_map > 0)
            
            if not np.any(saturated_this_frame):
                continue
            
            # Find connected components of saturated pixels
            labeled_sat, num_sat = ndimage.label(saturated_this_frame)
            
            for region_id in range(1, num_sat + 1):
                region_mask = labeled_sat == region_id
                area = np.sum(region_mask)
                
                # Apply minimum area threshold from paper
                if area >= self.config.snowball_min_sat_area:
                    # Calculate enclosing ellipse
                    ellipse_params = self._fit_minimum_enclosing_ellipse(region_mask)
                    
                    if ellipse_params is not None:
                        newly_saturated_regions[f"sat_{frame_idx}_{region_id}"] = {
                            'frame': frame_idx,
                            'mask': region_mask,
                            'area': area,
                            'ellipse': ellipse_params,
                            'centroid': ellipse_params['center']
                        }
        
        return newly_saturated_regions
    
    def _find_newly_jumped_pixels(self, jump_map: np.ndarray,
                                first_appearance: np.ndarray,
                                diff_stack: np.ndarray) -> Dict:
        """Find pixels flagged as jumps (cosmic ray hits) during observation."""
        newly_jumped_regions = {}
        
        for frame_idx in range(1, diff_stack.shape[0]):
            # Find pixels that show jumps in this frame
            jumped_this_frame = (first_appearance == frame_idx) & (jump_map > 0)
            
            if not np.any(jumped_this_frame):
                continue
            
            # Find connected components of jump pixels
            labeled_jump, num_jump = ndimage.label(jumped_this_frame)
            
            for region_id in range(1, num_jump + 1):
                region_mask = labeled_jump == region_id
                area = np.sum(region_mask)
                
                # Apply minimum area threshold
                if area >= self.config.snowball_min_jump_area:
                    ellipse_params = self._fit_minimum_enclosing_ellipse(region_mask)
                    
                    if ellipse_params is not None:
                        newly_jumped_regions[f"jump_{frame_idx}_{region_id}"] = {
                            'frame': frame_idx,
                            'mask': region_mask,
                            'area': area,
                            'ellipse': ellipse_params,
                            'centroid': ellipse_params['center']
                        }
        
        return newly_jumped_regions
    
    def _associate_jumps_with_saturation(self, newly_saturated: Dict, 
                                       newly_jumped: Dict,
                                       diff_stack: np.ndarray) -> List[Dict]:
        """
        Associate jump ellipses with saturated cores to identify snowballs.
        From paper: "For each jump ellipse that has a newly saturated pixel 
        at the center of the ellipse, add the jump ellipse parameters to the list of snowballs."
        """
        candidates = []
        
        for jump_key, jump_data in newly_jumped.items():
            jump_frame = jump_data['frame']
            jump_center = jump_data['centroid']
            
            # Look for saturated pixels at the center of this jump ellipse
            # Check same frame and nearby frames
            associated_saturation = None
            min_distance = float('inf')
            
            for sat_key, sat_data in newly_saturated.items():
                sat_frame = sat_data['frame']
                sat_center = sat_data['centroid']
                
                # Check if frames are close (snowball can saturate slightly after jump)
                frame_diff = abs(sat_frame - jump_frame)
                if frame_diff <= 2:  # Allow small frame difference
                    
                    # Check if saturated core is near center of jump ellipse
                    distance = np.sqrt((jump_center[0] - sat_center[0])**2 + 
                                     (jump_center[1] - sat_center[1])**2)
                    
                    # Must be within the jump ellipse or very close
                    if distance < min_distance and distance <= self.config.snowball_max_sat_jump_distance:
                        min_distance = distance
                        associated_saturation = sat_data
            
            # Special case: Near detector edges, don't require saturated core
            is_near_edge = self._is_near_detector_edge(jump_center, diff_stack.shape[1:])
            
            if associated_saturation is not None or is_near_edge:
                candidate = {
                    'type': 'snowball_candidate',
                    'frame': jump_frame,
                    'jump_data': jump_data,
                    'saturation_data': associated_saturation,
                    'is_edge_case': is_near_edge,
                    'center_distance': min_distance if associated_saturation else None
                }
                candidates.append(candidate)
        
        return candidates
    
    def _validate_snowball_characteristics(self, candidate: Dict, 
                                         diff_stack: np.ndarray) -> bool:
        """
        Validate that candidate exhibits snowball characteristics from the paper:
        1. Exponential decay with radius
        2. Reasonable size (5-1000+ pixels)
        3. Circular to moderately elliptical shape
        4. Intensity in expected range
        """
        jump_data = candidate['jump_data']
        saturation_data = candidate.get('saturation_data')
        frame_idx = candidate['frame']
        
        # Check size constraints
        total_area = jump_data['area']
        if saturation_data:
            total_area += saturation_data['area']
        
        if not (self.config.snowball_min_total_area <= total_area <= self.config.snowball_max_total_area):
            return False
        
        # Check ellipticity (snowballs are mostly circular)
        ellipse = jump_data['ellipse']
        eccentricity = ellipse['eccentricity']
        if eccentricity > self.config.snowball_max_eccentricity:
            return False
        
        # Analyze radial profile for exponential decay
        if frame_idx < diff_stack.shape[0]:
            has_exponential_profile = self._check_exponential_decay(
                candidate, diff_stack[frame_idx]
            )
            if not has_exponential_profile:
                return False
        
        # Check for charge spilling characteristics in subsequent frames
        if frame_idx + 1 < diff_stack.shape[0]:
            has_charge_spilling = self._check_charge_spilling(
                candidate, diff_stack[frame_idx + 1:frame_idx + 3]
            )
            # Charge spilling is expected but not required for validation
        
        return True
    
    def _check_exponential_decay(self, candidate: Dict, diff_image: np.ndarray) -> bool:
        """
        Check if the intensity profile shows exponential decay from center.
        From paper: "snowball shows an exponential decay in the accumulated charge from the center"
        """
        center = candidate['jump_data']['centroid']
        ellipse = candidate['jump_data']['ellipse']
        
        # Create radial bins around the center
        max_radius = max(ellipse['major_axis_length'], ellipse['minor_axis_length']) / 2
        radii = np.linspace(1, min(max_radius * 2, 50), 20)
        
        radial_intensities = []
        for radius in radii:
            # Create circular mask at this radius
            y, x = np.ogrid[:diff_image.shape[0], :diff_image.shape[1]]
            mask = ((y - center[0])**2 + (x - center[1])**2 <= radius**2) & \
                   ((y - center[0])**2 + (x - center[1])**2 > (radius-2)**2)
            
            if np.any(mask):
                intensity = np.mean(diff_image[mask])
                radial_intensities.append(intensity)
            else:
                radial_intensities.append(0)
        
        # Check for monotonic decrease (simplified exponential check)
        if len(radial_intensities) >= 3:
            # At least the first few points should be decreasing
            decreasing_count = 0
            for i in range(1, min(5, len(radial_intensities))):
                if radial_intensities[i] <= radial_intensities[i-1]:
                    decreasing_count += 1
            
            return decreasing_count >= 2  # Allow some noise
        
        return True  # Insufficient data, assume valid
    
    def _check_charge_spilling(self, candidate: Dict, subsequent_frames: np.ndarray) -> bool:
        """
        Check for charge spilling ring in subsequent frames.
        From paper: "This ring extends into the third frame and shows a positive signal 
        for the rest of the integration. This signal is due to charge migration from the saturated core."
        """
        if subsequent_frames.shape[0] == 0:
            return False
        
        saturation_data = candidate.get('saturation_data')
        if not saturation_data:
            return False
        
        sat_center = saturation_data['centroid']
        sat_ellipse = saturation_data['ellipse']
        
        # Look for ring of positive signal around saturated core
        for frame_idx in range(subsequent_frames.shape[0]):
            frame = subsequent_frames[frame_idx]
            
            # Create ring mask around saturated core
            inner_radius = sat_ellipse['major_axis_length'] / 2
            outer_radius = inner_radius + self.config.snowball_charge_spill_width
            
            y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
            ring_mask = ((y - sat_center[0])**2 + (x - sat_center[1])**2 <= outer_radius**2) & \
                       ((y - sat_center[0])**2 + (x - sat_center[1])**2 > inner_radius**2)
            
            if np.any(ring_mask):
                ring_intensity = np.mean(frame[ring_mask])
                if ring_intensity > self.config.snowball_charge_spill_threshold:
                    return True
        
        return False
    
    def _fit_minimum_enclosing_ellipse(self, mask: np.ndarray) -> Optional[Dict]:
        """Fit minimum enclosing ellipse to a binary mask region."""
        try:
            # Get region properties
            props = regionprops(mask.astype(int))
            if not props:
                return None
            
            prop = props[0]
            
            # Use scikit-image regionprops ellipse parameters
            center = prop.centroid
            major_axis = prop.major_axis_length
            minor_axis = prop.minor_axis_length
            orientation = prop.orientation
            eccentricity = prop.eccentricity
            
            return {
                'center': center,
                'major_axis_length': major_axis,
                'minor_axis_length': minor_axis,
                'orientation': orientation,
                'eccentricity': eccentricity,
                'area': prop.area
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to fit ellipse: {e}")
            return None
    
    def _is_near_detector_edge(self, center: Tuple[float, float], 
                             image_shape: Tuple[int, int]) -> bool:
        """Check if point is near detector edge (within edge_size pixels)."""
        edge_size = getattr(self.config, 'snowball_edge_size', 25)
        
        y, x = center
        height, width = image_shape
        
        return (y < edge_size or y >= height - edge_size or 
                x < edge_size or x >= width - edge_size)
    
    def create_snowball_mask(self, candidates: List[Dict], 
                           image_shape: Tuple[int, int]) -> Dict:
        """
        Create expanded masks for snowball flagging following JWST methodology.
        
        From paper:
        - Expand saturated core by sat_expand pixels for charge spilling ring
        - Expand jump ellipse by expand_factor for diffuse halo
        - Limit expansions by max_extended_radius
        """
        expanded_saturation_mask = np.zeros(image_shape, dtype=bool)
        expanded_jump_mask = np.zeros(image_shape, dtype=bool)
        
        for candidate in candidates:
            # Expand saturated core if present and large enough
            saturation_data = candidate.get('saturation_data')
            if saturation_data:
                sat_ellipse = saturation_data['ellipse']
                minor_radius = sat_ellipse['minor_axis_length'] / 2
                
                if minor_radius >= self.config.snowball_min_sat_radius_extend:
                    # Expand saturation region
                    expanded_sat_mask = self._expand_ellipse_region(
                        sat_ellipse, image_shape, 
                        expansion_pixels=self.config.snowball_sat_expand
                    )
                    expanded_saturation_mask |= expanded_sat_mask
            
            # Expand jump ellipse for diffuse halo
            jump_data = candidate['jump_data']
            jump_ellipse = jump_data['ellipse']
            
            # Expand minor axis by factor, major axis by same number of pixels
            minor_expansion_pixels = (jump_ellipse['minor_axis_length'] / 2) * \
                                   (self.config.snowball_expand_factor - 1)
            
            expanded_jump_mask = self._expand_ellipse_region(
                jump_ellipse, image_shape,
                expansion_pixels=minor_expansion_pixels,
                limit_radius=self.config.snowball_max_extended_radius
            )
            expanded_jump_mask |= expanded_jump_mask
        
        return {
            'expanded_saturation': expanded_saturation_mask,
            'expanded_jump': expanded_jump_mask,
            'combined': expanded_saturation_mask | expanded_jump_mask
        }
    
    def _expand_ellipse_region(self, ellipse_params: Dict, image_shape: Tuple[int, int],
                             expansion_pixels: float, limit_radius: Optional[float] = None) -> np.ndarray:
        """Expand an ellipse region by specified number of pixels."""
        center = ellipse_params['center']
        major_axis = ellipse_params['major_axis_length'] / 2 + expansion_pixels
        minor_axis = ellipse_params['minor_axis_length'] / 2 + expansion_pixels
        orientation = ellipse_params['orientation']
        
        # Apply radius limit if specified
        if limit_radius:
            major_axis = min(major_axis, limit_radius)
            minor_axis = min(minor_axis, limit_radius)
        
        # Create expanded ellipse mask
        mask = np.zeros(image_shape, dtype=bool)
        
        try:
            # Use scikit-image ellipse drawing
            rr, cc = ellipse(center[0], center[1], minor_axis, major_axis, 
                           shape=image_shape, rotation=orientation)
            mask[rr, cc] = True
        except Exception as e:
            self.logger.debug(f"Failed to create expanded ellipse: {e}")
        
        return mask
    
    def classify(self, candidates: Dict) -> List[Dict]:
        """Classify snowball candidates with confidence scoring."""
        classified = []
        
        for candidate in candidates['candidates']:
            # Calculate confidence based on multiple factors
            confidence = self._calculate_snowball_confidence(candidate)

            print(candidate, confidence)
            
            if confidence > getattr(self.config, 'snowball_min_confidence', 0.2):
                classified_event = {
                    'type': 'snowball',
                    'confidence': confidence,
                    **candidate
                }
                classified.append(classified_event)
        
        return classified
    
    def _calculate_snowball_confidence(self, candidate: Dict) -> float:
        """Calculate confidence score for snowball detection."""
        confidence = 0.0
        
        # Has saturated core (40% of confidence)
        if candidate.get('saturation_data'):
            confidence += 0.4
        elif candidate.get('is_edge_case'):
            confidence += 0.2  # Reduced confidence for edge cases
        
        # Good ellipse fit (20% of confidence)
        jump_ellipse = candidate['jump_data']['ellipse']
        if jump_ellipse['eccentricity'] < 0.5:  # Relatively circular
            confidence += 0.2
        elif jump_ellipse['eccentricity'] < 0.7:
            confidence += 0.1
        
        # Appropriate size (20% of confidence)
        total_area = candidate['jump_data']['area']
        if candidate.get('saturation_data'):
            total_area += candidate['saturation_data']['area']
        
        if 10 <= total_area <= 100:  # Typical snowball size
            confidence += 0.2
        elif 5 <= total_area <= 1000:  # Extended range
            confidence += 0.1
        
        # Distance between saturation and jump centers (20% of confidence)
        if candidate.get('center_distance') is not None:
            if candidate['center_distance'] < 5:
                confidence += 0.2
            elif candidate['center_distance'] < 10:
                confidence += 0.1
        
        return min(0.95, confidence)