import numpy as np
from numba import jit, prange
import logging
from typing import Dict, Union
from astropy.io import fits
import h5py
from tqdm import tqdm

class FrameDifferencer:
    """Computes frame differences for anomaly detection."""
    
    def __init__(self, reference_corrector=None):
        self.reference_corrector = reference_corrector
        self.logger = logging.getLogger(__name__)
    
    def compute_differences(self, frames: Union[np.ndarray, str], 
                          apply_correction: bool = True) -> Dict:
        """Compute frame differences from array or FITS file."""
        if isinstance(frames, str):
            return self._compute_fits_differences(frames, apply_correction)
        else:
            return self._compute_array_differences(frames, apply_correction)
    
    def _compute_array_differences(self, frames: np.ndarray, 
                                 apply_correction: bool) -> Dict:
        """Compute differences from numpy array."""
        if apply_correction and self.reference_corrector:
            self.logger.info("Applying reference pixel correction...")
            frames = self.reference_corrector.correct_batch(frames)
        
        # Use first frame as reference
        reference_frame = frames[0].astype(np.float32)
        
        # Compute differences using Numba
        diff_stack = self._numba_frame_differences(
            frames[1:].astype(np.float32), reference_frame
        )
        
        return {
            'differences': diff_stack,
            'frame_times': np.arange(1, len(diff_stack) + 1),
            'reference_frame': reference_frame,
            'total_frames': len(diff_stack)
        }
    
    def _compute_fits_differences(self, file_path: str, 
                                apply_correction: bool) -> Dict:
        """Compute differences from FITS file."""
        self.logger.info(f"Processing FITS file: {file_path}")
        
        with fits.open(file_path) as hdul:
            # Load reference frame
            frame_0 = hdul[1].data.astype(np.float32)
            
            if apply_correction and self.reference_corrector:
                frame_0 = self.reference_corrector.correct_frame(frame_0)
            
            # Process remaining frames
            n_frames = len(hdul) - 1
            diff_stack = []
            
            for i in tqdm(range(2, n_frames + 1), desc='Computing differences'):
                frame = hdul[i].data.astype(np.float32)
                
                if apply_correction and self.reference_corrector:
                    frame = self.reference_corrector.correct_frame(frame)
                
                diff = frame - frame_0
                diff_stack.append(diff)
            
            return {
                'differences': np.array(diff_stack),
                'frame_times': np.arange(1, len(diff_stack) + 1),
                'reference_frame': frame_0,
                'total_frames': len(diff_stack)
            }
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_frame_differences(frames, reference_frame):
        """Numba-optimized parallel difference computation."""
        n_frames, height, width = frames.shape
        diff_stack = np.empty_like(frames)
        
        for i in prange(n_frames):
            diff_stack[i] = frames[i] - reference_frame
        
        return diff_stack