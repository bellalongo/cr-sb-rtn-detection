import numpy as np
from numba import jit, prange
import logging
from typing import Union

class ReferencePixelCorrector:
    """Performs reference pixel correction on H2RG detector frames."""
    
    def __init__(self, x_window: int = 64, y_window: int = 4):
        self.x_window = x_window
        self.y_window = y_window
        self.logger = logging.getLogger(__name__)
    
    def correct_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply reference pixel correction to a single frame."""
        return self._subtract_reference_pixels_numba(
            frame.astype(np.float32), self.x_window, self.y_window
        )
    
    def correct_batch(self, frames: np.ndarray) -> np.ndarray:
        """Apply reference pixel correction to multiple frames."""
        return batch_reference_correction(
            frames.astype(np.float32), self.x_window, self.y_window
        )
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _subtract_reference_pixels_numba(frame, x_opt=64, y_opt=4):
        """Numba-optimized reference pixel correction."""
        corrected_frame = frame.copy()
        
        # Extract reference pixels
        up_ref = frame[:4, :]
        down_ref = frame[2044:, :]
        left_ref = frame[:, :4]
        right_ref = frame[:, 2044:]
        
        # Process each channel
        for ch in range(32):
            corrected_frame = _perform_ud_correction_numba(
                corrected_frame, up_ref, down_ref, ch, x_opt
            )
        
        # Apply left/right correction
        corrected_frame = _perform_lr_correction_numba(
            corrected_frame, up_ref, down_ref, left_ref, right_ref, y_opt
        )
        
        return corrected_frame

@jit(nopython=True, cache=True)
def _perform_ud_correction_numba(corrected_frame, up_ref, down_ref, ch, x_opt):
    """Numba-optimized up/down correction."""
    if ch == 0:
        col_start, col_end = 4, 64
    elif ch == 31:
        col_start, col_end = ch * 64, 2044
    else:
        col_start, col_end = ch * 64, (ch + 1) * 64
    
    for col in range(col_start, min(col_end, 2044)):
        window_start = max(0, col - x_opt)
        window_end = min(2048, col + x_opt + 1)
        
        up_sum = 0.0
        down_sum = 0.0
        count = 0
        
        for i in range(4):
            for j in range(window_start, window_end):
                up_sum += up_ref[i, j]
                down_sum += down_ref[i, j]
                count += 1
        
        up_avg = up_sum / count
        down_avg = down_sum / count
        slope = (up_avg - down_avg) / 2044
        
        for row in range(4, 2044):
            ref_correction = down_avg + (row - 1.5) * slope
            corrected_frame[row, col] -= ref_correction
    
    return corrected_frame

@jit(nopython=True, cache=True)
def _perform_lr_correction_numba(corrected_frame, up_ref, down_ref, left_ref, right_ref, y_opt):
    """Numba-optimized left/right correction."""
    left_ref_corrected = left_ref.copy()
    right_ref_corrected = right_ref.copy()
    
    # Calculate full averages
    up_sum = np.sum(up_ref)
    down_sum = np.sum(down_ref)
    count = up_ref.size
    
    up_avg_full = up_sum / count
    down_avg_full = down_sum / count
    slope_full = (up_avg_full - down_avg_full) / 2044
    
    # Correct reference pixels
    for row in range(4, 2044):
        ref_correction = down_avg_full + (row - 1.5) * slope_full
        for col in range(4):
            left_ref_corrected[row, col] -= ref_correction
            right_ref_corrected[row, col] -= ref_correction
    
    # Apply sliding window correction
    for row in range(4, 2044):
        window_start = max(4, row - y_opt)
        window_end = min(2044, row + y_opt + 1)
        
        left_sum = 0.0
        right_sum = 0.0
        count = 0
        
        for i in range(window_start, window_end):
            for j in range(4):
                left_sum += left_ref_corrected[i, j]
                right_sum += right_ref_corrected[i, j]
                count += 1
        
        left_avg = left_sum / count
        right_avg = right_sum / count
        lr_correction = (left_avg + right_avg) / 2
        
        for col in range(4, 2044):
            corrected_frame[row, col] -= lr_correction
    
    return corrected_frame

@jit(nopython=True, parallel=True, cache=True)
def batch_reference_correction(frame_stack, x_opt=64, y_opt=4):
    """Batch reference pixel correction using parallel processing."""
    n_frames, height, width = frame_stack.shape
    corrected_stack = np.empty_like(frame_stack)
    
    for frame_idx in prange(n_frames):
        corrected_stack[frame_idx] = ReferencePixelCorrector._subtract_reference_pixels_numba(
            frame_stack[frame_idx], x_opt, y_opt
        )
    
    return corrected_stack
