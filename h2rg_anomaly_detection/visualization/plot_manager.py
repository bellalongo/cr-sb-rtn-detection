import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import logging
import seaborn as sns

# class PlotManager:
#     """Manages visualization of anomaly detection results."""
    
#     def __init__(self, config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
#         plt.style.use('seaborn-v0_8-darkgrid')
    
#     def plot_exposure_summary(self, exposure_results: Dict, 
#                             diff_stack: np.ndarray,
#                             temporal_data: Dict,
#                             save_path: Optional[str] = None):
#         """Create comprehensive visualization for single exposure."""
#         fig = plt.figure(figsize=(20, 12), dpi=self.config.plot_dpi)
#         gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
#         # 1. First appearance map
#         ax1 = fig.add_subplot(gs[0, 0])
#         im1 = ax1.imshow(temporal_data['first_appearance'], 
#                         cmap='viridis', aspect='auto')
#         ax1.set_title('First Appearance Frame')
#         plt.colorbar(im1, ax=ax1, label='Frame')
        
#         # 2. Max intensity map
#         ax2 = fig.add_subplot(gs[0, 1])
#         im2 = ax2.imshow(temporal_data['max_intensity'], 
#                         cmap='hot', aspect='auto')
#         ax2.set_title('Maximum Intensity')
#         plt.colorbar(im2, ax=ax2, label='DN')
        
#         # 3. Persistence map
#         ax3 = fig.add_subplot(gs[0, 2])
#         im3 = ax3.imshow(temporal_data['persistence_count'], 
#                         cmap='plasma', aspect='auto')
#         ax3.set_title('Persistence Count')
#         plt.colorbar(im3, ax=ax3, label='Frames')
        
#         # 4. Anomaly classification map
#         ax4 = fig.add_subplot(gs[0, 3])
#         classification_map = self._create_classification_map(exposure_results)
#         im4 = ax4.imshow(classification_map, cmap='tab10', aspect='auto')
#         ax4.set_title('Anomaly Classification')
        
#         # Create legend for classifications
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor=self.config.anomaly_colors['cosmic_ray'], 
#                   label=f"Cosmic Rays ({exposure_results['summary']['cosmic_rays']})"),
#             Patch(facecolor=self.config.anomaly_colors['snowball'], 
#                   label=f"Snowballs ({exposure_results['summary']['snowballs']})"),
#             Patch(facecolor=self.config.anomaly_colors['telegraph_noise'], 
#                   label=f"RTN ({exposure_results['summary']['telegraph_noise']})"),
#             Patch(facecolor=self.config.anomaly_colors['unknown'], 
#                   label=f"Unknown ({exposure_results['summary']['unknown']})")
#         ]
#         ax4.legend(handles=legend_elements, loc='center left', 
#                   bbox_to_anchor=(1, 0.5))
        
#         # 5. Example cosmic ray
#         ax5 = fig.add_subplot(gs[1, 0])
#         self._plot_example_event(ax5, exposure_results['cosmic_rays'], 
#                                diff_stack, 'Cosmic Ray Example')
        
#         # 6. Example snowball
#         ax6 = fig.add_subplot(gs[1, 1])
#         self._plot_example_event(ax6, exposure_results['snowballs'], 
#                                diff_stack, 'Snowball Example')
        
#         # 7. Example RTN time series
#         ax7 = fig.add_subplot(gs[1, 2:])
#         self._plot_rtn_example(ax7, exposure_results['telegraph_noise'], 
#                              diff_stack)
        
#         # 8. Temporal evolution
#         ax8 = fig.add_subplot(gs[2, :2])
#         self._plot_temporal_evolution(ax8, temporal_data)
        
#         # 9. Statistics
#         ax9 = fig.add_subplot(gs[2, 2:])
#         self._plot_statistics(ax9, exposure_results)
        
#         plt.suptitle(f"H2RG Anomaly Detection Results - Exposure {exposure_results.get('exposure_id', 0)}", 
#                     fontsize=16)
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=self.config.plot_dpi)
#             self.logger.info(f"Saved plot to {save_path}")
        
#         plt.show()
    
#     def plot_multi_exposure_summary(self, combined_results: Dict,
#                                   save_path: Optional[str] = None):
#         """Create summary visualization for multiple exposures."""
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.config.plot_dpi)
        
#         # 1. Anomaly counts per exposure
#         ax1 = axes[0, 0]
#         exposures = [r['exposure_id'] for r in combined_results['exposures']]
#         cosmic_rays = [r['summary']['cosmic_rays'] for r in combined_results['exposures']]
#         snowballs = [r['summary']['snowballs'] for r in combined_results['exposures']]
#         rtn = [r['summary']['telegraph_noise'] for r in combined_results['exposures']]
        
#         x = np.arange(len(exposures))
#         width = 0.25
        
#         ax1.bar(x - width, cosmic_rays, width, label='Cosmic Rays', 
#                color=self.config.anomaly_colors['cosmic_ray'])
#         ax1.bar(x, snowballs, width, label='Snowballs',
#                color=self.config.anomaly_colors['snowball'])
#         ax1.bar(x + width, rtn, width, label='RTN',
#                color=self.config.anomaly_colors['telegraph_noise'])
        
#         ax1.set_xlabel('Exposure')
#         ax1.set_ylabel('Count')
#         ax1.set_title('Anomaly Counts by Exposure')
#         ax1.set_xticks(x)
#         ax1.set_xticklabels(exposures)
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # 2. Hot pixel distribution
#         ax2 = axes[0, 1]
#         if combined_results['hot_pixels']:
#             hot_pixel_intensities = [hp['mean_intensity'] 
#                                    for hp in combined_results['hot_pixels']]
#             ax2.hist(hot_pixel_intensities, bins=30, 
#                     color=self.config.anomaly_colors['hot_pixel'],
#                     edgecolor='black', alpha=0.7)
#             ax2.set_xlabel('Mean Intensity (DN)')
#             ax2.set_ylabel('Count')
#             ax2.set_title(f"Hot Pixel Intensity Distribution (n={len(combined_results['hot_pixels'])})")
#         else:
#             ax2.text(0.5, 0.5, 'No hot pixels detected', 
#                     ha='center', va='center', transform=ax2.transAxes)
#             ax2.set_title('Hot Pixel Distribution')
#         ax2.grid(True, alpha=0.3)
        
#         # 3. Summary statistics
#         ax3 = axes[1, 0]
#         ax3.axis('off')
#         summary = combined_results['summary']
#         stats_text = f"""Summary Statistics:
        
# Total Exposures: {summary['total_exposures']}
# Total Cosmic Rays: {summary['total_cosmic_rays']} (avg: {summary['avg_cosmic_rays_per_exposure']:.1f}/exp)
# Total Snowballs: {summary['total_snowballs']} (avg: {summary['avg_snowballs_per_exposure']:.1f}/exp)
# Total RTN: {summary['total_telegraph_noise']} (avg: {summary['avg_rtn_per_exposure']:.1f}/exp)
# Total Hot Pixels: {summary['total_hot_pixels']}
# Total Unknown: {summary['total_unknown']}"""
        
#         ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
#                 fontsize=12, verticalalignment='top', fontfamily='monospace')
        
#         # 4. Classification pie chart
#         ax4 = axes[1, 1]
#         labels = []
#         sizes = []
#         colors = []
        
#         if summary['total_cosmic_rays'] > 0:
#             labels.append(f"Cosmic Rays\n({summary['total_cosmic_rays']})")
#             sizes.append(summary['total_cosmic_rays'])
#             colors.append(self.config.anomaly_colors['cosmic_ray'])
        
#         if summary['total_snowballs'] > 0:
#             labels.append(f"Snowballs\n({summary['total_snowballs']})")
#             sizes.append(summary['total_snowballs'])
#             colors.append(self.config.anomaly_colors['snowball'])
        
#         if summary['total_telegraph_noise'] > 0:
#             labels.append(f"RTN\n({summary['total_telegraph_noise']})")
#             sizes.append(summary['total_telegraph_noise'])
#             colors.append(self.config.anomaly_colors['telegraph_noise'])
        
#         if summary['total_hot_pixels'] > 0:
#             labels.append(f"Hot Pixels\n({summary['total_hot_pixels']})")
#             sizes.append(summary['total_hot_pixels'])
#             colors.append(self.config.anomaly_colors['hot_pixel'])
        
#         if summary['total_unknown'] > 0:
#             labels.append(f"Unknown\n({summary['total_unknown']})")
#             sizes.append(summary['total_unknown'])
#             colors.append(self.config.anomaly_colors['unknown'])
        
#         if sizes:
#             ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
#                    startangle=90)
#             ax4.set_title('Overall Anomaly Distribution')
#         else:
#             ax4.text(0.5, 0.5, 'No anomalies detected', 
#                     ha='center', va='center', transform=ax4.transAxes)
#             ax4.set_title('Anomaly Distribution')
        
#         plt.suptitle('H2RG Multi-Exposure Anomaly Detection Summary', fontsize=16)
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=self.config.plot_dpi)
#             self.logger.info(f"Saved multi-exposure plot to {save_path}")
        
#         plt.show()
    
#     def _create_classification_map(self, exposure_results: Dict) -> np.ndarray:
#         """Create a map showing classified anomalies."""
#         # Initialize with zeros (background)
#         class_map = np.zeros((2048, 2048))
        
#         # Assign different values for different anomaly types
#         for cr in exposure_results['cosmic_rays']:
#             if 'mask' in cr:
#                 class_map[cr['mask']] = 1
#             elif 'pixel_coords' in cr:
#                 for y, x in cr['pixel_coords']:
#                     class_map[y, x] = 1
        
#         for sb in exposure_results['snowballs']:
#             if 'mask' in sb:
#                 class_map[sb['mask']] = 2
        
#         for rtn in exposure_results['telegraph_noise']:
#             y, x = rtn['position']
#             class_map[y, x] = 3
        
#         for unk in exposure_results['unknown']:
#             y, x = unk['position']
#             class_map[y, x] = 4
        
#         return class_map
    
#     def _plot_example_event(self, ax, events: List[Dict], 
#                           diff_stack: np.ndarray, title: str):
#         """Plot an example of a specific event type."""
#         if events:
#             event = events[0]  # Take first example
            
#             if 'centroid' in event:
#                 y, x = int(event['centroid'][0]), int(event['centroid'][1])
#             elif 'position' in event:
#                 y, x = event['position']
#             else:
#                 ax.text(0.5, 0.5, f'No {title} found', 
#                        ha='center', va='center', transform=ax.transAxes)
#                 ax.set_title(title)
#                 return
            
#             # Extract region around event
#             radius = 20
#             y_start = max(0, y - radius)
#             y_end = min(diff_stack.shape[1], y + radius)
#             x_start = max(0, x - radius)
#             x_end = min(diff_stack.shape[2], x + radius)
            
#             # Use frame where event first appears
#             frame_idx = event.get('first_frame', diff_stack.shape[0] // 2)
#             region = diff_stack[frame_idx, y_start:y_end, x_start:x_end]
            
#             im = ax.imshow(region, cmap='hot', aspect='auto')
#             ax.set_title(f"{title} (Frame {frame_idx})")
#             plt.colorbar(im, ax=ax, label='DN')
            
#             # Mark center
#             center_y = y - y_start
#             center_x = x - x_start
#             ax.plot(center_x, center_y, 'c+', markersize=10, markeredgewidth=2)
#         else:
#             ax.text(0.5, 0.5, f'No {title} found', 
#                    ha='center', va='center', transform=ax.transAxes)
#             ax.set_title(title)
    
#     def _plot_rtn_example(self, ax, rtn_events: List[Dict], 
#                         diff_stack: np.ndarray):
#         """Plot RTN time series example."""
#         if rtn_events:
#             event = rtn_events[0]
#             time_series = event.get('time_series', None)
            
#             if time_series is not None:
#                 frames = np.arange(len(time_series))
#                 ax.plot(frames, time_series, 'b-', linewidth=0.5, alpha=0.7)
#                 ax.axhline(y=event['high_state_value'], color='r', 
#                           linestyle='--', label=f"High: {event['high_state_value']:.1f}")
#                 ax.axhline(y=event['low_state_value'], color='g', 
#                           linestyle='--', label=f"Low: {event['low_state_value']:.1f}")
#                 ax.set_xlabel('Frame')
#                 ax.set_ylabel('Signal (DN)')
#                 ax.set_title(f"RTN Example - Amplitude: {event['amplitude']:.1f} DN, "
#                            f"Frequency: {event['frequency']:.3f} Hz")
#                 ax.legend()
#                 ax.grid(True, alpha=0.3)
#         else:
#             ax.text(0.5, 0.5, 'No RTN events found', 
#                    ha='center', va='center', transform=ax.transAxes)
#             ax.set_title('RTN Time Series Example')
    
#     def _plot_temporal_evolution(self, ax, temporal_data: Dict):
#         """Plot temporal evolution of anomalies."""
#         evolution = temporal_data['temporal_evolution']
#         frames = [e['frame'] for e in evolution]
#         n_anomalies = [e['n_anomalies'] for e in evolution]
        
#         ax.plot(frames, n_anomalies, 'b-', linewidth=2)
#         ax.fill_between(frames, n_anomalies, alpha=0.3)
#         ax.set_xlabel('Frame')
#         ax.set_ylabel('Number of Anomalies')
#         ax.set_title('Temporal Evolution of Anomalies')
#         ax.grid(True, alpha=0.3)
    
#     def _plot_statistics(self, ax, exposure_results: Dict):
#         """Plot summary statistics."""
#         ax.axis('off')
        
#         summary = exposure_results['summary']
#         total = summary['total_anomalies']
        
#         stats_text = f"""Anomaly Statistics:
        
# Total Anomalies: {total}
# Cosmic Rays: {summary['cosmic_rays']} ({summary['cosmic_rays']/max(1,total)*100:.1f}%)
# Snowballs: {summary['snowballs']} ({summary['snowballs']/max(1,total)*100:.1f}%)
# Telegraph Noise: {summary['telegraph_noise']} ({summary['telegraph_noise']/max(1,total)*100:.1f}%)
# Unknown: {summary['unknown']} ({summary['unknown']/max(1,total)*100:.1f}%)

# Classification Rate: {summary['classification_rate']*100:.1f}%"""
        
#         ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
#                fontsize=11, verticalalignment='top', fontfamily='monospace',
#                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


class PlotManager:
    """Updated PlotManager with Reds colormap and seaborn white theme."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_exposure_summary(self, exposure_results: Dict, 
                            data_cube: np.ndarray,
                            temporal_data: Dict,
                            save_path: Optional[str] = None):
        """Create comprehensive visualization for single exposure using Reds colormap."""
        # fig = plt.figure(figsize=(10, 10), dpi=self.config.plot_dpi)
        # gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # # 1. First appearance map
        # ax1 = fig.add_subplot(gs[0, 0])
        # im1 = ax1.imshow(temporal_data['first_appearance'], 
        #                 cmap='Reds', aspect='auto')
        # ax1.set_title('First Appearance Frame', fontsize=12, fontweight='bold')
        # cbar1 = plt.colorbar(im1, ax=ax1, label='Frame')
        # cbar1.ax.tick_params(labelsize=10)
        
        # # 2. Max intensity map
        # ax2 = fig.add_subplot(gs[0, 1])
        # im2 = ax2.imshow(temporal_data['max_intensity'], 
        #                 cmap='Reds', aspect='auto')
        # ax2.set_title('Maximum Intensity', fontsize=12, fontweight='bold')
        # cbar2 = plt.colorbar(im2, ax=ax2, label='DN')
        # cbar2.ax.tick_params(labelsize=10)
        
        # # 3. Persistence map
        # ax3 = fig.add_subplot(gs[0, 2])
        # im3 = ax3.imshow(temporal_data['persistence_count'], 
        #                 cmap='Reds', aspect='auto')
        # ax3.set_title('Persistence Count', fontsize=12, fontweight='bold')
        # cbar3 = plt.colorbar(im3, ax=ax3, label='Frames')
        # cbar3.ax.tick_params(labelsize=10)
        
        # # 4. Anomaly classification map
        # ax4 = fig.add_subplot(gs[0, 3])
        # classification_map = self._create_classification_map(exposure_results, data_cube.shape[1:])
        # im4 = ax4.imshow(classification_map, cmap='Reds', aspect='auto')
        # ax4.set_title('Anomaly Classification', fontsize=12, fontweight='bold')
        
        # # Create legend for classifications
        # legend_elements = [
        #     Patch(facecolor=self.config.anomaly_colors['cosmic_ray'], 
        #           label=f"Cosmic Rays ({exposure_results['summary']['cosmic_rays']})"),
        #     Patch(facecolor=self.config.anomaly_colors['snowball'], 
        #           label=f"Snowballs ({exposure_results['summary']['snowballs']})"),
        #     Patch(facecolor=self.config.anomaly_colors['telegraph_noise'], 
        #           label=f"RTN ({exposure_results['summary']['telegraph_noise']})"),
        #     Patch(facecolor=self.config.anomaly_colors['hot_pixel'], 
        #           label=f"Hot Pixels ({exposure_results['summary']['hot_pixels']})")
        # ]
        # ax4.legend(handles=legend_elements, loc='center left', 
        #           bbox_to_anchor=(1, 0.5), fontsize=10)

        """
            DELETE EVERYTHING UNDER ME
        """
        # events = exposure_results['cosmic_rays']
        # if events:
        #     for event in events:
        #         palette = sns.color_palette('Blues', 6)
        #         sns.set_theme(style="white", palette="Blues")

        #         if 'centroid' in event:
        #             y, x = int(event['centroid'][0]), int(event['centroid'][1])
        #         elif 'position' in event:
        #             y, x = event['position']

        #         fig = plt.figure(figsize=(10,10))
                
        #         # Extract region around event
        #         radius = 20
        #         y_start = max(0, y - radius)
        #         y_end = min(data_cube.shape[1], y + radius)
        #         x_start = max(0, x - radius)
        #         x_end = min(data_cube.shape[2], x + radius)
                
        #         # Use frame where event first appears
        #         frame_idx = event.get('first_frame', data_cube.shape[0] // 2)
        #         region = data_cube[frame_idx, y_start:y_end, x_start:x_end]
                
        #         im = plt.imshow(region, cmap='Blues', aspect='auto')
        #         plt.title(f'Cosmic Ray hit at ({y},{x})')
        #         plt.show()

        #         print(event)

        # events = exposure_results['telegraph_noise']
        # if events:
        #     for event in events:
        #         sns.set_theme(style="white", palette="Reds")

        #         if 'centroid' in event:
        #             y, x = int(event['centroid'][0]), int(event['centroid'][1])
        #         elif 'position' in event:
        #             y, x = event['position']

        #         fig = plt.figure(figsize=(10,10))
                
        #         # Extract region around event
        #         radius = 20
        #         y_start = max(0, y - radius)
        #         y_end = min(data_cube.shape[1], y + radius)
        #         x_start = max(0, x - radius)
        #         x_end = min(data_cube.shape[2], x + radius)
                
        #         # Use frame where event first appears
        #         frame_idx = event.get('first_frame', data_cube.shape[0] // 2)
        #         region = data_cube[frame_idx, y_start:y_end, x_start:x_end]

        #         fig, (ax1, ax2) = plt.subplots(5, 10)
                
        #         im = ax1.imshow(region, cmap='cubehelix', aspect='auto')
        #         ax1.set_title(f'Telegraph noise at ({y},{x})')
        #         cbar = plt.colorbar(im, ax = ax1, location='right')



                
        #         plt.show()

        #         print(event)

        # events = exposure_results['telegraph_noise']
        # if events:
        #     for event in events:
        #         sns.set_theme(style="white", palette="Reds")
        #         palette = sns.color_palette('Reds', 6)

        #         if 'centroid' in event:
        #             y, x = int(event['centroid'][0]), int(event['centroid'][1])
        #         elif 'position' in event:
        #             y, x = event['position']

        #         # Create figure with 2 subplots arranged vertically
        #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        #         sns.set_theme(style="white", palette="Reds")
                
        #         # Extract region around event for spatial plot
        #         radius = 20
        #         y_start = max(0, y - radius)
        #         y_end = min(data_cube.shape[1], y + radius)
        #         x_start = max(0, x - radius)
        #         x_end = min(data_cube.shape[2], x + radius)
                
        #         # Use frame where event first appears
        #         frame_idx = event.get('first_frame', data_cube.shape[0] // 2)
        #         region = data_cube[frame_idx, y_start:y_end, x_start:x_end]

        #         # Plot 1: Spatial image of the telegraph noise region
        #         im = ax1.imshow(region, cmap='Reds', aspect='auto')
        #         ax1.set_title(f'Telegraph noise at ({y},{x}) - Frame {frame_idx}')
        #         ax1.set_xlabel('X pixel')
        #         ax1.set_ylabel('Y pixel')
                
        #         cbar = plt.colorbar(im, ax=ax1, label='DN')

        #         # Plot 2: Time series of the telegraph noise pixel
        #         if 'time_series' in event:
        #             time_series = event['time_series']
        #             frames = np.arange(len(time_series))
                    
        #             # Plot the time series
        #             ax2.plot(frames, time_series, linewidth=1, alpha=0.8, label='Pixel Signal', c = palette[3])
                    
        #             # # Add horizontal lines for high and low states if available
        #             # if 'high_state_value' in event:
        #             #     ax2.axhline(y=event['high_state_value'], 
        #             #             linestyle='--', linewidth=2, alpha=0.7,
        #             #             label=f"High: {event['high_state_value']:.1f} DN")
        #             # if 'low_state_value' in event:
        #             #     ax2.axhline(y=event['low_state_value'], 
        #             #             linestyle='--', linewidth=2, alpha=0.7,
        #             #             label=f"Low: {event['low_state_value']:.1f} DN")
                    
        #             ax2.set_xlabel('Frame Index')
        #             ax2.set_ylabel('Pixel Value (DN)')
        #             ax2.set_title(f'Time Series - Pixel ({y},{x})')
        #             ax2.grid(True, alpha=0.3)
        #             ax2.legend()
                    
        #             # Add some statistics to the title if available
        #             title_parts = [f'Time Series - Pixel ({y},{x})']
        #             if 'amplitude' in event:
        #                 title_parts.append(f'Amplitude: {event["amplitude"]:.1f} DN')
        #             if 'frequency' in event:
        #                 title_parts.append(f'Frequency: {event["frequency"]:.3f} Hz')
        #             ax2.set_title(' | '.join(title_parts))
                    
        #         else:
        #             # If no time series data, extract it from data_cube
        #             pixel_time_series = data_cube[:, y, x]
        #             frames = np.arange(len(pixel_time_series))
                    
        #             ax2.plot(frames, pixel_time_series, 'b-', linewidth=1, alpha=0.8)
        #             ax2.set_xlabel('Frame Index')
        #             ax2.set_ylabel('Pixel Value (DN)')
        #             ax2.set_title(f'Time Series - Pixel ({y},{x})')
        #             ax2.grid(True, alpha=0.3)

                
        #         print(event.get('first_frame', data_cube.shape[0] // 2))

        #         plt.tight_layout()
        #         plt.show()

        events = exposure_results['cosmic_rays']
        if events:
            for event in events:
                sns.set_theme(style="white", palette="Blues")
                palette = sns.color_palette('Blues', 6)

                if 'centroid' in event:
                    y, x = int(event['centroid'][0]), int(event['centroid'][1])
                elif 'position' in event:
                    y, x = event['position']

                # Create figure with 2 subplots arranged vertically
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                sns.set_theme(style="white", palette="Blues")
                
                # Extract region around event for spatial plot
                radius = 20
                y_start = max(0, y - radius)
                y_end = min(data_cube.shape[1], y + radius)
                x_start = max(0, x - radius)
                x_end = min(data_cube.shape[2], x + radius)
                
                # Use frame where event first appears
                frame_idx = event.get('first_frame', data_cube.shape[0] // 2)
                region = data_cube[frame_idx, y_start:y_end, x_start:x_end]

                # Plot 1: Spatial image of the telegraph noise region
                im = ax1.imshow(region, cmap='Blues', aspect='auto')
                ax1.set_title(f'Cosmic Ray at ({y},{x}) - Frame {frame_idx}')
                ax1.set_xlabel('X pixel')
                ax1.set_ylabel('Y pixel')
                
                cbar = plt.colorbar(im, ax=ax1, label='DN')

                # Plot 2: Time series of the telegraph noise pixel
                # if 'time_series' in event:
                #     time_series = event['time_series']
                #     frames = np.arange(len(time_series))
                
                # Plot the time series
                time_series = data_cube[:, y, x]
                frames = np.arange(len(data_cube))

                ax2.plot(frames, time_series, linewidth=1, alpha=0.8, label='Pixel Signal', c = palette[3])
                
                # # Add horizontal lines for high and low states if available
                # if 'high_state_value' in event:
                #     ax2.axhline(y=event['high_state_value'], 
                #             linestyle='--', linewidth=2, alpha=0.7,
                #             label=f"High: {event['high_state_value']:.1f} DN")
                # if 'low_state_value' in event:
                #     ax2.axhline(y=event['low_state_value'], 
                #             linestyle='--', linewidth=2, alpha=0.7,
                #             label=f"Low: {event['low_state_value']:.1f} DN")
                
                ax2.set_xlabel('Frame Index')
                ax2.set_ylabel('Pixel Value (DN)')
                ax2.set_title(f'Time Series - Pixel ({y},{x})')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Add some statistics to the title if available
                title_parts = [f'Time Series - Pixel ({y},{x})']
                if 'amplitude' in event:
                    title_parts.append(f'Amplitude: {event["amplitude"]:.1f} DN')
                if 'frequency' in event:
                    title_parts.append(f'Frequency: {event["frequency"]:.3f} Hz')
                ax2.set_title(' | '.join(title_parts))
                    
                # else:
                #     # If no time series data, extract it from data_cube
                #     pixel_time_series = data_cube[:, y, x]
                #     frames = np.arange(len(pixel_time_series))
                    
                #     ax2.plot(frames, pixel_time_series, 'b-', linewidth=1, alpha=0.8)
                #     ax2.set_xlabel('Frame Index')
                #     ax2.set_ylabel('Pixel Value (DN)')
                #     ax2.set_title(f'Time Series - Pixel ({y},{x})')
                #     ax2.grid(True, alpha=0.3)

                
                print(event.get('first_frame', data_cube.shape[0] // 2))

                plt.tight_layout()
                plt.show()

        

        
        # # 5. Example cosmic ray
        # ax5 = fig.add_subplot(gs[1, 0])
        # self._plot_example_event(ax5, exposure_results['cosmic_rays'], 
        #                        data_cube, 'Cosmic Ray Example')
        
        # # 6. Example snowball
        # ax6 = fig.add_subplot(gs[1, 1])
        # self._plot_example_event(ax6, exposure_results['snowballs'], 
        #                        data_cube, 'Snowball Example')
        
        # # 7. Example RTN time series
        # ax7 = fig.add_subplot(gs[1, 2:])
        # self._plot_rtn_example(ax7, exposure_results['telegraph_noise'])
        
        # # 8. Temporal evolution
        # ax8 = fig.add_subplot(gs[2, :2])
        # self._plot_temporal_evolution(ax8, temporal_data)
        
        # # 9. Statistics
        # ax9 = fig.add_subplot(gs[2, 2:])
        # self._plot_statistics(ax9, exposure_results)
        
        plt.suptitle(f"H2RG Anomaly Detection Results", 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.plot_dpi, 
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def _create_classification_map(self, exposure_results: Dict, shape: Tuple) -> np.ndarray:
        """Create a map showing classified anomalies."""
        class_map = np.zeros(shape)
        
        # Assign different values for different anomaly types
        for cr in exposure_results['cosmic_rays']:
            if 'mask' in cr:
                for coord in cr['mask']:
                    class_map[coord[0], coord[1]] = 1
            else:
                y, x = cr['position']
                class_map[y, x] = 1
        
        for sb in exposure_results['snowballs']:
            if 'mask' in sb:
                for coord in sb['mask']:
                    class_map[coord[0], coord[1]] = 2
            else:
                y, x = sb['position']
                class_map[y, x] = 2
        
        for rtn in exposure_results['telegraph_noise']:
            y, x = rtn['position']
            class_map[y, x] = 3
        
        # for hp in exposure_results['hot_pixels']:
        #     y, x = hp['position']
        #     class_map[y, x] = 4
        
        return class_map
    
    def _plot_example_event(self, ax, events: List[Dict], 
                          data_cube: np.ndarray, title: str):
        """Plot an example of a specific event type."""
        if events:
            event = events[0]  # Take first example
            
            if 'centroid' in event:
                y, x = int(event['centroid'][0]), int(event['centroid'][1])
            elif 'position' in event:
                y, x = event['position']
            else:
                ax.text(0.5, 0.5, f'No {title} found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                return
            
            # Extract region around event
            radius = 20
            y_start = max(0, y - radius)
            y_end = min(data_cube.shape[1], y + radius)
            x_start = max(0, x - radius)
            x_end = min(data_cube.shape[2], x + radius)
            
            # Use frame where event first appears
            frame_idx = event.get('first_frame', data_cube.shape[0] // 2)
            region = data_cube[frame_idx, y_start:y_end, x_start:x_end]
            
            im = ax.imshow(region, cmap='Reds', aspect='auto')
            ax.set_title(f"{title} (Frame {frame_idx})", fontsize=12, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, label='DN')
            cbar.ax.tick_params(labelsize=9)
            
            # Mark center
            center_y = y - y_start
            center_x = x - x_start
            # ax.plot(center_x, center_y, 'cyan', marker='+', markersize=10, markeredgewidth=2)
        else:
            ax.text(0.5, 0.5, f'No {title} found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    def _plot_rtn_example(self, ax, rtn_events: List[Dict]):
        """Plot RTN time series example."""
        if rtn_events:
            event = rtn_events[0]
            time_series = event.get('time_series', None)
            
            if time_series is not None:
                frames = np.arange(len(time_series))
                ax.plot(frames, time_series, 'darkred', linewidth=1.5, alpha=0.8)
                ax.axhline(y=event['high_state_value'], color='red', 
                          linestyle='--', linewidth=2, label=f"High: {event['high_state_value']:.1f}")
                ax.axhline(y=event['low_state_value'], color='darkred', 
                          linestyle='--', linewidth=2, label=f"Low: {event['low_state_value']:.1f}")
                ax.set_xlabel('Frame', fontsize=11)
                ax.set_ylabel('Signal (DN)', fontsize=11)
                ax.set_title(f"RTN Example - Amplitude: {event['amplitude']:.1f} DN, "
                           f"Frequency: {event['frequency']:.3f} Hz", 
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No RTN events found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('RTN Time Series Example', fontsize=12, fontweight='bold')
    
    def _plot_temporal_evolution(self, ax, temporal_data: Dict):
        """Plot temporal evolution of anomalies."""
        evolution = temporal_data['temporal_evolution']
        frames = [e['frame'] for e in evolution]
        n_anomalies = [e['n_anomalies'] for e in evolution]
        
        ax.plot(frames, n_anomalies, 'darkred', linewidth=2)
        ax.fill_between(frames, n_anomalies, alpha=0.3, color='red')
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Number of Anomalies', fontsize=11)
        ax.set_title('Temporal Evolution of Anomalies', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    def _plot_statistics(self, ax, exposure_results: Dict):
        """Plot summary statistics."""
        ax.axis('off')
        
        summary = exposure_results['summary']
        total = summary['total_anomalies']
        
        # stats_text = f"""Anomaly Statistics:
