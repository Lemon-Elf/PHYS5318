import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import csv

class Displacement:
    def __init__(self, start_pos, end_pos, min_pos, max_pos, avg_frame_to_frame_displacement, 
                 total_displacement, range_of_motion, start_pos_uncertainty=0, end_pos_uncertainty=0,
                 min_pos_uncertainty=0, max_pos_uncertainty=0, avg_frame_to_frame_displacement_uncertainty=0,
                 total_displacement_uncertainty=0, range_of_motion_uncertainty=0):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.avg_frame_to_frame_displacement = avg_frame_to_frame_displacement
        self.total_displacement = total_displacement
        self.range_of_motion = range_of_motion
        self.start_pos_uncertainty = start_pos_uncertainty
        self.end_pos_uncertainty = end_pos_uncertainty
        self.min_pos_uncertainty = min_pos_uncertainty
        self.max_pos_uncertainty = max_pos_uncertainty
        self.avg_frame_to_frame_displacement_uncertainty = avg_frame_to_frame_displacement_uncertainty
        self.total_displacement_uncertainty = total_displacement_uncertainty
        self.range_of_motion_uncertainty = range_of_motion_uncertainty

class Video:
    def __init__(self, video_path: str, flow_rate: float, index: int):
        self.flow_rate = flow_rate
        self.video_path = video_path
        self.index = index
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.pixel_size = 0  # microns per pixel, to be calculated later
        self.pixel_size_uncertainty = 0  # Uncertainty in microns per pixel, to be calculated later
        self.frames = None  # Will hold the grayscale frames as a numpy array
        self.max_intensity_positions = None  # To store max intensity positions across frames
        self.cleaned_zscore = None  # To store cleaned max intensity positions after outlier removal
        self.displacement: Displacement = None  # To store calculated displacementsm
        self.velocity = None  # To store the velocity at the selected minimum
        self.velocity_uncertainty = None  # To store the uncertainty in velocity

def get_video_frames(video: Video):
    """
    Extract frames from the video and convert to grayscale.
    Will modify the video objects attributes.
    """
    # Open up a video for test run.
    cap = cv2.VideoCapture(video.video_path)  # Use the specified video file

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)          # Should be 30 fps per the lab
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract frames
    frames = []
    while True: 
        ret, frame = cap.read()   # ret=False when video ends
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(gray)

    cap.release()
    frames = np.array(frames)  # shape: (N_frames, height, width)

    # Update video object attributes
    video.fps = fps
    video.width = width
    video.height = height
    video.total_frames = len(frames)  # Use actual extracted frames, not metadata
    video.frames = frames
    print(f"Loaded video '{video.video_path}' with {len(frames)} frames at {fps} fps, " +\
          f"resolution {width}x{height}")
    return

def display_frame(video: Video)->None:
    # Display multiple frames in a grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    frames_to_show = [0, video.total_frames // 3, 2 * video.total_frames // 3, 
                      video.total_frames - 1] # Equally spaced frame.

    for idx, ax in enumerate(axes.flat):
        if idx < len(frames_to_show):
            frame_num = frames_to_show[idx]
            ax.imshow(video.frames[frame_num], cmap='gray')
            ax.set_title(f"Frame {frame_num}")
        ax.axis('off')

    plt.suptitle(f"Float Rate: {video.flow_rate}, index: {video.index}", fontsize=16)
    plt.tight_layout()
    save_path = f"./figures/display_frame_{video.flow_rate}_{video.index}.png"
    plt.savefig(save_path)
    print(f"Saved frame display to {save_path}")
    plt.close()
    return

def range_of_row(video: Video)->None:
    # Select a range of rows from multiple frames and display them
    start_row = 240    # Starting row index
    end_row = 270      # Ending row index

    frames_to_show = [i * video.total_frames // 6 for i in range(6)]  

    # Extract the row range from all frames
    frame_region = video.frames[:, start_row:end_row, :]

    # Display selected frames with the row range
    fig, axes = plt.subplots(6, 1, figsize=(14, 10))

    for idx, ax in enumerate(axes.flat):
        if idx < len(frames_to_show):
            frame_num = frames_to_show[idx]
            ax.imshow(frame_region[frame_num], cmap='gray')
            ax.set_title(f"Frame {frame_num}")
        ax.axis('off')

    plt.suptitle(f"Flow Rate: {video.flow_rate}, index: {video.index}, " + \
                 f"Row Range ({start_row}-{end_row})")
    plt.tight_layout()
    plt.savefig(f"./figures/range_of_row_{video.flow_rate}_{video.index}.png")
    print(f"Saved range of row display to ./figures/range_of_row_{video.flow_rate}_{video.index}.png")
    plt.close()
    return

def peak_region(video: Video)->None:
    start_row = 240    # Starting row index
    end_row = 270      # Ending row index

    # For this one frame, plot the intensity along one row over column position
    frame_region = video.frames[0, start_row:end_row, :]
    row_index = 15  # Example row index
    intensity_profile = frame_region[row_index, :]  # Get intensity values across columns for this row
    
    background_level = np.percentile(intensity_profile, 25)  # Lower quartile as background
    threshold = background_level + 0.5 * (np.max(intensity_profile) - background_level)

    # Find columns where intensity exceeds threshold
    peak_columns = np.where(intensity_profile > threshold)[0]
    
    bead_diameter = 4.16  # microns, as given in the lab
    bead_uncertainity = 0.21
    pixel_size = bead_diameter / (peak_columns[-1] - peak_columns[0])  # microns per pixel
    pixel_size_uncertainty = bead_uncertainity / (peak_columns[-1] - peak_columns[0])  
    video.pixel_size = pixel_size  # Store pixel size in video object for later use
    video.pixel_size_uncertainty = pixel_size_uncertainty 
    print(f"Peak columns (above threshold): {peak_columns}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(intensity_profile)
    # Draw two vertial lines at the column indices of the peak region
    plt.axvline(x=peak_columns[0], color='r', linestyle='--', label='Peak Region')
    plt.axvline(x=peak_columns[-1], color='r', linestyle='--')
    plt.legend()
    plt.title(f"Intensity Profile for Frame 0, Row {start_row + row_index}")
    plt.xlabel("Column Index")
    plt.ylabel("Intensity")
    plt.grid()
    plt.savefig(f"./figures/peak_region_{video.flow_rate}_{video.index}.png")
    plt.close()
    print(f"Saved intensity profile to ./figures/peak_region_{video.flow_rate}_{video.index}.png")
    return

def gaussian_smoothing(video: Video)->None:
    start_row = 240    # Starting row index
    end_row = 270      # Ending row index
    frame_region_single = video.frames[0, start_row:end_row, :]
    row_index = 15
    intensity_profile = frame_region_single[row_index, :]

    # Apply different smoothing filters
    sigma = 5  # Standard deviation for Gaussian filter
    smoothed_gaussian = gaussian_filter1d(intensity_profile, sigma=sigma)

    # Find peaks
    peak_gaussian = np.argmax(smoothed_gaussian)

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(intensity_profile, 'gray', label='Original', alpha=0.5)
    plt.plot(smoothed_gaussian, 'g-', label='Gaussian Smoothed', linewidth=2)
    plt.axvline(x=peak_gaussian, color='g', linestyle='--', label=f'Peak at {peak_gaussian}')
    plt.title(f'Gaussian Filter (σ={sigma})')
    plt.xlabel('Column Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./figures/gaussian_smoothing_{video.flow_rate}_{video.index}.png")
    print(f"Saved Gaussian smoothing plot to ./figures/gaussian_smoothing_{video.flow_rate}_{video.index}.png")
    plt.close()
    return

def max_intensities(video: Video)->None:
    # Analyze all frames and find max intensity position for each frame
    start_row = 240    # Starting row index
    end_row = 270      # Ending row index
    sigma = 5          # Standard deviation for Gaussian smoothing
    frame_region = video.frames[:, start_row:end_row, :]
    row_index = 15  # Example row index

    # Track the column position of maximum intensity for each frame
    max_intensity_positions = []

    for frame_num in range(len(frame_region)):
        intensity_profile = frame_region[frame_num, row_index, :]
        smoothed_gaussian = gaussian_filter1d(intensity_profile, sigma=sigma)
        max_col_index = np.argmax(smoothed_gaussian)
        max_intensity_positions.append(max_col_index)

    max_intensity_positions = np.array(max_intensity_positions)
    video.max_intensity_positions = max_intensity_positions  # Store in video object for later use

    # Plot: Frame number (x-axis) vs Max intensity position (y-axis)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(max_intensity_positions)), max_intensity_positions, marker='o', markersize=4)
    plt.title(f"Position of Maximum Intensity Across Frames (Row {start_row + row_index})")
    plt.xlabel("Frame Number")
    plt.ylabel("Column Index (Position)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./figures/max_intensities_{video.flow_rate}_{video.index}.png")
    print(f"Saved max intensity positions plot to ./figures/max_intensities_{video.flow_rate}_{video.index}.png")
    plt.close()

def methods_comparison(video: Video)->None:
    # Different methods to remove outliers
    from scipy.signal import medfilt
    max_intensity_positions = video.max_intensity_positions
    # Method 1: Median filter (removes spikes/outliers)
    median_filtered = medfilt(max_intensity_positions, kernel_size=5)

    # Method 2: IQR (Interquartile Range) method
    Q1 = np.percentile(max_intensity_positions, 25)
    Q3 = np.percentile(max_intensity_positions, 75)
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR
    cleaned_iqr = max_intensity_positions.copy()
    outlier_mask = (cleaned_iqr < Q1 - outlier_threshold) | (cleaned_iqr > Q3 + outlier_threshold)
    cleaned_iqr[outlier_mask] = np.median(max_intensity_positions)

    # Method 3: Z-score method (remove points > 2 std from mean)
    z_scores = np.abs((max_intensity_positions - np.mean(max_intensity_positions)) / np.std(max_intensity_positions))
    cleaned_zscore = max_intensity_positions.copy()
    cleaned_zscore[z_scores > 2] = np.median(max_intensity_positions) # Replace outliers with median
    video.cleaned_zscore = cleaned_zscore  # Store cleaned data in video object for later use

    # Method 4: Combination - Median filter + Savitzky-Golay
    median_then_savgol = savgol_filter(medfilt(max_intensity_positions, kernel_size=5), window_length=11, polyorder=1)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(range(len(max_intensity_positions)), max_intensity_positions, 'o', markersize=4, label='Original')
    axes[0, 0].plot(range(len(median_filtered)), median_filtered, 'r-', linewidth=2, label='Median Filter (k=5)')
    axes[0, 0].set_title('Median Filter - Removes Sharp Spikes')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Position (pixels)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(range(len(max_intensity_positions)), max_intensity_positions, 'o', markersize=4, label='Original')
    axes[0, 1].plot(range(len(cleaned_iqr)), cleaned_iqr, 'g-', linewidth=2, label='IQR Method')
    axes[0, 1].scatter(np.where(outlier_mask)[0], max_intensity_positions[outlier_mask], color='red', s=50, label='Detected Outliers')
    axes[0, 1].set_title('IQR Method - Replace Outliers with Median')
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Position (pixels)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(len(max_intensity_positions)), max_intensity_positions, 'o', markersize=4, label='Original')
    axes[1, 0].plot(range(len(cleaned_zscore)), cleaned_zscore, 'orange', linewidth=2, label='Z-Score Method')
    axes[1, 0].scatter(np.where(z_scores > 2)[0], max_intensity_positions[z_scores > 2], color='red', s=50, label='Detected Outliers')
    axes[1, 0].set_title('Z-Score Method (|z| > 2)')
    axes[1, 0].set_xlabel('Frame Number')
    axes[1, 0].set_ylabel('Position (pixels)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(range(len(max_intensity_positions)), max_intensity_positions, 'o', markersize=4, alpha=0.5, label='Original')
    axes[1, 1].plot(range(len(median_then_savgol)), median_then_savgol, 'purple', linewidth=2, label='Median + Savitzky-Golay')
    axes[1, 1].set_title('Combination Method - Best of Both')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('Position (pixels)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Outlier Detection Methods Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"./figures/methods_comparison_{video.flow_rate}_{video.index}.png")
    plt.close()
    print(f"Saved methods comparison plot to ./figures/methods_comparison_{video.flow_rate}_{video.index}.png")

def delta_d(video: Video)->None:
    # Calculate displacement using Z-score cleaned data
    # Frame-to-frame displacement
    cleaned_zscore = video.cleaned_zscore
    displacements_cleaned = np.diff(cleaned_zscore)
    frame_to_frame_displacement_cleaned = np.mean(np.abs(displacements_cleaned))

    # Total displacement (overall change from start to end)
    total_displacement_cleaned = cleaned_zscore[-1] - cleaned_zscore[0]

    # Range of motion (max - min)
    range_of_motion_cleaned = np.max(cleaned_zscore) - np.min(cleaned_zscore)

    # Propagate pixel size uncertainty: sigma = |position_in_pixels| * sigma_pixel_size
    start_pos_uncertainty = np.abs(cleaned_zscore[0]) * video.pixel_size_uncertainty
    end_pos_uncertainty = np.abs(cleaned_zscore[-1]) * video.pixel_size_uncertainty
    min_pos_uncertainty = np.abs(np.min(cleaned_zscore)) * video.pixel_size_uncertainty
    max_pos_uncertainty = np.abs(np.max(cleaned_zscore)) * video.pixel_size_uncertainty
    avg_frame_to_frame_displacement_uncertainty = frame_to_frame_displacement_cleaned * video.pixel_size_uncertainty
    total_displacement_uncertainty = np.abs(total_displacement_cleaned) * video.pixel_size_uncertainty
    range_of_motion_uncertainty = range_of_motion_cleaned * video.pixel_size_uncertainty

    disp = Displacement(
        start_pos=cleaned_zscore[0] * video.pixel_size,
        end_pos=cleaned_zscore[-1] * video.pixel_size,  
        min_pos=np.min(cleaned_zscore) * video.pixel_size,   
        max_pos=np.max(cleaned_zscore) * video.pixel_size,   
        avg_frame_to_frame_displacement=frame_to_frame_displacement_cleaned * video.pixel_size,
        total_displacement=total_displacement_cleaned * video.pixel_size,
        range_of_motion=range_of_motion_cleaned * video.pixel_size,
        start_pos_uncertainty=start_pos_uncertainty,
        end_pos_uncertainty=end_pos_uncertainty,
        min_pos_uncertainty=min_pos_uncertainty,
        max_pos_uncertainty=max_pos_uncertainty,
        avg_frame_to_frame_displacement_uncertainty=avg_frame_to_frame_displacement_uncertainty,
        total_displacement_uncertainty=total_displacement_uncertainty,
        range_of_motion_uncertainty=range_of_motion_uncertainty
    )

    video.displacement = disp  # Store displacement in video object for later use

# Check discontinuity: a minimum is discontinuous if it has no neighbors
# OR if the data around it is sparse
def _is_discontinuous(index, velocity_array, window=5):
    """
    Check if a point is discontinuous (isolated).
    A point is discontinuous if it has little to no variation around it
    or if the neighboring values are far from it.
    """
    if index - window < 0 or index + window >= len(velocity_array):
        return True  # At boundary
    
    left_region = velocity_array[max(0, index-window):index]
    right_region = velocity_array[index+1:min(len(velocity_array), index+window+1)]
    
    # If either neighbor region is empty or very small, it's discontinuous
    if len(left_region) == 0 or len(right_region) == 0:
        return True
    
    # Check if there's a sharp discontinuity (big gap from minimum)
    left_diff = np.min(np.abs(left_region - velocity_array[index]))
    right_diff = np.min(np.abs(right_region - velocity_array[index]))
    
    # If minimum gap is large (point is isolated), it's discontinuous
    threshold = 5.0  # pixels/sec - adjust as needed
    if min(left_diff, right_diff) > threshold:
        return True
    
    return False

def velocity(video: Video)->None:
    # Calculate velocity directly from Savitzky-Golay fit
    # Since Savitzky-Golay preserves derivatives, we can use deriv() parameter

    # Option 1: Use savgol_filter to get velocity directly (first derivative)
    cleaned_zscore = video.cleaned_zscore
    time_per_frame = 1 / video.fps
    pixel_size = video.pixel_size
    window_length = 46  # Points to use for local fitting (51 frames = ~1.7 sec)
    polyorder = 3       # Polynomial order

    savgol_position = savgol_filter(cleaned_zscore, window_length=window_length, polyorder=polyorder)
    
    velocity_savgol_pixels = savgol_filter(cleaned_zscore, window_length=window_length, 
                                           polyorder=polyorder, deriv=1)
    
    # deriv=1 gives us dPosition/dFrame, so divide by time_per_frame
    velocity_savgol_per_sec = velocity_savgol_pixels / time_per_frame

    # Convert to microns/second
    # Plot position, velocity, and acceleration
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    time_array = np.arange(len(cleaned_zscore)) * time_per_frame 

    # Find minima by finding peaks in the negative velocity curve
    neg_velocity = -velocity_savgol_per_sec
    minima_peaks, minima_props = find_peaks(neg_velocity, distance=5)  # distance to avoid cluttering

    # Get the actual velocity values at minima positions
    velocity_at_minima = velocity_savgol_per_sec[minima_peaks]

    # Sort minima by value (most negative first) to get global and second global
    sorted_indices = np.argsort(velocity_at_minima)
    sorted_minima_indices = minima_peaks[sorted_indices]
    sorted_minima_values = velocity_at_minima[sorted_indices]

    # Find the best minimum (global or second global)
    selected_minimum_idx = None
    selected_minimum_value = None

    for global_rank, sorted_idx in enumerate(sorted_minima_indices):
        if not _is_discontinuous(sorted_idx, velocity_savgol_per_sec, window=5):
            selected_minimum_idx = sorted_idx
            selected_minimum_value = velocity_savgol_per_sec[sorted_idx]
            break
    else:
        # If all minima are discontinuous, fall back to the global minimum (most negative)
        if len(sorted_minima_indices) > 0:
            selected_minimum_idx = sorted_minima_indices[0]
            selected_minimum_value = sorted_minima_values[0]

    # Propagate pixel size uncertainty through velocity: sigma_v = |position_in_pixels| * sigma_pixel_size / time_per_frame
    # Since velocity = (position_in_pixels / time_per_frame) * pixel_size
    # And position_in_pixels has no uncertainty (it's measured), only pixel_size does
    # sigma_velocity = |selected_minimum_value| * video.pixel_size_uncertainty / time_per_frame
    velocity_uncertainty = np.abs(selected_minimum_value) * video.pixel_size_uncertainty / time_per_frame
    
    video.velocity = selected_minimum_value * video.pixel_size # Store selected minimum value (in um/sec) in video object
    video.velocity_uncertainty = velocity_uncertainty # Store velocity uncertainty in video object

    # Plot all minima and highlight the selected one
    # Plot position, velocity, and acceleration
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Position
    axes[0].scatter(time_array, cleaned_zscore, alpha=0.4, s=30, color='blue')
    axes[0].plot(time_array, savgol_position, 'r-', linewidth=2.5, label='Savitzky-Golay Fit')
    axes[0].set_ylabel('Position (pixels)', fontsize=12)
    axes[0].set_title('Position vs Time', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_array, velocity_savgol_per_sec, 'g-', linewidth=2.5, label='Velocity')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot all detected minima
    axes[1].scatter(minima_peaks * time_per_frame, velocity_at_minima, 
            color='blue', s=80, alpha=0.6, label='Detected Minima', zorder=3)

    # Highlight the selected minimum
    if selected_minimum_idx is not None:
        axes[1].scatter(selected_minimum_idx * time_per_frame, selected_minimum_value, 
                color='red', s=200, marker='*', label='Selected Minimum', zorder=5, edgecolors='darkred', linewidth=2)

    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Velocity (pixels/sec)', fontsize=12)
    axes[1].set_title("Velocity vs Time with Detected Minima", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()  
    plt.savefig(f"./figures/velocity_{video.flow_rate}_{video.index}.png")
    print(f"Saved velocity plot to ./figures/velocity_{video.flow_rate}_{video.index}.png")
    plt.close()
    return


def main():
    # Orgamize the data into array.
    video_files = glob.glob("./lab_4/Bead_004*.mpg")  # Adjust the path as needed
    video_files.sort()  # Ensure files are in order
    videos = []
    this_flow = []
    count = 0
    for i in range(len(video_files)):  # Process every other video
        this_video = video_files[i][:-4].split("_")  # Get filename without path
        flow_rate = int(this_video[3][:3])  # Extract flow rate from filename
        index = int(this_video[-1][:2])  # Extract index from filename
        video = Video(video_files[i], flow_rate, index)

        if count != 3: 
            count += 1
            this_flow.append(video)  # Extract flow rate from filename
        else:
            count = 0
            videos.append(this_flow)
            this_flow = []
            this_flow.append(video)  # Start new flow group with current video
            count += 1

    # Begin analysis for each video
    for flow_videos in videos:
        for video in flow_videos:
            print(f"Processing: Flow Rate: {video.flow_rate}, Index: {video.index}")
            get_video_frames(video)
            display_frame(video)
            range_of_row(video)
            peak_region(video)
            gaussian_smoothing(video)
            max_intensities(video)
            methods_comparison(video)
            delta_d(video)
            velocity(video)
            print(f"Displacement for Flow Rate {video.flow_rate}, Index {video.index}: " +\
                  f"{video.displacement.range_of_motion:.2f} ± {video.displacement.range_of_motion_uncertainty:.2f} microns")
            print(f"Velocity: {video.velocity:.2f} ± {video.velocity_uncertainty:.2f} μm/sec")
            print() 
    
    # Save results to CSV file
    csv_filename = "./figures/results_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Flow Rate (μL/min)', 'Video Index', 'Range of Motion (microns)', 
                      'Range of Motion Uncertainty (microns)', 'Velocity (μm/sec)', 'Velocity Uncertainty (μm/sec)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for flow_videos in videos:
            for video in flow_videos:
                writer.writerow({
                    'Flow Rate (μL/min)': flow_videos[0].flow_rate,
                    'Video Index': video.index,
                    'Range of Motion (microns)': f"{video.displacement.range_of_motion:.2f}",
                    'Range of Motion Uncertainty (microns)': f"{video.displacement.range_of_motion_uncertainty:.2f}",
                    'Velocity (μm/sec)': f"{video.velocity:.2f}",
                    'Velocity Uncertainty (μm/sec)': f"{video.velocity_uncertainty:.2f}"
                })
    
    print(f"Results saved to {csv_filename}")
    print("\nSummary of Range of Motion for All Videos:")
    for flow_videos in videos:
        print(f"Flow Rate: {flow_videos[0].flow_rate} μL/min")
        print(f"Range of motion: [{flow_videos[0].displacement.range_of_motion:.2f} ± {flow_videos[0].displacement.range_of_motion_uncertainty:.2f} microns, " +\
              f"{flow_videos[1].displacement.range_of_motion:.2f} ± {flow_videos[1].displacement.range_of_motion_uncertainty:.2f} microns, " +\
              f"{flow_videos[-1].displacement.range_of_motion:.2f} ± {flow_videos[-1].displacement.range_of_motion_uncertainty:.2f} microns]")
        print(f"Velocities: [{flow_videos[0].velocity:.2f} ± {flow_videos[0].velocity_uncertainty:.2f} μm/sec, " +\
              f"{flow_videos[1].velocity:.2f} ± {flow_videos[1].velocity_uncertainty:.2f} μm/sec, " +\
              f"{flow_videos[-1].velocity:.2f} ± {flow_videos[-1].velocity_uncertainty:.2f} μm/sec]\n")

    return

if __name__ == "__main__":
    main()

