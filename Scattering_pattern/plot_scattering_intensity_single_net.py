### Plot the scattering intensity I(q) vs q #########
import shutil

hkl='010'
test=False
#shutil.copy('./bmn/010/0/Run1/ioLAMMPS.py', '.')
shutil.copy('./bmn/'+hkl+'/0/Run1/relax.py', '.')
shutil.copy('./bmn/'+hkl+'/0/Run1/param.py', '.')


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import ioLAMMPS
import shlex 
from scipy import signal


import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

import time
import math
import random
##import netgen
import ioLAMMPS
import matplotlib
import numpy as np
from relax import Optimizer
from numpy import linalg as LA
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import param as p
from scipy.spatial.transform import Rotation
import multiprocessing
import time
import os
from numba import jit
import shlex
from numba import prange

from numba import njit

from scipy.optimize import curve_fit
from scipy.signal import find_peaks as scipy_find_peaks

import matplotlib.pyplot as plt

# Set global font to a clean Serif (common for journals) or Sans-Serif
plt.rcParams.update({
    
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],    
    
    "mathtext.fontset": "custom",
    "font.size": 12,                   # Legible base size
    "axes.labelweight": "bold",        # Makes ALL x/y labels bold
    "axes.titleweight": "bold",        # Makes ALL subplot titles bold
    "axes.linewidth": 1.5,             # Thicker axis lines (spines)
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.frameon": False            # Cleaner look
})



nets_arr=['bmn','bod','boe','nbo-a','sgn','srs','srs-a','srs-b','srs-c3','srs-c4','sxt','utb']

##['bmn','bod','boe','nbo-a','sgn','srs']#,'srs-a','srs-b','srs-c3','srs-c4','sxt','utb']#['bmn']#,'bod','boe','nbo-a','sgn','srs','srs-a','srs-b','srs-c3','srs-c4','sxt','utb']


n=len(nets_arr)
fig, axes = plt.subplots(n, 2, squeeze=False, figsize=(9, 4 * n), gridspec_kw={'width_ratios': [1, 1],'height_ratios': [1]*n}, layout="constrained" )##, layout="constrained")##, figsize=(12, 3 * n))
##plt.figure(1) ## scattering pattern before force relaxation 
##plt.figure(2) ## scattering pattern after force relaxation 
##plt.figure(3) ## pair correlation function - before and after force relaxation

# Set a very small horizontal padding between subplots
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.1)


flat_axes = axes.flatten()
##panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)'] # Add more as needed

import string

# Generate a list of letters: a, b, c... z, aa, ab, ac...
alphabet = list(string.ascii_lowercase)
extended_labels = alphabet + [f"{a}{b}" for a in string.ascii_lowercase for b in string.ascii_lowercase]

# Create the final format: ['a)', 'b)', ..., 'ag)']
panel_labels = [f"{l})" for l in extended_labels[:n*3]]


panel_labels=panel_labels[0:n*3]

col_titles = ["Before Force Relaxation", "After Force Relaxation", "Pair Correlation Function"]
row_headers = nets_arr ##[f"Network {k}" for k in range(1, n + 1)]

#'''
for i in range(n):
    for j in range (2):#, label in enumerate(panel_labels):
        ax = axes[i, j]
        label_idx = (i * 2) + j
        label = panel_labels[label_idx]
        #ax.set_box_aspect(1)
        # 2. Position the label: 
        # x = -0.15 (moves it left of the axis)
        # y = 1.02 (slightly above the plot, but lower than before)
        ax.text(-0.15, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='bottom', ha='right')
        #if j == 2:
          #ax.set_box_aspect(1)
                
        ##if i == 0:
            ##ax.set_title(col_titles[j], fontsize=16, fontweight='bold', pad=20)
        # --- Row Titles (Left Column Only) ---
        if j == 0:
            # We use annotate to place it far to the left of the y-axis
            ax.annotate(row_headers[i], xy=(0, 0.5), xytext=(-70, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=16, fontweight='bold', 
                        ha='right', va='center', rotation=90)
'''

for i in range(n):
    for j in range(3):
        ax = axes[i, j]
        label_idx = (i * 3) + j
        label = panel_labels[label_idx]
        
        # Ensures all plots (heatmaps and line plots) are perfectly square
        ax.set_box_aspect(1) 
        
        # 1. Position the panel labels (a, b, c...)
        # Note: Moved y to 1.10 to give more breathing room for the new top legend
        ax.text(-0.15, 1.10, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='bottom', ha='right')

        # 2. Handle Legend and Titles for Column 3 (j=2)
        if j == 2:
            # Place legend ABOVE the plot, serving as the column header
            # ncol=2 makes it horizontal; bbox_to_anchor centers it
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                      ncol=2, frameon=False, 
                      prop={'weight': 'bold', 'size': 10})
            
            # Remove title for this column to avoid overlap with the legend
            if i == 0:
                ax.set_title("", pad=35) 
        
        # 3. Standard Titles for Columns 1 and 2
        elif i == 0:
            ax.set_title(col_titles[j], fontsize=16, fontweight='bold', pad=35)

        # 4. Row Titles (Left Column Only)
        if j == 0:
            ax.annotate(row_headers[i], xy=(0, 0.5), xytext=(-70, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=16, fontweight='bold', 
                        ha='right', va='center', rotation=90)            

'''        
        

################## FUNCTIONS FOR SCATTERING PATTERN CALCULATION ###########################

sigma_filter=1.0
sigma_window=10
lim=15



def get_origin_from_file(filename, ite):
    """
    Parses a multi-frame file.
    
    It looks for lines starting with "Lattice=" and checks
    if their "Time=" value matches 'ite'. If it does,
    it returns the "Origin=" coordinates from that line.
    
    Args:
        filename (str): The path to the input file.
        ite (int): The iteration number to check against.
        
    Returns:
        list: A list of float coordinates [x, y, z] if a match is found,
              otherwise None.
    """
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # 1. Check if the line is a header line
                if line.startswith("Lattice="):
                    
                    # 2. Parse this header line into a dictionary
                    data = {}
                    try:
                        # --- THIS IS THE FIX ---
                        # Use shlex.split to correctly handle spaces inside quotes
                        parts = shlex.split(line)
                        
                        for part in parts:
                            # Split only on the first '='
                            key, value = part.split('=', 1)
                            # shlex already handles stripping quotes
                            data[key] = value
                            
                    except Exception as e:
                        # Skip this malformed header line
                        print(f"Warning: Skipping malformed header: {line.strip()}. Error: {e}")
                        continue

                    # 3. Check the "Time" condition
                    try:
                        # Handle potential trailing characters like '.' in "Time=1656."
                        time_val_str = data['Time'].rstrip('."') 
                        time_val = int(time_val_str)
                    except (KeyError, ValueError):
                        # This header didn't have a valid 'Time' key, so we skip it
                        continue
                        
                    # 4. If Time matches, get and return the "Origin"
                    if time_val == ite:
                        try:
                            origin_str = data['Origin']
                            # Convert '0.0 0.0 0.0' to a list of floats [0.0, 0.0, 0.0]
                            origin_coords = [float(x) for x in origin_str.split()]
                            return origin_coords  # Success!
                        except (KeyError, ValueError, IndexError) as e:
                            print(f"Error: Time matched, but 'Origin' was invalid. Error: {e}")
                            return None # Found the right line, but it's broken

        # If we get here, the file ended without a match
        return None

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None
     

def calculate_rigorous_2d_scattering(positions, weights, box_size, plane_axes=(0, 1), bins=512, sigma=1.0,sigma_w=6):
    """
    Calculates 2D structure factor with atomic weights and gaussian smoothing.
    
    weights: (N,) array of atomic numbers or scattering lengths for each atom
    sigma: Spread of the Gaussian in PIXELS. 
           (sigma=1.0 roughly means the atom is 1 pixel wide).
    """
    
    coords_2d = positions[:, plane_axes]
    L_2d = box_size[list(plane_axes)]

    lx = L_2d[0][1] - L_2d[0][0]
    ly = L_2d[1][1] - L_2d[1][0]
    
    # We keep 'bins' for the X-axis and scale Y-axis bins proportionally
    bins_y = int(bins * (ly / lx))
    
    # 1. Weighted Histogram
    # The 'weights' parameter adds the atomic number to the bin instead of just counting 1
    density_map, x_edges, y_edges = np.histogram2d(
        coords_2d[:, 0], 
        coords_2d[:, 1], 
        bins=bins, 
        range=[[L_2d[0][0], L_2d[0][1]], [L_2d[1][0], L_2d[1][1]]],
        weights=weights # <--- KEY ADDITION 1: Atomic Weighting
    )
    print([[L_2d[0][0], L_2d[0][1]], [L_2d[1][0], L_2d[1][1]]])
    
    # 2. Gaussian Smoothing (Real-space convolution)
    # This turns "delta functions" into "Gaussian blobs"
    if sigma > 0:
        density_map = gaussian_filter(density_map, sigma=sigma, mode='wrap') 
        # mode='wrap' is CRITICAL for periodic boundary conditions!

    # --- NEW STEP: Windowing (Fixes Box Boundary / Streaks) ---
    # Create a large window that fades the WHOLE BOX to zero at the edges
    win_x = signal.windows.gaussian(bins, std=bins/sigma_w)
    win_y = signal.windows.gaussian(bins_y, std=bins_y/sigma_w)
    window_2d = np.outer(win_x, win_y)

    density_map *= window_2d # Fades the crystal edges to black
    # ---------------------------------------------------------


    # 3. FFT
    fft_result = np.fft.fftshift(np.fft.fft2(density_map))
    intensity = np.abs(fft_result)**2
    
    # 4. Axes
    qx = np.fft.fftshift(np.fft.fftfreq(bins, d=lx/bins)) * 2 * np.pi
    qy = np.fft.fftshift(np.fft.fftfreq(bins, d=ly/bins_y)) * 2 * np.pi
    
    return qx, qy, intensity


################## NEW FUNCTIONS FOR 1D CUT AND DISORDER FITTING ###########################

def extract_1d_cut(qx, qy, intensity_2d, direction='x', avg_width=5):
    """
    Extract a 1D I(q) line cut along a specific direction through the center
    of the 2D scattering pattern (i.e., through q=0).
    
    Args:
        qx, qy: 1D arrays of q-values along each axis.
        intensity_2d: 2D array of scattering intensity.
        direction: 'x' for cut along qx (at qy~0),
                   'y' for cut along qy (at qx~0),
                   'xy' for cut along the diagonal qx=qy line (45 degrees),
                   or a float angle in degrees from the qx axis.
        avg_width: number of pixels to average perpendicular to the cut direction.
    
    Returns:
        q_1d: 1D array of q values along the cut direction.
        I_1d: 1D array of intensity values along the cut.
    """
    nx, ny = intensity_2d.shape
    half_w = avg_width // 2
    
    if direction == 'x':
        center = ny // 2
        lo = max(center - half_w, 0)
        hi = min(center + half_w + 1, ny)
        cut = np.mean(intensity_2d[:, lo:hi], axis=1)
        q_1d = qx
        
    elif direction == 'y':
        center = nx // 2
        lo = max(center - half_w, 0)
        hi = min(center + half_w + 1, nx)
        cut = np.mean(intensity_2d[lo:hi, :], axis=0)
        q_1d = qy
        
    elif direction == 'xy' or direction == 45 or direction == 45.0:
        # Diagonal cut along qx = qy (45-degree line)
        # Use the shorter dimension to determine how many samples
        n_diag = min(nx, ny)
        
        # Center indices
        cx = nx // 2
        cy = ny // 2
        
        # Sample points along the diagonal from (-n_diag/2, -n_diag/2) to (+n_diag/2, +n_diag/2)
        # relative to center
        diag_indices = np.arange(n_diag) - n_diag // 2
        
        cut = np.zeros(n_diag)
        for k, d in enumerate(diag_indices):
            ix = cx + d
            iy = cy + d
            if 0 <= ix < nx and 0 <= iy < ny:
                # Average over a perpendicular strip of width avg_width
                # Perpendicular to (1,1) direction is (-1,1) direction
                vals = []
                for w in range(-half_w, half_w + 1):
                    ix_w = ix - w  # shift along (-1,1) perpendicular
                    iy_w = iy + w
                    if 0 <= ix_w < nx and 0 <= iy_w < ny:
                        vals.append(intensity_2d[ix_w, iy_w])
                cut[k] = np.mean(vals) if len(vals) > 0 else 0.0
            else:
                cut[k] = 0.0
        
        # q along diagonal: q = sqrt(qx^2 + qy^2), with sign
        # Map pixel offsets to actual q values
        dqx = qx[1] - qx[0] if len(qx) > 1 else 1.0
        dqy = qy[1] - qy[0] if len(qy) > 1 else 1.0
        # q magnitude along diagonal = sqrt(2) * pixel_offset * dq (if dqx ~ dqy)
        q_1d = diag_indices * np.sqrt(dqx**2 + dqy**2)
    
    elif isinstance(direction, (int, float)):
        # Arbitrary angle in degrees from qx axis
        angle_rad = np.radians(float(direction))
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        # Perpendicular direction
        perp_cos = -np.sin(angle_rad)
        perp_sin = np.cos(angle_rad)
        
        cx = nx // 2
        cy = ny // 2
        
        # Number of samples along the line
        n_line = max(nx, ny)
        line_offsets = np.arange(n_line) - n_line // 2
        
        dqx = qx[1] - qx[0] if len(qx) > 1 else 1.0
        dqy = qy[1] - qy[0] if len(qy) > 1 else 1.0
        
        cut = np.zeros(n_line)
        for k, d in enumerate(line_offsets):
            vals = []
            for w in range(-half_w, half_w + 1):
                # Pixel position: center + d*(along direction) + w*(perp direction)
                ix = int(round(cx + d * cos_a + w * perp_cos))
                iy = int(round(cy + d * sin_a + w * perp_sin))
                if 0 <= ix < nx and 0 <= iy < ny:
                    vals.append(intensity_2d[ix, iy])
            cut[k] = np.mean(vals) if len(vals) > 0 else 0.0
        
        # q along this direction
        q_1d = line_offsets * np.sqrt((cos_a * dqx)**2 + (sin_a * dqy)**2)
    
    else:
        raise ValueError("direction must be 'x', 'y', 'xy', or an angle in degrees")
    
    return q_1d, cut

def radial_average_2d(qx, qy, intensity_2d, n_bins=200, q_min=None, q_max=None):
    """
    Radially average a 2D scattering pattern I(qx, qy) to get I(q) vs q,
    where q = sqrt(qx^2 + qy^2).
    
    Args:
        qx, qy: 1D arrays of q-values along each axis.
        intensity_2d: 2D array of scattering intensity, shape (len(qx), len(qy)).
        n_bins: number of radial bins.
        q_min: minimum q for binning (default: smallest nonzero q).
        q_max: maximum q for binning (default: max possible q).
    
    Returns:
        q_radial: 1D array of q bin centers.
        I_radial: 1D array of radially averaged intensity.
        I_std: 1D array of standard deviation in each bin (for error bars).
        counts: 1D array of number of pixels in each bin.
    """
    # Create 2D meshgrid of q magnitudes
    QX, QY = np.meshgrid(qx, qy, indexing='ij')
    Q_mag = np.sqrt(QX**2 + QY**2)
    
    # Flatten everything
    q_flat = Q_mag.ravel()
    I_flat = intensity_2d.ravel()
    
    # Set q range
    if q_min is None:
        # Exclude the central beam (q=0)
        dqx = np.abs(qx[1] - qx[0]) if len(qx) > 1 else 1.0
        dqy = np.abs(qy[1] - qy[0]) if len(qy) > 1 else 1.0
        q_min = max(dqx, dqy) * 0.5
    if q_max is None:
        q_max = np.min([np.abs(qx).max(), np.abs(qy).max()])
    
    # Create radial bins
    q_edges = np.linspace(q_min, q_max, n_bins + 1)
    q_radial = 0.5 * (q_edges[:-1] + q_edges[1:])
    
    # Bin the intensities
    I_radial = np.zeros(n_bins)
    I_std = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    
    # Digitize: assign each pixel to a bin
    bin_indices = np.digitize(q_flat, q_edges) - 1  # 0-indexed
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        n_in_bin = np.sum(mask)
        counts[b] = n_in_bin
        if n_in_bin > 0:
            I_radial[b] = np.mean(I_flat[mask])
            I_std[b] = np.std(I_flat[mask])
    
    return q_radial, I_radial, I_std, counts
    
def debye_waller_envelope(q, A, mean_u_sq, C):
    """
    Debye-Waller envelope for Bragg peak intensities in disorder of the first kind.
    
    Model: I_peak(q) = A * exp(-mean_u_sq * q^2) + C
    
    where mean_u_sq = <u^2> is the mean-square displacement of atoms from
    their ideal lattice sites.
    
    Args:
        q: q-values (array or scalar).
        A: amplitude (intensity at q=0).
        mean_u_sq: <u^2>, the mean-square displacement parameter.
        C: constant background / baseline.
    
    Returns:
        Model intensity values.
    """
    return A * np.exp(-mean_u_sq * q**2) + C


def fit_disorder_first_kind(q_1d, I_1d, prominence_frac=0.02, height_frac=0.05):
    """
    Identify Bragg peaks in a 1D I(q) cut and fit their intensities to the
    Debye-Waller envelope expected for disorder of the first kind.
    
    Disorder of the first kind: each atom is independently displaced from its
    ideal lattice position. Bragg peaks remain sharp (delta-function-like, 
    broadened only by finite crystal size), but their intensities decay as:
        I_peak(q) = A * exp(-<u^2> * q^2)
    
    Args:
        q_1d: 1D array of q values (use positive side only).
        I_1d: 1D array of intensity values.
        prominence_frac: peak prominence threshold as fraction of max intensity.
        height_frac: minimum peak height as fraction of max intensity.
    
    Returns:
        q_peaks: q-positions of found Bragg peaks.
        I_peaks: intensities at found Bragg peaks.
        q_fit: q-array for the fitted envelope curve.
        I_fit: fitted envelope intensity values.
        popt: fitted parameters [A, <u^2>, C].
        pcov: covariance matrix from curve_fit.
    """
    # Use only positive q
    pos_mask = q_1d > 0.1  # avoid the central beam region
    q_pos = q_1d[pos_mask]
    I_pos = I_1d[pos_mask]
    
    max_I = np.max(I_pos)
    prominence = max_I * prominence_frac
    min_height = max_I * height_frac
    
    # Find peaks
    peak_indices, properties = scipy_find_peaks(I_pos, prominence=prominence, 
                                                 height=min_height, distance=5)
    
    if len(peak_indices) < 2:
        print(f"Warning: Only {len(peak_indices)} peak(s) found. Need >= 2 for fitting.")
        q_peaks = q_pos[peak_indices] if len(peak_indices) > 0 else np.array([])
        I_peaks = properties['peak_heights'] if len(peak_indices) > 0 else np.array([])
        return q_peaks, I_peaks, None, None, None, None
    
    q_peaks = q_pos[peak_indices]
    I_peaks = properties['peak_heights']
    
    print(f"  Found {len(q_peaks)} Bragg peaks at q = {np.round(q_peaks, 3)}")
    
    # Fit Debye-Waller envelope to peak intensities
    try:
        p0 = [I_peaks[0], 0.01, 0.0]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(debye_waller_envelope, q_peaks, I_peaks,
                               p0=p0, bounds=bounds, maxfev=10000)
        
        q_fit = np.linspace(0.01, q_peaks.max() * 1.2, 300)
        I_fit = debye_waller_envelope(q_fit, *popt)
        
        perr = np.sqrt(np.diag(pcov))
        print(f"  Debye-Waller fit: A = {popt[0]:.4e} +- {perr[0]:.2e}")
        print(f"                    <u^2> = {popt[1]:.6f} +- {perr[1]:.6f} nm^2")
        print(f"                    sigma_u = {np.sqrt(popt[1]):.4f} nm (RMS displacement)")
        print(f"                    C = {popt[2]:.4e} +- {perr[2]:.2e}")
        
        return q_peaks, I_peaks, q_fit, I_fit, popt, pcov
    
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return q_peaks, I_peaks, None, None, None, None


################## END NEW FUNCTIONS ###########################

   
        
ite_arr=[0]#,10,20,30,40,50,100,500,700]#np.arange(0,1600,100)
r_avg=[]
angle_avg=[]
r_unique_all_ite=[]
angle_unique_all_ite=[]
ite=0 ## analyze only the starting iteration
plane_axes_arr=[(0,1)]#[(0,1),(1,2),(2,0)]



net_cnt=-1
net_cnt_scatter=-3

fig_1d, axes_1d = plt.subplots(len(nets_arr), 3, figsize=(18, 4*len(nets_arr)))

for net in nets_arr:
        net_cnt=net_cnt+1
        net_cnt_scatter=net_cnt_scatter+3
        folder='./'+net+'/'+hkl+'/0/Run1/'
        '''
          ############### as formed 3D net, before relax (after swelling only) ##################
        vflag = 0
        N = 12   
        print('--------------------------')   
        print('----Reading Network-------')   
        print('--------------------------')   

        #print('./cR3=3/'+net+'110/0/Run1/restart_network_'+str(ite)+'.txt')
        [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
            atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart_with_folder(folder+'network_after_swelling_only.txt',vflag,folder)##"network_after_swelling_and_relax.txt",vflag)##"restart_network_"+str(ite)+".txt",vflag)##"network_after_swelling_only.txt",  vflag)

        print('xlo, xhi',xlo, xhi) 
        print('ylo, yhi',ylo, yhi) 
        print('zlo, zhi',zlo, zhi) 
        print('n_atoms', n_atoms) 
        print('n_bonds', n_bonds) 
        print('atom_types = ', atom_types) 
        print('bond_types = ', bond_types) 
        print('mass = ', mass) 
        print('primary loops = ', len(loop_atoms)) 
        print('--------------------------')
        n_chains=n_bonds
        n_links=n_atoms

        print(np.shape(atoms))
            
        [xlo,ylo,zlo]=get_origin_from_file(folder+'full_trajectory_atoms_only_correct_box_orient.xyz', ite)

        H_data=np.loadtxt(folder+'H_matrix', skiprows=1)
        H_flat=H_data[1,1:]
        H=H_flat.reshape((3, 3))
        xhi=xlo+H[0,0]
        yhi=ylo+H[1,1]
        zhi=zlo+H[2,2]

        box_dims =np.array([[xlo,xhi],[ylo,yhi],[zlo,zhi]])#np.array([[xlo,max(atoms[:,0])],[ylo,max(atoms[:,1])],[zlo,max(atoms[:,2])]])# np.array([[xlo,xhi],[ylo,yhi],[zlo,zhi]])
        weights=np.ones(len(atoms))

        print(np.min(atoms),np.max(atoms))

        ##plane_axes_arr=[(0,1),(1,2),(2,0)]
        for plane_axes in plane_axes_arr:
            # --- RUN ANALYSIS ---
            # Calculate pattern with smoothing (sigma=2.0 pixels)
            
            qx, qy, I_smooth = calculate_rigorous_2d_scattering(atoms, weights, box_dims,plane_axes=plane_axes, sigma=sigma_filter, sigma_w=sigma_window)

            
            # --- PLOTTING ---
            ##fig, ax = plt.subplots( )## plt.subplots(1, 2, figsize=(12, 6))
            
            ax=axes[net_cnt,0] ## row n, column 1 (col_idx=0)
            ##ax.set_title('Before force relaxation')
            ax.set_aspect('auto')
            
            #ax.set_xlabel(r"$$\mathbf{q_x \, (nm^{-1})}$$", fontname='Arial')
            #ax.set_ylabel(r"$$\mathbf{g(r)}$$", fontname='Arial')



            # Plot Raw (Discrete)
            im1 = ax.imshow(np.log10(I_smooth + 1), extent=[qx.min(), qx.max(), qy.min(), qy.max()],  cmap='inferno',origin='lower',interpolation='bilinear')
            ##ax.set_title('Before relax: Gaussian Smoothed (spread and windowed)'+str(plane_axes))

            # Plot Smoothed (Gaussian)
            #im2 = ax[1].imshow(np.log10(I_smooth + 1), extent=[qx.min(), qx.max(), qy.min(), qy.max()], cmap='inferno',origin='lower',interpolation='bilinear')
            #ax[1].set_title('Gaussian Smoothed (Finite Atoms)\nNote: Clean peaks, darker background')
            

            ##ax.set_xlabel('$$q_x$$ ($$nm^{-1}$$)')
            ##ax.set_ylabel('$$q_y$$ ($$nm^{-1}$$)')
            
            ax.set_xlabel(r"$$\mathbf{q_x \, (nm^{-1})}$$")#, fontname='Arial')
            ax.set_ylabel(r"$$\mathbf{g(r)}$$")#, fontname='Arial')

            
            plt.tight_layout()
            
            ax.set_xlim([-lim, lim]) 
            ax.set_ylim([-lim, lim])
            ##plt.savefig('scattering_pattern_'+str(plane_axes)+'_before_relax')
            #plt.show()

        '''


        ############### After swelling and relax ##################
        vflag = 0
        N = 12   
        print('--------------------------')   
        print('----Reading Network-------')   
        print('--------------------------')   

        #print('./cR3=3/'+net+'110/0/Run1/restart_network_'+str(ite)+'.txt')
        [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
            atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart_with_folder(folder+'network_after_swelling_and_relax.txt',vflag,folder)##"network_after_swelling_and_relax.txt",vflag)##"restart_network_"+str(ite)+".txt",vflag)##"network_after_swelling_only.txt",  vflag)

        print('xlo, xhi',xlo, xhi) 
        print('ylo, yhi',ylo, yhi) 
        print('zlo, zhi',zlo, zhi) 
        print('n_atoms', n_atoms) 
        print('n_bonds', n_bonds) 
        print('atom_types = ', atom_types) 
        print('bond_types = ', bond_types) 
        print('mass = ', mass) 
        print('primary loops = ', len(loop_atoms)) 
        print('--------------------------')
        n_chains=n_bonds
        n_links=n_atoms

        print(np.shape(atoms))
            
        [xlo,ylo,zlo]=get_origin_from_file(folder+'full_trajectory_atoms_only_correct_box_orient.xyz', ite)

        H_data=np.loadtxt(folder+'H_matrix', skiprows=1)
        H_flat=H_data[1,1:]
        H=H_flat.reshape((3, 3))
        xhi=xlo+H[0,0]
        yhi=ylo+H[1,1]
        zhi=zlo+H[2,2]

        box_dims =np.array([[xlo,xhi],[ylo,yhi],[zlo,zhi]])#np.array([[xlo,max(atoms[:,0])],[ylo,max(atoms[:,1])],[zlo,max(atoms[:,2])]])# np.array([[xlo,xhi],[ylo,yhi],[zlo,zhi]])
        weights=np.ones(len(atoms))

        print(np.min(atoms),np.max(atoms))

        
        for plane_axes in plane_axes_arr:
            # --- RUN ANALYSIS ---
            # Calculate pattern with smoothing (sigma=2.0 pixels)
            qx, qy, I_smooth = calculate_rigorous_2d_scattering(atoms, weights, box_dims,plane_axes=plane_axes, sigma=sigma_filter, sigma_w=sigma_window)

            # Calculate pattern without smoothing (sigma=0) for comparison
            ##qx, qy, I_raw = calculate_rigorous_2d_scattering(atoms, weights, box_dims,plane_axes=plane_axes,  sigma=0,sigma_w=0)

            # --- PLOTTING ---
            ax=axes[net_cnt,0] ## row n, column 2 (col_idx=1)
            ##ax.set_title('After force relaxation')
            ax.set_aspect('auto')
            #fig, ax = plt.subplots( )## plt.subplots(1, 2, figsize=(12, 6))

            # Plot Raw (Discrete)
            im1 = ax.imshow(np.log10(I_smooth + 1), extent=[qx.min(), qx.max(), qy.min(), qy.max()],  cmap='inferno',origin='lower',interpolation='bilinear')
            #ax.set_title('After relax-Gaussian Smoothed (spread and windowed)'+str(plane_axes))
            #ax.set_title('')
            # Plot Smoothed (Gaussian)
            #im2 = ax[1].imshow(np.log10(I_smooth + 1), extent=[qx.min(), qx.max(), qy.min(), qy.max()], cmap='inferno',origin='lower',interpolation='bilinear')
            #ax[1].set_title('Gaussian Smoothed (Finite Atoms)\nNote: Clean peaks, darker background')
            '''
            for a in ax:
                a.set_xlabel('$$q_x$$ ($$A^{-1}$$)')
                a.set_ylabel('$$q_y$$ ($$A^{-1}$$)')
            '''

            ax.set_xlabel(r"$\mathbf{q_x \, (nm^{-1})}$")#, fontname='Arial')
            ax.set_ylabel(r"$\mathbf{q_y \, (nm^{-1})}$")#, fontname='Arial')
            plt.tight_layout()
            
            ax.set_xlim([-lim, lim]) 
            ax.set_ylim([-lim, lim])
            ##plt.savefig('scattering_pattern_'+str(plane_axes)+'_after_relax')
            
            #plt.show()

            # ============================================================
            # NEW: 1D cut along qx direction and Debye-Waller fit
            # ============================================================
            print("\n--- Extracting 1D I(q) cut and fitting disorder of the first kind ---")
            
            # Extract 1D cut along qx (at qy ~ 0), averaging over a few rows
            #q_cut, I_cut = extract_1d_cut(qx, qy, I_smooth, direction='xy', avg_width=5)
            # Radially averaged I(q)
            q_radial, I_radial, I_radial_std, radial_counts = radial_average_2d(
                qx, qy, I_smooth, n_bins=300, q_min=None, q_max=lim)
            
            # Use radially averaged data for plotting and fitting instead of line cut
            q_cut = q_radial
            I_cut = I_radial
            
            # Use log scale for plotting; work with the raw intensity for fitting
            ax2 = axes[net_cnt, 1]
            ax2.set_aspect('auto')
            
            # Plot the full 1D cut (positive q side only for clarity)
            pos_mask_plot = q_cut > 0.01
            ax2.semilogy(q_cut[pos_mask_plot], I_cut[pos_mask_plot], '-', 
                         color='#2b159e', linewidth=1.0, alpha=0.7, label=r'$I(q_x)$ cut')
            
            # Find peaks and fit Debye-Waller envelope
            q_peaks, I_peaks, q_fit, I_fit, popt, pcov = fit_disorder_first_kind(
                q_cut, I_cut, prominence_frac=0.001, height_frac=0.000001)
            
            # Plot the found Bragg peak positions
            if len(q_peaks) > 0:
                ax2.semilogy(q_peaks, I_peaks, 'o', color='#e63946', markersize=6,
                             zorder=5, label=f'Bragg peaks ({len(q_peaks)})')
            
            # Plot the Debye-Waller fit envelope
            if q_fit is not None and popt is not None:
                perr = np.sqrt(np.diag(pcov))
                sigma_u = np.sqrt(popt[1])
                label_fit = (r'DW fit: $\langle u^2 \rangle$=' 
                             + f'{popt[1]:.4f} nm^2\n'
                             + r'$\sigma_u$=' + f'{sigma_u:.3f} nm')
                ax2.semilogy(q_fit, I_fit, '--', color='#e63946', linewidth=2.0,
                             label=label_fit)
            
            ax2.set_xlabel(r"$\mathbf{q \, (nm^{-1})}$")
            ax2.set_ylabel(r"$\mathbf{I(q)}$")
            ax2.set_xlim([0, lim])
            ax2.legend(fontsize=8, loc='upper right')
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
            ax2.minorticks_on()
            
            # Save 1D data and fit results to text files
            np.savetxt('Iq_1d_cut_' + net + '_' + hkl + '.txt',
                       np.column_stack([q_cut, I_cut]),
                       header='q (nm^-1)    I(q)', fmt='%.6e')
            if len(q_peaks) > 0:
                np.savetxt('Bragg_peaks_' + net + '_' + hkl + '.txt',
                           np.column_stack([q_peaks, I_peaks]),
                           header='q_peak (nm^-1)    I_peak', fmt='%.6e')
            if popt is not None:
                with open('DW_fit_results_' + net + '_' + hkl + '.txt', 'w') as f_out:
                    f_out.write("# Debye-Waller fit for disorder of the first kind\n")
                    f_out.write(f"# Model: I_peak(q) = A * exp(-<u^2> * q^2) + C\n")
                    f_out.write(f"# A = {popt[0]:.6e} +/- {perr[0]:.6e}\n")
                    f_out.write(f"# <u^2> = {popt[1]:.6e} +/- {perr[1]:.6e} nm^2\n")
                    f_out.write(f"# sigma_u (RMS displacement) = {sigma_u:.6e} nm\n")
                    f_out.write(f"# C (background) = {popt[2]:.6e} +/- {perr[2]:.6e}\n")
                    f_out.write(f"# Number of Bragg peaks used = {len(q_peaks)}\n")
            
            print("--- Done with 1D cut and fitting ---\n")
            # ============================================================
            
            # ============================================================
            # NEW: Separate detailed figure for 1D I(q) and DW fit
            # ============================================================
            
            
            # ---------- Panel 1: Full 1D I(q) line cut ----------
            ##ax_a = axes_1d[0]
            #pos_mask_full = q_cut > 0.01
            #ax_a.semilogy(q_cut[pos_mask_full], I_cut[pos_mask_full], '-',
                          ##color='#2b159e', linewidth=1.0, label=r'$I(q_x)$ line cut')
                          
            # ---------- Panel 1: Radially averaged I(q) ----------
            ax_a = axes_1d[int(net_cnt_scatter/3),0]
            pos_mask_full = q_cut > 0.01
            ax_a.semilogy(q_cut[pos_mask_full], I_cut[pos_mask_full], '-',
                          color='#2b159e', linewidth=1.0, label=r'$I(q)$ radial avg')
            # Add shaded error band from radial std
            valid = pos_mask_full
            I_upper = I_radial[valid] + I_radial_std[valid]
            I_lower = I_radial[valid] - I_radial_std[valid]
            #ax_a.fill_between(q_radial[valid], I_lower, I_upper,
                              #color='#2b159e', alpha=0.15, label=r'$\pm 1\sigma$')
                              
                          
            if len(q_peaks) > 0:
                ax_a.semilogy(q_peaks, I_peaks, 'o', color='#e63946',
                              markersize=7, zorder=5, markeredgecolor='k',
                              markeredgewidth=0.5, label=f'Bragg peaks ({len(q_peaks)})')
            if q_fit is not None:
                ax_a.semilogy(q_fit, I_fit, '--', color='#e63946', linewidth=2.0,
                              label=r'DW envelope')
            ax_a.set_xlabel(r"$\mathbf{q \; (nm^{-1})}$")
            ax_a.set_ylabel(r"$\mathbf{I(q)}$")
            ax_a.set_title(str(net)+": Bragg Peak Envelope", fontweight='bold')
            ax_a.legend(fontsize=9, loc='upper right')
            ax_a.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_a.minorticks_on()
            ax_a.set_xlim([0, lim])
            
            # ---------- Panel 2: ln(I_peak) vs q^2 (linearity test) ----------
            ax_b = axes_1d[int(net_cnt_scatter/3),1]
            if len(q_peaks) >= 2:
                q_peaks_sq = q_peaks**2
                ln_I_peaks = np.log(I_peaks)
                
                ax_b.plot(q_peaks_sq, ln_I_peaks, 'o', color='#e63946',
                          markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                          label='Bragg peak intensities')
                
                # Linear fit to ln(I) vs q^2: ln(I) = ln(A) - <u^2> * q^2
                coeffs = np.polyfit(q_peaks_sq, ln_I_peaks, 1)
                q2_line = np.linspace(0, q_peaks_sq.max() * 1.2, 200)
                ln_I_line = np.polyval(coeffs, q2_line)
                
                slope = coeffs[0]
                intercept = coeffs[1]
                u_sq_from_slope = -slope
                R_squared = 1 - np.sum((ln_I_peaks - np.polyval(coeffs, q_peaks_sq))**2) / \
                            np.sum((ln_I_peaks - np.mean(ln_I_peaks))**2)
                
                ax_b.plot(q2_line, ln_I_line, '--', color='#2b159e', linewidth=2.0,
                          label=(r'Linear fit: $\langle u^2 \rangle$='
                                 + f'{u_sq_from_slope:.4f} nm^2\n'
                                 + f'$R^2$={R_squared:.4f}'))
                
                ax_b.set_xlabel(r"$\mathbf{q^2 \; (nm^{-2})}$")
                ax_b.set_ylabel(r"$\mathbf{\ln[I_{peak}(q)]}$")
                ax_b.set_title(r"Disorder 1st Kind Test: $\ln(I)$ vs $q^2$",
                               fontweight='bold')
                ax_b.legend(fontsize=9, loc='upper right')
                ax_b.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                ax_b.minorticks_on()
                
                print(f"  Linear fit (ln(I) vs q^2): slope = {slope:.6f}, "
                      f"<u^2> = {u_sq_from_slope:.6f} nm^2, R^2 = {R_squared:.4f}")
            else:
                ax_b.text(0.5, 0.5, 'Not enough peaks\nfor linearity test',
                          transform=ax_b.transAxes, ha='center', va='center',
                          fontsize=14, color='gray')
                ax_b.set_title(r"Disorder 1st Kind Test: $\ln(I)$ vs $q^2$",
                               fontweight='bold')
            
            # ---------- Panel 3: I(q) with DW fit on linear scale ----------
            ax_c = axes_1d[int(net_cnt_scatter/3),2]
            ax_c.plot(q_cut[pos_mask_full], I_cut[pos_mask_full], '-',
                      color='#2b159e', linewidth=0.8, alpha=0.5, label=r'$I(q)$')
            if len(q_peaks) > 0:
                ax_c.plot(q_peaks, I_peaks, 'o', color='#e63946',
                          markersize=7, zorder=5, markeredgecolor='k',
                          markeredgewidth=0.5, label='Bragg peaks')
            if q_fit is not None and popt is not None:
                perr_plot = np.sqrt(np.diag(pcov))
                sigma_u_plot = np.sqrt(popt[1])
                ax_c.plot(q_fit, I_fit, '--', color='#e63946', linewidth=2.5,
                          label=(r'DW: $A$=' + f'{popt[0]:.2e}\n'
                                 + r'$\langle u^2 \rangle$='
                                 + f'{popt[1]:.4f} nm^2\n'
                                 + r'$\sigma_u$=' + f'{sigma_u_plot:.3f} nm'))
                
                # Also shade the confidence band
                # Upper/lower bound from parameter uncertainties
                I_fit_upper = debye_waller_envelope(q_fit, popt[0] + perr_plot[0],
                                                     popt[1] - perr_plot[1],
                                                     popt[2])
                I_fit_lower = debye_waller_envelope(q_fit, popt[0] - perr_plot[0],
                                                     popt[1] + perr_plot[1],
                                                     popt[2])
                ax_c.fill_between(q_fit, I_fit_lower, I_fit_upper,
                                  color='#e63946', alpha=0.15,
                                  label='Fit uncertainty')
            
            ax_c.set_xlabel(r"$\mathbf{q \; (nm^{-1})}$")
            ax_c.set_ylabel(r"$\mathbf{I(q)}$")
            ax_c.set_title("Bragg Peak Envelope (Linear Scale)", fontweight='bold')
            ax_c.legend(fontsize=8, loc='upper right')
            ax_c.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_c.minorticks_on()
            ax_c.set_xlim([0, lim])
            
            fig_1d.tight_layout()
            #fig_1d.savefig('Iq_1d_DW_fit_' + net + '_' + hkl + '.png',
                           #dpi=300, bbox_inches='tight')
            #fig_1d.savefig('Iq_1d_DW_fit_' + net + '_' + hkl + '.pdf',
                           #dpi=300, bbox_inches='tight')
            #print(f"  Saved: Iq_1d_DW_fit_{net}_{hkl}.png/pdf")
            # ============================================================
            
fig_1d.savefig('Iq_1d_DW_fit_all_nets_' + hkl + '.png',
                           dpi=300, bbox_inches='tight')
fig_1d.savefig('Iq_1d_DW_fit_all_nets_' + hkl + '.pdf',
                           dpi=300, bbox_inches='tight')
print(f"  Saved: Iq_1d_DW_fit_{net}_{hkl}.png/pdf")            
            
            


if(test==True):
  fig.savefig('scattering_pattern_pair_corr_fn_publication_bmn_only.png', dpi=300, bbox_inches='tight')

  stop

################## FUNCTIONS FOR PAIR CORRELATION FUNCTION CALCULATION ###########################

random.seed(a=None, version=2)
##random.seed(a=None, version=2)
print('First random number of this seed: %d' % (random.randint(0, 10000)))
# This is just to check whether different jobs have different seeds

def get_origin_from_file(filename, ite):
    """
    Parses a multi-frame file.
    
    It looks for lines starting with "Lattice=" and checks
    if their "Time=" value matches 'ite'. If it does,
    it returns the "Origin=" coordinates from that line.
    
    Args:
        filename (str): The path to the input file.
        ite (int): The iteration number to check against.
        
    Returns:
        list: A list of float coordinates [x, y, z] if a match is found,
              otherwise None.
    """
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # 1. Check if the line is a header line
                if line.startswith("Lattice="):
                    
                    # 2. Parse this header line into a dictionary
                    data = {}
                    try:
                        # --- THIS IS THE FIX ---
                        # Use shlex.split to correctly handle spaces inside quotes
                        parts = shlex.split(line)
                        
                        for part in parts:
                            # Split only on the first '='
                            key, value = part.split('=', 1)
                            # shlex already handles stripping quotes
                            data[key] = value
                            
                    except Exception as e:
                        # Skip this malformed header line
                        print(f"Warning: Skipping malformed header: {line.strip()}. Error: {e}")
                        continue

                    # 3. Check the "Time" condition
                    try:
                        # Handle potential trailing characters like '.' in "Time=1656."
                        time_val_str = data['Time'].rstrip('."') 
                        time_val = int(time_val_str)
                    except (KeyError, ValueError):
                        # This header didn't have a valid 'Time' key, so we skip it
                        continue
                        
                    # 4. If Time matches, get and return the "Origin"
                    if time_val == ite:
                        try:
                            origin_str = data['Origin']
                            # Convert '0.0 0.0 0.0' to a list of floats [0.0, 0.0, 0.0]
                            origin_coords = [float(x) for x in origin_str.split()]
                            return origin_coords  # Success!
                        except (KeyError, ValueError, IndexError) as e:
                            print(f"Error: Time matched, but 'Origin' was invalid. Error: {e}")
                            return None # Found the right line, but it's broken

        # If we get here, the file ended without a match
        return None

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None
        

def calculate_S_q(positions, q_values,origin,H,H_inv):
    """
    Calculates the static structure factor S(q) for a set of atomic positions.

    Args:
        positions (np.ndarray): A NumPy array of atomic positions, shape (N, 3)
                                where N is the number of atoms.
        q_values (np.ndarray): A 1D NumPy array of q-magnitudes at which to
                               calculate S(q).

    Returns:
        np.ndarray: A 1D NumPy array of S(q) values corresponding to the q_values.
    """
    n_atoms = positions.shape[0]

    # Calculate all unique pairwise distances between atoms
    # pdist returns a condensed distance matrix
    #distances = pdist(positions)
    distances = get_custom_pdist(positions,origin,H,H_inv)

    s_q = np.zeros_like(q_values)

    # Iterate over each q value
    for i, q in enumerate(q_values):
        if q == 0:
            # The limit of sin(qr)/qr as q->0 is 1.
            # sum(1) for all pairs is N*(N-1)
            # This is not practical for S(q) limit, we often skip q=0
            s_q[i] = np.nan # Or some other indicator
            continue

        # Calculate sin(q*r) / (q*r) for all distances
        sinc_array = np.sin(q * distances) / (q * distances)

        # Sum over all pairs (pdist gives N*(N-1)/2 pairs, so we multiply by 2)
        sum_val = 2 * np.sum(sinc_array)

        # Apply the Debye formula
        s_q[i] = 1 + (1 / n_atoms) * sum_val

    return s_q


@jit(nopython=True, parallel=True)
def compute_histogram(positions, box_length, bin_width, max_r,origin,H,H_inv):
    n_atoms = positions.shape[0]
    n_bins = int(max_r / bin_width)
    
    # We use a matrix for thread-safety in older Numba versions, 
    # or direct atomic adds. Here we use a reduction array per thread 
    # to be safe and fast.
    hist = np.zeros(n_bins, dtype=np.float64)
    
    for i in prange(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = get_dist(i,j,positions,origin,H,H_inv)
            
            if dist < max_r:
                bin_idx = int(dist / bin_width)
                # Safety check for index
                if bin_idx < n_bins:
                    # We add 2 because each pair (i,j) counts for i->j and j->i
                    # This simplifies normalization later
                    hist[bin_idx] += 2 
                    
    return hist
    
    
def calculate_sq_via_rdf(positions, q_values, box_length,origin,H,H_inv, bin_width=0.05):
    """
    Calculates S(q) by first computing g(r) and Fourier transforming it.
    This removes the finite-size 'box scattering' effect.
    """
    n_atoms = positions.shape[0]
    volume = box_length**3
    rho = n_atoms / volume  # Number density
    
    # Max r is usually half the box length to satisfy MIC
    max_r = box_length / 2.0 
    
    # 1. Get the Raw Histogram (N(r))
    print("Computing Histogram...")
    hist_counts = compute_histogram(positions, box_length, bin_width, max_r,origin,H,H_inv)
    
    # 2. Normalize to get g(r)
    # r_mid are the midpoints of the bins
    r = np.linspace(bin_width/2, max_r - bin_width/2, len(hist_counts))
    
    # Volume of spherical shell at distance r: 4*pi*r^2*dr
    # Ideal gas count in that shell: rho * volume_of_shell
    shell_vols = 4 * np.pi * r**2 * bin_width
    ideal_counts = rho * shell_vols * n_atoms
    
    # Avoid divide by zero at very small r
    with np.errstate(divide='ignore', invalid='ignore'):
        g_r = hist_counts / ideal_counts
    g_r[np.isnan(g_r)] = 0.0
    
    # 3. Calculate S(q) using the Fourier Transform formula
    # S(q) = 1 + 4*pi*rho * Integral( r^2 * (g(r)-1) * sin(qr)/qr * dr )
    
    print("Computing Fourier Transform...")
    s_q = np.zeros_like(q_values)
    
    # Pre-calculate the term (g(r) - 1) * r^2 * dr
    # This subtraction (-1) is the MAGIC STEP that removes the box plateau
    integrand_part = (g_r - 1.0) * r**2 * bin_width
    
    for i, q in enumerate(q_values):
        if q < 1e-6:
            # Theoretical limit S(0) is related to compressibility, 
            # usually not 1, but hard to measure in simulation.
            s_q[i] = np.nan 
            continue
            
        # sin(qr)/qr term
        sinc_term = np.sin(q * r) / (q * r)
        
        # Perform the integral (summation)
        integral = np.sum(integrand_part * sinc_term)
        
        s_q[i] = 1 + 4 * np.pi * rho * integral

    return s_q, r, g_r


# --- New FFT-based function (fast for large N) ---
def calculate_S_q_fft(positions, box_size, n_bins=128):
    """
    Calculates the static structure factor S(q) using the FFT method.

    Args:
        positions (np.ndarray): Array of atomic positions, shape (N, 3).
        box_size (float): The side length of the cubic simulation box.
        n_bins (int): The number of bins per dimension for the density grid.
                      A larger number gives higher resolution in q-space.

    Returns:
        tuple: A tuple containing:
            - q_bins (np.ndarray): The centers of the q-space bins.
            - s_q_binned (np.ndarray): The azimuthally averaged S(q).
    """
    n_atoms = positions.shape[0]

    # 1. Create a 3D histogram (density grid)
    rho_grid, edges = np.histogramdd(
        positions,
        bins=n_bins,
        range=[[0, box_size], [0, box_size], [0, box_size]]
    )

    # 2. Perform the 3D Fast Fourier Transform
    rho_q = np.fft.fftn(rho_grid)

    # 3. Calculate the squared magnitude of the Fourier components
    # This is proportional to the structure factor
    sq_grid = (1.0 / n_atoms) * np.abs(rho_q) ** 2

    # 4. Azimuthal averaging
    # Create grid of q-vector magnitudes corresponding to the FFT grid
    q_x = np.fft.fftfreq(n_bins, d=box_size / n_bins) * 2 * np.pi
    q_y = np.fft.fftfreq(n_bins, d=box_size / n_bins) * 2 * np.pi
    q_z = np.fft.fftfreq(n_bins, d=box_size / n_bins) * 2 * np.pi

    qx_grid, qy_grid, qz_grid = np.meshgrid(q_x, q_y, q_z, indexing='ij')
    q_magnitude_grid = np.sqrt(qx_grid ** 2 + qy_grid ** 2 + qz_grid ** 2)

    # Bin the S(q) values based on their q-magnitude
    q_max = np.max(q_magnitude_grid)
    num_q_bins = int(n_bins / 2)

    # Use histogram to perform the averaging
    # Sum of S(q) values in each bin
    s_q_sum, q_edges = np.histogram(
        q_magnitude_grid.ravel(),
        bins=num_q_bins,
        range=(0, q_max),
        weights=sq_grid.ravel()
    )
    # Count of S(q) values in each bin
    counts, _ = np.histogram(
        q_magnitude_grid.ravel(),
        bins=num_q_bins,
        range=(0, q_max)
    )

    # Avoid division by zero for empty bins
    s_q_binned = np.divide(s_q_sum, counts, out=np.zeros_like(s_q_sum), where=counts != 0)

    q_bins = (q_edges[:-1] + q_edges[1:]) / 2

    return q_bins, s_q_binned


def random_unit_vectors(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vecs = np.zeros((n, 3))

    # Set only the first entry
    vecs[0] = [1, 0, 0]
    #vecs = np.array([,0,0])##np.random.normal(size=(n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

@njit()
def P_q(q,R):
    Pq=3*((np.sin(q*R)-q*R*np.cos(q*R))/(q*R)**3)
    return Pq


@njit(parallel=False)
def Iq_numba(dot_table,R,q, N):
    q_count = dot_table.shape[0]
    S_q = np.zeros(q_count)
    V=(4/3)*np.pi*R**3
    for iq in range(q_count):
        dots = dot_table[iq]
        if choice=='Iq_calculation':
            term=V*P_q(q[iq], R)#3 * (np.sin(q[iq] * R) - q[iq] * R * np.cos(q[iq] * R)) / (q[iq]* R)**3

        elif choice=='Sq_calculation':
            term=np.ones(len(V))
        S_q[iq] = ((np.sum(term*np.cos(dots),axis=0)**2 + np.sum(term*np.sin(dots),axis=0)**2)) / N
    
    return S_q 
    

def Iq_average_directions(positions,R, q_values, num_directions, seed=None):
    N = len(positions)
    q_directions = random_unit_vectors(num_directions, seed=seed)
    q_directions = np.repeat(q_directions, len(q_values), axis=0)
    q_repeated = np.tile(q_values, num_directions)
    qvecs = q_directions * q_repeated[:, None]

    # Precompute all dot products
    dot_products = qvecs @ positions.T  # shape (num_directions * len(q_values), N)
    
    dot_products = dot_products.reshape(num_directions, len(q_values), N)
    # Average over directions
    S_q = np.zeros(len(q_values))
    for i in range(num_directions):
        ##print(i)

        S_q += Iq_numba(dot_products[i],R,q_values, N)#*P_q(q_values,R)**2

    S_q /= num_directions

    return S_q#*P_q(q_values,13.6)**2
    

  

def get_bondlength_idx(i):
            lnk_1 = mymin.bonds[i,2]-1
            lnk_2 = mymin.bonds[i,3]-1
            delr = mymin.atoms[lnk_1,:] - mymin.atoms[lnk_2,:]
            delr_0=delr.copy()
            
            #delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
            #delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
            #delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
            
            r1 = mymin.atoms[lnk_1,:]
            r2 = mymin.atoms[lnk_2,:]


            # fractional coords
            s = H_inv @ delr

            # wrap into [-0.5, 0.5)
            s_wrapped = s - np.round(s)

            # back to Cartesian
            delr= mymin.H @ s_wrapped

            #if(np.allclose(delr,delr_0)==False):
                #print(delr_0,delr)
                #print('PBC')
            

                 
            r = LA.norm(delr)
            r_arr[i]=r
            
            origin = np.array([mymin.xlo, mymin.ylo, mymin.zlo])
            #print('origin',origin)
          

            #origin = np.array([self.xlo, self.ylo, self.zlo])
            #frac = (self.atoms - origin) @ H_inv.T
          
            #atoms=self.atoms
            atoms_relative = r1 - origin
            s1 = (H_inv @ atoms_relative.T).T
            
            atoms_relative = r2 - origin
            s2 = (H_inv @ atoms_relative.T).T
            #s_wrapped=s1%1.0-s2%1.0
            
            #'''
            
            print('lnk_1',lnk_1,'lnk_2',lnk_2, 'delr_0',delr_0,'delr',delr, 's',s,'s_wrapped', s_wrapped,'r',r)
            print('mymin.atoms[lnk_1,:]',mymin.atoms[lnk_1,:],'mymin.atoms[lnk_2,:]',mymin.atoms[lnk_2,:])
            print( s1,s2,s1%1.0,  s2%1.0)
            print((mymin.H @ (s1%1.0).T).T+origin,(mymin.H @ (s2%1.0).T).T+origin)
            #'''
            xy_plane_normal = (0, 0, 1)
            v=delr
            angle = angle_between_vector_and_plane(v, xy_plane_normal)
            return delr,r, angle

@jit(nopython=True)            
def get_dist(lnk_1,lnk_2,positions,origin,H,H_inv):
            #lnk_1 = mymin.bonds[i,2]-1
            #lnk_2 = mymin.bonds[i,3]-1
            delr =positions[lnk_1,:] - positions[lnk_2,:]###mymin.atoms[lnk_1,:] - mymin.atoms[lnk_2,:]
            delr_0=delr.copy()
            
            
            #delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
            #delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
            #delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
            
            r1 = positions[lnk_1,:]
            r2 = positions[lnk_2,:]


            # fractional coords
            s = H_inv @ delr

            # wrap into [-0.5, 0.5)
            s_wrapped = s - np.round(s)

            # back to Cartesian
            delr= H @ s_wrapped

            #if(np.allclose(delr,delr_0)==False):
                #print(delr_0,delr)
                #print('PBC')
            

                 
            r = LA.norm(delr)
            ##r_arr[i]=r
            
            #origin = np.array([mymin.xlo, mymin.ylo, mymin.zlo])
            #print('origin',origin)
          

            #origin = np.array([self.xlo, self.ylo, self.zlo])
            #frac = (self.atoms - origin) @ H_inv.T
          
            #atoms=self.atoms
            atoms_relative = r1 - origin
            s1 = (H_inv @ atoms_relative.T).T
            
            atoms_relative = r2 - origin
            s2 = (H_inv @ atoms_relative.T).T
            #s_wrapped=s1%1.0-s2%1.0
            
            #'''
            
            #print('lnk_1',lnk_1,'lnk_2',lnk_2, 'delr_0',delr_0,'delr',delr, 's',s,'s_wrapped', s_wrapped,'r',r)
            #print('mymin.atoms[lnk_1,:]',mymin.atoms[lnk_1,:],'mymin.atoms[lnk_2,:]',mymin.atoms[lnk_2,:])
            #print( s1,s2,s1%1.0,  s2%1.0)
            #print((mymin.H @ (s1%1.0).T).T+origin,(mymin.H @ (s2%1.0).T).T+origin)
            #'''
            #xy_plane_normal = (0, 0, 1)
            #v=delr
            #angle = angle_between_vector_and_plane(v, xy_plane_normal)
            return r #delr,r, angle

@jit(nopython=True)
def get_custom_pdist(positions,origin,H,H_inv,r_max, n_bins):
    #n = mymin.atoms.shape[0]
    # Calculate number of pairs: N*(N-1)/2
    n_atoms=np.shape(positions)[0]
    num_pairs = n_atoms * (n_atoms - 1) // 2
    distances = np.zeros(num_pairs)
    
    n_particles = positions.shape[0]
    histogram = np.zeros(n_bins, dtype=np.int64)
    dr = r_max / n_bins
    
    idx = 0
    # Manual loop to mimic pdist/combinations
    for i in range(0,n_atoms):
        for j in range(i + 1, n_atoms):
            dist = get_dist(i, j,positions,origin,H,H_inv)
            distances[idx] = dist
            idx += 1
            #print(dist)
            
            if dist < r_max:
                ##dist = np.sqrt(dist_sq)
                bin_index = int(dist / dr)
                if bin_index < n_bins:
                    histogram[bin_index] += 2
                    
    return histogram, distances
    
## read in orthogonal axis alligned simulation box (lattice)
def compute_static_structure_factor(coords, q_vecs):
    n_q = q_vecs.shape[0]
    n_atoms = coords.shape[0]
    S_q = np.zeros(n_q)
    
    # Parallel loop over q-vectors
    for i in prange(n_q):
        qx = q_vecs[i, 0]
        qy = q_vecs[i, 1]
        qz = q_vecs[i, 2]
        
        sum_cos = 0.0
        sum_sin = 0.0
        
        # Inner loop over atoms
        for j in range(n_atoms):
            # Dot product q*r
            phase = qx*coords[j, 0] + qy*coords[j, 1] + qz*coords[j, 2]
            sum_cos += np.cos(phase)
            sum_sin += np.sin(phase)
            
        # S(q) = |Sum|^2 / N
        S_q[i] = (sum_cos**2 + sum_sin**2) / n_atoms
        
    return S_q
    
    

def generate_Sq(ite,folder):
    netgen_flag = 0
    if (netgen_flag == 0):

        vflag = 0
        N = 12
        print('--------------------------')
        print('----Reading Network-------')
        print('--------------------------')
        [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
         atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart_with_folder(folder+"restart_network_"+str(ite)+".txt", vflag,folder)

        print('xlo, xhi', xlo, xhi)
        print('ylo, yhi', ylo, yhi)
        print('zlo, zhi', zlo, zhi)
        print('n_atoms', n_atoms)
        print('n_bonds', n_bonds)
        print('atom_types = ', atom_types)
        print('bond_types = ', bond_types)
        print('mass = ', mass)
        print('primary loops = ', len(loop_atoms))
        print('--------------------------')
        n_chains = n_bonds
        n_links = n_atoms

    elif (netgen_flag == 1):

        func = p.func
        N = p.N
        b = p.b
        rho = p.rho
        l0 = 1
        prob = 1.0
        n_chains = p.n_chains
        n_links = int(2 * n_chains / func)
        ##   L = 32.5984

        cR3 = p.cR3
        conc = cR3 / (N * b ** 2) ** 1.5  # (chains/nm3)
        ##n_chains=len(G.edges)
        L = (tot_chains / conc) ** (1 / 3)
        netgen.generate_network(prob, func, N, L, l0, n_chains, n_links)

        [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
         atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS("network.txt", N, 0)


    else:
        print('Invalid network generation flag')

    hkl = p.hkl

    #fstr = open('stress', 'w')
    #fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5]\n')

    #flen = open('strand_lengths', 'w')
    #flen.write('#lambda, ave(R), max(R)\n')

    #fkmc = open('KMC_stats', 'w')
    #fkmc.write('#lambda, init bonds, final bonds\n')
    # -------------------------------------#
    #       Simulation Parameters         #
    # -------------------------------------#

    # N  = 12
    Nb = N
    b = p.b
    K = p.K
    r0 = 0.0
    U0 = p.U0
    tau = p.tau
    del_t = p.del_t
    erate = p.erate
    lam_max = p.lam_max
    tol = p.tol
    max_itr = p.max_itr
    write_itr = p.write_itr
    wrt_step = p.wrt_step
    angle_deg = p.angle_deg
    
    [xlo,ylo,zlo]=get_origin_from_file(folder+"full_trajectory_atoms_only_correct_box_orient.xyz", ite)
    print('xlo,ylo,zlo',[xlo,ylo,zlo])
    mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, N, 'Mao')
    print(mymin.xlo,mymin.ylo,mymin.zlo)
    hkl=p.hkl

    H_data=np.loadtxt(folder+"H_matrix", skiprows=1)
    H_flat=H_data[ite+2,1:]
    H=H_flat.reshape((3, 3))
    print(H)
    mymin.H=H
    H_inv=np.linalg.inv(H)
    
    volume = np.abs(np.linalg.det(mymin.H))
            #print('VOLUME',volume)
    inv_volume = 1.0 / volume
    
    
    
    #mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, N, 'Mao')


    ##ioLAMMPS.writeLAMMPS('network_before_swelling.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo,
    ##mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)


    # --- Example Usage ---

    # 1. Generate some random atomic positions (simulating a liquid or gas)
    num_atoms = 10000
    box_size = volume**(1/3)
    # Positions are random within a cube of side length `box_size`
    #random_positions = np.random.rand(num_atoms, 3) * box_size
    positions=atoms##np.random.rand(num_atoms, 3) * box_size#np.random.uniform(-25,25, size=(len(atoms),3))#atoms+np.random.uniform(-15,15) #mymin.atoms
    
    
    
    box_min = np.array([xlo,ylo,zlo]) 
    box_max = np.array([xlo+box_size,ylo+box_size,zlo+box_size])
    ##positions=np.random.uniform(low=box_min, high=box_max, size=(n_atoms, 3))
  
    particle_sizes=np.ones(len(positions))
    N=len(atoms)
    
    # 2. Define the range of q values to probe
    ##q_space = np.logspace(-3, 1, 200)
    print('starting S(q) calculation')
    
    start_t=time.time()
    
    nm=1#e-9
    seeds=[1]#np.arange(50,100,10)
    n_directions=1
    
    #q_vals = np.logspace(np.log(2*np.pi/(xlo+volume**(1/3))),np.log(2*np.pi/(xlo+0.0001)),500)#(np.log(1/(xlo+volume**(1/3))), np.log(1/(xlo+0.0001)), 500)
    #print(np.log(2*np.pi/(xlo+volume**(1/3))),np.log(2*np.pi/(xlo+0.0001)))
    
    
    q_vals =np.logspace(0,1.5,500)## np.logspace(np.log(2*np.pi/(volume**(1/3))),3,500)#np.logspace(-1,0.5,500)##np.logspace(-np.log(np.pi/(volume**(1/3))),np.log(np.pi/(volume**(1/3))),500)
    
    L = volume**(1/3)  # Your box length
    
    
    
    end_t=time.time()
    print(f"time taken for ite={ite}:",end_t-start_t)
    
    import matplotlib.pyplot as plt
    plt.loglog(q_vals, S_q,'.-',label=str(ite))
    plt.xlabel("q")
    if choice=='Iq_calculation':
        plt.ylabel("I(q)")
    elif choice=='Sq_calculation':
        plt.ylabel("S(q)")
    #plt.title("SAS")
    plt.legend()
    plt.grid(True)
    #plt.show()
    
    
    
    

def generate_gr(ite,folder, net_cnt):
    print('ite=',ite)
    netgen_flag = 0
    if (netgen_flag == 0):

        vflag = 0
        N = 12
        print('--------------------------')
        print('----Reading Network-------')
        print('--------------------------')
        if(ite==-1):
          [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
         atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart_with_folder(folder+"network_after_swelling_only.txt",vflag,folder)#"+str(ite)+".txt", vflag)
        else:
          [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
         atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS_restart_with_folder(folder+"restart_network_"+str(ite)+".txt", vflag,folder)

        print('xlo, xhi', xlo, xhi)
        print('ylo, yhi', ylo, yhi)
        print('zlo, zhi', zlo, zhi)
        print('n_atoms', n_atoms)
        print('n_bonds', n_bonds)
        print('atom_types = ', atom_types)
        print('bond_types = ', bond_types)
        print('mass = ', mass)
        print('primary loops = ', len(loop_atoms))
        print('--------------------------')
        n_chains = n_bonds
        n_links = n_atoms

    elif (netgen_flag == 1):

        func = p.func
        N = p.N
        b = p.b
        rho = p.rho
        l0 = 1
        prob = 1.0
        n_chains = p.n_chains
        n_links = int(2 * n_chains / func)
        ##   L = 32.5984

        cR3 = p.cR3
        conc = cR3 / (N * b ** 2) ** 1.5  # (chains/nm3)
        ##n_chains=len(G.edges)
        L = (tot_chains / conc) ** (1 / 3)
        netgen.generate_network(prob, func, N, L, l0, n_chains, n_links)

        [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds,
         atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS("network.txt", N, 0)


    else:
        print('Invalid network generation flag')

    hkl = p.hkl

    # -------------------------------------#
    #       Simulation Parameters         #
    # -------------------------------------#

    # N  = 12
    Nb = N
    b = p.b
    K = p.K
    r0 = 0.0
    U0 = p.U0
    tau = p.tau
    del_t = p.del_t
    erate = p.erate
    lam_max = p.lam_max
    tol = p.tol
    max_itr = p.max_itr
    write_itr = p.write_itr
    wrt_step = p.wrt_step
    angle_deg = p.angle_deg
    
    [xlo,ylo,zlo]=get_origin_from_file(folder+"full_trajectory_atoms_only_correct_box_orient.xyz", ite)
    print('xlo,ylo,zlo',[xlo,ylo,zlo])
    min_val=np.genfromtxt(folder+'min_max_val_N.txt')[0]
    mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, min_val,p.bin_width, 'Mao')
    print(mymin.xlo,mymin.ylo,mymin.zlo)
    hkl=p.hkl

    H_data=np.loadtxt(folder+"H_matrix", skiprows=1)
    H_flat=H_data[ite+2,1:]
    H=H_flat.reshape((3, 3))
    print(H)
    mymin.H=H
    H_inv=np.linalg.inv(H)
    
    
    
    volume = np.abs(np.linalg.det(mymin.H))
            #print('VOLUME',volume)
    inv_volume = 1.0 / volume
    L = volume**(1/3)  # Your box length
    
    r_max=4 ## L/2
    n_bins=500
    
    '''
    
    print('starting g(r) calculation')
    
    start_t=time.time()
    
    
    histogram, distances=get_custom_pdist(mymin.atoms,np.array([xlo,ylo,zlo]),H,H_inv, r_max,n_bins)
    #print([i for i in distances])
    
    
    r_edges = np.linspace(0, r_max, n_bins + 1)
    # Volume of shell = (4/3) * pi * (r_outer^3 - r_inner^3)
    volumes = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    
    density = n_atoms / volume
    
    norm_factor = density * volumes * n_atoms
    
    g_r = histogram / norm_factor
    
    # Calculate bin centers for plotting
    r = (r_edges[:-1] + r_edges[1:]) / 2
    
    
    ##plt.figure(figsize=(8, 5))
    
    ##plt.figure()
    # Main Plot
    ax=axes[net_cnt,1] ## row number n, column number 3 (index=2)
    ax.set_aspect('auto')
    if(ite==0):
      ax.plot(r[1:], g_r[1:], linewidth=2, label='After force relaxation',color='#8c1953')#'ite='+str(ite))
    if(ite==-1):
      ax.plot(r[1:], g_r[1:], linewidth=2, label='Before force relaxation',color='#2b159e')#
    #plt.xscale('log')
    
    # Reference line at g(r) = 1 (Ideal Gas / No Correlation)
    #ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)##, label='Ideal Gas Limit')
    
    # Labels and Styling
    #ax.set_xlabel(r' $r$ (nm)', fontsize=12)
    #ax.set_ylabel(r'$g(r)$', fontsize=12)
    
    ax.set_xlabel(r"$\mathbf{r \, (nm)}$")#, fontname='Arial')
    ax.set_ylabel(r"$\mathbf{g(r)}$")#, fontname='Arial')
    #plt.title(title, fontsize=14)
    #plt.legend(fontsize=10)
    
    
    
    # Minor grid helps read peak positions
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.minorticks_on()
    '''
    
    ##plt.tight_layout()
    
    ##plt.savefig('gr_ite='+str(ite))
    
    ##plt.figure(1)
    # Main Plot
    '''
    plt.plot(r[1:], g_r[1:], linewidth=2, label='ite='+str(ite))
    #plt.xscale('log')
    
    # Reference line at g(r) = 1 (Ideal Gas / No Correlation)
    ##plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal Gas Limit')
    
    # Labels and Styling
    plt.xlabel(r'Distance $r$ ( simulation units)', fontsize=12)
    plt.ylabel(r'$g(r)$', fontsize=12)
    #plt.title(title, fontsize=14)
    '''
    ##if(net_cnt==0):
      ##ax.legend(fontsize=10,loc='best')
    ax.set_aspect('auto')
    
    
    
    # Minor grid helps read peak positions
    ##plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ##plt.minorticks_on()
    
    ##plt.tight_layout()
    
    
    ##np.savetxt('gr_ite='+str(ite)+'.txt', np.transpose(np.array([r, g_r])))
    
    
    

   
    
   
    
    
    
    end_t=time.time()
    #print(f"time taken for ite={ite}:",end_t-start_t)
    
    
    


################################# END of ALL FUNCTION ##################################

ite_arr =[0]#[-1,0]#[0,400,500]#,400,500]#,100,500]#,100,200,300,500,600,700]#np.arange(0, ite_break, ite_step)
    # Record the start time
start_time = time.time()
##num_cores = 2#multiprocessing.cpu_count()
##choice='Sq_calculation'
outpath=''
##plt.figure(1)

net_cnt=-1
 
for net in nets_arr:
  net_cnt=net_cnt+1    
  folder='./'+net+'/'+hkl+'/0/Run1/'
  for ite in ite_arr:
    generate_gr(ite,folder,net_cnt)
  
fig.tight_layout()

fig.savefig('scattering_pattern_pair_corr_fn_publication_bmn_only.png', dpi=300, bbox_inches='tight')
fig.savefig('scattering_pattern_pair_corr_fn_publication_bmn_only.pdf', dpi=300, bbox_inches='tight')

plt.show()