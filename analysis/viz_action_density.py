#!/usr/bin/env python3
"""
Visualization: QCD Vacuum Structure via Action Density.

Creates CSSM Adelaide-style 3D isosurface renderings of the gauge field
action density. The lattice data is interpolated onto a fine grid and
smooth isosurfaces are extracted using marching cubes, giving the
characteristic organic, blobby shapes of instantons and topological objects.

With cooling (APE smearing), UV fluctuations are stripped away to reveal
the underlying instanton structure.

Reads binary action density files produced by the C++ code:
  Format: 4 ints (nX, nY, nZ, nT) + latticeVolume doubles
  Site ordering: idx = x + nX*(y + nY*(z + nZ*t))

Output:
  - Static 3D isosurface renderings (single time slice, multiple thresholds)
  - Cooling evolution strip (uncooled -> cooled)
  - Cooling statistics plots
  - Animated cooling GIF
"""

import os
import sys
import struct
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from scipy.ndimage import gaussian_filter, zoom
from skimage import measure


def read_action_density(filename):
    """Read binary action density file.

    Returns (nX, nY, nZ, nT, data) where data is shape (nX, nY, nZ, nT).
    """
    with open(filename, 'rb') as f:
        dims = struct.unpack('4i', f.read(16))
        nX, nY, nZ, nT = dims
        vol = nX * nY * nZ * nT
        data = np.array(struct.unpack(f'{vol}d', f.read(vol * 8)))

    # Reshape: idx = x + nX*(y + nY*(z + nZ*t))
    field = data.reshape((nT, nZ, nY, nX))
    field = field.transpose((3, 2, 1, 0))  # -> (nX, nY, nZ, nT)
    return nX, nY, nZ, nT, field


def upsample_and_smooth(field_3d, upsample_factor=4, sigma=1.2):
    """Interpolate lattice data onto a finer grid and smooth.

    This converts the discrete lattice data into a smooth continuous
    field suitable for isosurface extraction.
    """
    # Upsample using spline interpolation
    fine = zoom(field_3d, upsample_factor, order=3)
    # Gaussian smooth for organic shapes
    fine = gaussian_filter(fine, sigma=sigma)
    return fine


def compute_isosurface(field_3d, threshold, upsample_factor=4):
    """Extract an isosurface from a 3D field using marching cubes.

    Returns (verts, faces) where verts are in original lattice coordinates.
    """
    # Upsample and smooth
    fine = upsample_and_smooth(field_3d, upsample_factor=upsample_factor)

    if fine.max() < threshold:
        return None, None

    try:
        verts, faces, _, _ = measure.marching_cubes(fine, level=threshold)
    except ValueError:
        return None, None

    # Scale vertices back to original lattice coordinates
    nX, nY, nZ = field_3d.shape
    verts[:, 0] *= (nX - 1) / (fine.shape[0] - 1)
    verts[:, 1] *= (nY - 1) / (fine.shape[1] - 1)
    verts[:, 2] *= (nZ - 1) / (fine.shape[2] - 1)

    return verts, faces


def render_isosurface(ax, field_3d, thresholds, colors, alphas,
                       upsample_factor=4, edge_alpha=0.05):
    """Render multiple nested isosurfaces on a 3D axis.

    thresholds: list of isosurface levels (outer to inner)
    colors: list of colors for each level
    alphas: list of alpha transparencies for each level
    """
    nX, nY, nZ = field_3d.shape

    for threshold, color, alpha in zip(thresholds, colors, alphas):
        verts, faces = compute_isosurface(field_3d, threshold, upsample_factor)
        if verts is None:
            continue

        # Build triangle list for Poly3DCollection
        tri_verts = verts[faces]

        mesh = Poly3DCollection(tri_verts, alpha=alpha, linewidths=0.1)
        mesh.set_facecolor(color)
        mesh.set_edgecolor((*mcolors.to_rgb(color), edge_alpha))
        ax.add_collection3d(mesh)

    ax.set_xlim(0, nX - 1)
    ax.set_ylim(0, nY - 1)
    ax.set_zlim(0, nZ - 1)


def style_3d_axis(ax, nX, nY, nZ, title=''):
    """Apply consistent dark styling to a 3D axis."""
    ax.set_xlabel('x', color='white', fontsize=9)
    ax.set_ylabel('y', color='white', fontsize=9)
    ax.set_zlabel('z', color='white', fontsize=9)
    ax.set_title(title, fontsize=11, color='white', pad=10)
    ax.tick_params(colors='white', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')
    ax.grid(True, alpha=0.1, color='gray')
    ax.set_xlim(0, nX - 1)
    ax.set_ylim(0, nY - 1)
    ax.set_zlim(0, nZ - 1)


def get_adaptive_thresholds(field_3d, n_levels=3):
    """Compute isosurface thresholds adapted to the field's value range.

    For uncooled fields (high action), uses higher thresholds.
    For cooled fields (low action), uses lower thresholds to reveal structure.
    """
    vmax = field_3d.max()
    vmean = field_3d.mean()

    if vmean < 0.01:
        # Very cooled: almost no action anywhere
        return [], [], []

    if vmean > 1.0:
        # Uncooled / weakly cooled: lots of UV fluctuations
        p75 = np.percentile(field_3d, 75)
        p90 = np.percentile(field_3d, 90)
        p97 = np.percentile(field_3d, 97)
        thresholds = [p75, p90, p97]
        colors = ['#FF6600', '#FF2200', '#FFEE00']
        alphas = [0.12, 0.25, 0.6]
    else:
        # Partially cooled: instanton-like lumps
        # Use fractions of max to pick out the bumps
        thresholds = [0.15 * vmax, 0.35 * vmax, 0.6 * vmax]
        colors = ['#4488FF', '#FF6600', '#FFEE00']
        alphas = [0.12, 0.3, 0.6]

    # Filter out thresholds that are too small to produce surfaces
    valid = [(t, c, a) for t, c, a in zip(thresholds, colors, alphas)
             if t > 1e-6 and field_3d.max() > t]
    if not valid:
        return [], [], []
    thresholds, colors, alphas = zip(*valid)
    return list(thresholds), list(colors), list(alphas)


def render_time_slices(field_4d, config_num, cool_step, output_dir, run_dir):
    """Render a 2x2 grid of time slices with isosurfaces."""
    nX, nY, nZ, nT = field_4d.shape
    t_slices = np.linspace(0, nT - 1, 4, dtype=int)

    fig = plt.figure(figsize=(16, 14), facecolor='#080818')

    cool_label = f"Cool step {cool_step}" if cool_step >= 0 else "Uncooled"
    fig.suptitle(f'QCD Vacuum: Action Density Isosurfaces  (Config {config_num}, {cool_label})',
                 fontsize=16, color='white', fontweight='bold', y=0.96)

    for i, t in enumerate(t_slices):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d', facecolor='#080818')
        slice_3d = field_4d[:, :, :, t]

        thresholds, colors, alphas = get_adaptive_thresholds(slice_3d)
        if thresholds:
            render_isosurface(ax, slice_3d, thresholds, colors, alphas)

        style_3d_axis(ax, nX, nY, nZ,
                      f't = {t}  ($S_{{avg}}$ = {slice_3d.mean():.3f})')
        ax.view_init(elev=25, azim=45 + i * 15)

    if cool_step >= 0:
        outfile = os.path.join(run_dir,
                               f'viz_action_density_cfg{config_num:04d}_cool{cool_step:03d}.png')
    else:
        outfile = os.path.join(run_dir,
                               f'viz_action_density_cfg{config_num:04d}.png')

    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outfile}")


def render_cooling_comparison(field_list, cool_steps, config_num, run_dir):
    """Render a comparison strip showing cooling evolution with isosurfaces."""
    n_panels = min(len(field_list), 6)
    if n_panels < 2:
        return

    indices = np.linspace(0, len(field_list) - 1, n_panels, dtype=int)

    fig = plt.figure(figsize=(5 * n_panels, 10), facecolor='#080818')
    fig.suptitle(f'QCD Vacuum Cooling: UV Fluctuations Stripped Away  (Config {config_num})',
                 fontsize=16, color='white', fontweight='bold', y=0.97)

    nT = field_list[0].shape[3]
    nX, nY, nZ = field_list[0].shape[:3]
    t_mid = nT // 2

    for panel_idx, data_idx in enumerate(indices):
        ax = fig.add_subplot(1, n_panels, panel_idx + 1, projection='3d',
                             facecolor='#080818')
        field = field_list[data_idx]
        step = cool_steps[data_idx]
        slice_3d = field[:, :, :, t_mid]

        thresholds, colors, alphas = get_adaptive_thresholds(slice_3d)
        if thresholds:
            render_isosurface(ax, slice_3d, thresholds, colors, alphas)

        avg_action = slice_3d.mean()
        style_3d_axis(ax, nX, nY, nZ,
                      f'Cool step {step}\n$\\langle S \\rangle$ = {avg_action:.3f}')
        ax.view_init(elev=25, azim=45)

    outfile = os.path.join(run_dir, f'viz_cooling_evolution_cfg{config_num:04d}.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outfile}")


def render_cooling_animation(field_list, cool_steps, config_num, run_dir):
    """Create an animated GIF of the cooling process with isosurfaces."""
    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available, skipping GIF animation")
        return

    nX, nY, nZ, nT = field_list[0].shape
    t_mid = nT // 2

    frames = []
    tmpdir = os.path.join(run_dir, '_cooling_frames')
    os.makedirs(tmpdir, exist_ok=True)

    # Subsample cooling steps for the animation (every 2nd step)
    step_indices = list(range(0, len(field_list), 2))
    if (len(field_list) - 1) not in step_indices:
        step_indices.append(len(field_list) - 1)

    print(f"  Generating {len(step_indices)} animation frames...")

    for frame_idx, data_idx in enumerate(step_indices):
        field = field_list[data_idx]
        step = cool_steps[data_idx]
        slice_3d = field[:, :, :, t_mid]
        avg_action = slice_3d.mean()

        fig = plt.figure(figsize=(8, 7), facecolor='#080818')
        ax = fig.add_subplot(111, projection='3d', facecolor='#080818')

        thresholds, colors, alphas = get_adaptive_thresholds(slice_3d)
        if thresholds:
            render_isosurface(ax, slice_3d, thresholds, colors, alphas)

        style_3d_axis(ax, nX, nY, nZ,
                      f'QCD Vacuum (Config {config_num}, t={t_mid})\n'
                      f'Cooling Step {step}  |  '
                      f'$\\langle S \\rangle$ = {avg_action:.3f}')
        ax.view_init(elev=25, azim=45 + step * 2)

        frame_path = os.path.join(tmpdir, f'frame_{frame_idx:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        frames.append(Image.open(frame_path))

    gif_path = os.path.join(run_dir, f'viz_cooling_cfg{config_num:04d}.gif')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0
    )
    print(f"  Saved GIF: {gif_path}")

    for f_name in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f_name))
    os.rmdir(tmpdir)


def render_summary_stats(field_list, cool_steps, config_num, run_dir):
    """Plot cooling statistics: average action and action variance vs cooling step."""
    fig = plt.figure(figsize=(14, 6), facecolor='#080818')

    nT = field_list[0].shape[3]
    t_mid = nT // 2

    avg_actions = []
    max_actions = []
    std_actions = []

    for field in field_list:
        s = field[:, :, :, t_mid]
        avg_actions.append(s.mean())
        max_actions.append(s.max())
        std_actions.append(s.std())

    ax1 = fig.add_subplot(1, 2, 1, facecolor='#080818')
    ax1.plot(cool_steps, avg_actions, 'o-', color='#FF4444', linewidth=2, markersize=4)
    ax1.set_xlabel('Cooling Step', fontsize=12, color='white')
    ax1.set_ylabel('$\\langle S(x) \\rangle$', fontsize=12, color='white')
    ax1.set_title('Average Action Density vs Cooling', fontsize=13, color='white')
    ax1.grid(True, alpha=0.15, color='gray')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    ax2 = fig.add_subplot(1, 2, 2, facecolor='#080818')
    ax2.plot(cool_steps, max_actions, 's-', color='#FF8844', linewidth=2,
             markersize=4, label='Max $S(x)$')
    ax2.plot(cool_steps, std_actions, 'o-', color='#4488FF', linewidth=2,
             markersize=4, label='$\\sigma_S$')
    ax2.set_xlabel('Cooling Step', fontsize=12, color='white')
    ax2.set_ylabel('Action Density', fontsize=12, color='white')
    ax2.set_title('Action Density Peak & Fluctuations', fontsize=13, color='white')
    ax2.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray', labelcolor='white')
    ax2.grid(True, alpha=0.15, color='gray')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    fig.suptitle(f'Cooling Statistics (Config {config_num}, t={t_mid})',
                 fontsize=15, color='white', fontweight='bold', y=1.02)

    outfile = os.path.join(run_dir, f'viz_cooling_stats_cfg{config_num:04d}.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {outfile}")


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    search_dirs = []
    for d in os.listdir(run_dir):
        full_path = os.path.join(run_dir, d)
        if os.path.isdir(full_path) and d.startswith('SU3_'):
            search_dirs.append(full_path)
    search_dirs.append(run_dir)

    found_data = False

    for search_dir in search_dirs:
        files = sorted(glob.glob(os.path.join(search_dir, 'action_density_cfg*.bin')))
        if not files:
            continue

        found_data = True
        dirname = os.path.basename(search_dir) if search_dir != run_dir else 'run'
        print(f"\nProcessing action density data in {dirname}/")
        print(f"  Found {len(files)} binary files")

        config_data = {}
        for f in files:
            basename = os.path.basename(f)
            parts = basename.replace('.bin', '').split('_')
            cfg_str = [p for p in parts if p.startswith('cfg')][0]
            config_num = int(cfg_str[3:])
            cool_parts = [p for p in parts if p.startswith('cool')]
            cool_step = int(cool_parts[0][4:]) if cool_parts else -1
            if config_num not in config_data:
                config_data[config_num] = []
            config_data[config_num].append((cool_step, f))

        for config_num in sorted(config_data.keys()):
            entries = sorted(config_data[config_num])
            print(f"\n  Config {config_num}: {len(entries)} cooling steps")

            if len(entries) == 1:
                cool_step, filepath = entries[0]
                nX, nY, nZ, nT, field = read_action_density(filepath)
                print(f"    Lattice: {nX}x{nY}x{nZ}x{nT}")
                print(f"    <S> = {field.mean():.4f}, max = {field.max():.4f}")
                render_time_slices(field, config_num, cool_step, search_dir, run_dir)
            else:
                field_list = []
                cool_steps_list = []
                for cool_step, filepath in entries:
                    nX, nY, nZ, nT, field = read_action_density(filepath)
                    field_list.append(field)
                    cool_steps_list.append(cool_step)
                    if cool_step == 0 or cool_step == entries[-1][0]:
                        print(f"    Step {cool_step:3d}: <S> = {field.mean():.4f}, "
                              f"max = {field.max():.4f}")

                print(f"    Lattice: {nX}x{nY}x{nZ}x{nT}")

                # Render uncooled time slices
                render_time_slices(field_list[0], config_num, 0,
                                   search_dir, run_dir)

                # Render partially cooled (step 5) - shows instanton emergence
                if len(field_list) > 5:
                    render_time_slices(field_list[5], config_num, cool_steps_list[5],
                                       search_dir, run_dir)

                # Render fully cooled time slices
                render_time_slices(field_list[-1], config_num, cool_steps_list[-1],
                                   search_dir, run_dir)

                # Cooling comparison strip
                render_cooling_comparison(field_list, cool_steps_list,
                                          config_num, run_dir)

                # Cooling statistics
                render_summary_stats(field_list, cool_steps_list,
                                     config_num, run_dir)

                # Cooling animation GIF
                render_cooling_animation(field_list, cool_steps_list,
                                         config_num, run_dir)

    if not found_data:
        print("No action density data found!")
        print("Run the simulation with 'Measure Action Density: on' in the input file.")
        print("  Example input file: inputs/action_density_viz_8x8x8x8_b56.inp")
        sys.exit(1)


if __name__ == '__main__':
    main()
