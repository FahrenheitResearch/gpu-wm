#!/usr/bin/env python3
"""
GPU-WM Output Visualization
Reads binary output files and creates plots
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import glob

def read_output(filename):
    """Read a GPU-WM binary output file."""
    with open(filename, 'rb') as f:
        # Header: nx(int), ny(int), nz(int), time(double), dx(double), dy(double), ztop(double)
        nx = struct.unpack('i', f.read(4))[0]
        ny = struct.unpack('i', f.read(4))[0]
        nz = struct.unpack('i', f.read(4))[0]
        # Padding for struct alignment
        f.read(4)  # padding
        time = struct.unpack('d', f.read(8))[0]
        dx = struct.unpack('d', f.read(8))[0]
        dy = struct.unpack('d', f.read(8))[0]
        ztop = struct.unpack('d', f.read(8))[0]

        n3d = nx * ny * nz

        # Z levels
        z = np.frombuffer(f.read(nz * 8), dtype=np.float64)

        # Fields
        fields = {}
        field_names = ['u', 'v', 'w', 'theta', 'qv', 'qc', 'qr', 'p', 'rho']
        for name in field_names:
            data = np.frombuffer(f.read(n3d * 8), dtype=np.float64)
            fields[name] = data.reshape((nz, ny, nx))

    return {
        'nx': nx, 'ny': ny, 'nz': nz,
        'time': time, 'dx': dx, 'dy': dy, 'ztop': ztop,
        'z': z, 'fields': fields
    }


def plot_cross_section(data, output_dir='plots'):
    """Create cross-section plots through domain center."""
    os.makedirs(output_dir, exist_ok=True)

    fields = data['fields']
    nx, ny, nz = data['nx'], data['ny'], data['nz']
    dx, dy = data['dx'], data['dy']
    z = data['z']
    time = data['time']

    jc = ny // 2  # Y cross-section through center

    x = np.arange(nx) * dx / 1000.0  # km

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GPU-WM  t = {time:.0f} s ({time/60:.1f} min)', fontsize=16)

    # Potential temperature perturbation
    ax = axes[0, 0]
    theta_pert = fields['theta'][:, jc, :] - np.mean(fields['theta'][:, jc, :], axis=1, keepdims=True)
    c = ax.pcolormesh(x, z/1000, theta_pert, cmap='RdBu_r', shading='auto')
    plt.colorbar(c, ax=ax, label='K')
    ax.set_title('θ perturbation')
    ax.set_ylabel('Height (km)')

    # Vertical velocity
    ax = axes[0, 1]
    c = ax.pcolormesh(x, z/1000, fields['w'][:, jc, :], cmap='RdBu_r', shading='auto')
    plt.colorbar(c, ax=ax, label='m/s')
    ax.set_title('Vertical velocity (w)')

    # Horizontal wind speed
    ax = axes[0, 2]
    wspd = np.sqrt(fields['u'][:, jc, :]**2 + fields['v'][:, jc, :]**2)
    c = ax.pcolormesh(x, z/1000, wspd, cmap='viridis', shading='auto')
    plt.colorbar(c, ax=ax, label='m/s')
    ax.set_title('Wind speed')

    # Cloud water
    ax = axes[1, 0]
    c = ax.pcolormesh(x, z/1000, fields['qc'][:, jc, :] * 1000, cmap='Blues', shading='auto')
    plt.colorbar(c, ax=ax, label='g/kg')
    ax.set_title('Cloud water (qc)')
    ax.set_ylabel('Height (km)')
    ax.set_xlabel('x (km)')

    # Rain water
    ax = axes[1, 1]
    c = ax.pcolormesh(x, z/1000, fields['qr'][:, jc, :] * 1000, cmap='Greens', shading='auto')
    plt.colorbar(c, ax=ax, label='g/kg')
    ax.set_title('Rain water (qr)')
    ax.set_xlabel('x (km)')

    # Water vapor
    ax = axes[1, 2]
    c = ax.pcolormesh(x, z/1000, fields['qv'][:, jc, :] * 1000, cmap='YlGnBu', shading='auto')
    plt.colorbar(c, ax=ax, label='g/kg')
    ax.set_title('Water vapor (qv)')
    ax.set_xlabel('x (km)')

    plt.tight_layout()
    outfile = os.path.join(output_dir, f'gpuwm_t{time:08.0f}.png')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f'Saved: {outfile}')


def plot_plan_view(data, level_km=1.0, output_dir='plots'):
    """Create plan-view plots at a given height."""
    os.makedirs(output_dir, exist_ok=True)

    fields = data['fields']
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    z = data['z']
    time = data['time']

    # Find nearest level
    k = np.argmin(np.abs(z/1000 - level_km))

    x = np.arange(nx) * dx / 1000
    y = np.arange(ny) * dy / 1000

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'GPU-WM  z={z[k]/1000:.1f} km  t={time:.0f}s', fontsize=14)

    ax = axes[0]
    c = ax.pcolormesh(x, y, fields['w'][k], cmap='RdBu_r', shading='auto')
    plt.colorbar(c, ax=ax, label='m/s')
    ax.set_title('w')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')

    ax = axes[1]
    theta_pert = fields['theta'][k] - np.mean(fields['theta'][k])
    c = ax.pcolormesh(x, y, theta_pert, cmap='RdBu_r', shading='auto')
    plt.colorbar(c, ax=ax, label='K')
    ax.set_title('θ perturbation')
    ax.set_xlabel('x (km)')
    ax.set_aspect('equal')

    ax = axes[2]
    c = ax.pcolormesh(x, y, fields['qc'][k]*1000, cmap='Blues', shading='auto')
    plt.colorbar(c, ax=ax, label='g/kg')
    ax.set_title('Cloud water')
    ax.set_xlabel('x (km)')
    ax.set_aspect('equal')

    plt.tight_layout()
    outfile = os.path.join(output_dir, f'gpuwm_plan_t{time:08.0f}.png')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Process all output files
        files = sorted(glob.glob('output/gpuwm_*.bin'))
        if not files:
            print('No output files found in output/')
            sys.exit(1)
    else:
        files = sys.argv[1:]

    for f in files:
        print(f'Reading {f}...')
        data = read_output(f)
        plot_cross_section(data)
        plot_plan_view(data)
