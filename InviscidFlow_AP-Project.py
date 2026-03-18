import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Users\dipta\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import os
from datetime import datetime


#  NACA 4-digit airfoil geometry helper

def naca4_mask(X, Y, naca='2412', chord=60.0, aoa_deg=5.0,
               leading_x=None, leading_y=None):
    """
    Return a boolean mask that is True inside the NACA 4-digit airfoil.

    Parameters:
    X, Y        : 2-D meshgrids (grid units)
    naca        : 4-digit NACA string
    chord       : chord length in grid units
    aoa_deg     : angle of attack (positive = nose up)
    leading_x   : x-coordinate of leading edge centre  (defaults to grid left quarter)
    leading_y   : y-coordinate of leading edge centre  (defaults to grid midheight)
    """
    ny, nx = X.shape
    if leading_x is None:
        leading_x = nx * 0.25
    if leading_y is None:
        leading_y = ny * 0.5

    m  = int(naca[0]) / 100.0   # max camber
    p  = int(naca[1]) / 10.0    # location of max camber
    t  = int(naca[2:]) / 100.0  # max thickness

    aoa = np.radians(aoa_deg)
    cos_a, sin_a = np.cos(-aoa), np.sin(-aoa)

    # Rotate grid points into airfoil frame
    dx = X - leading_x
    dy = Y - leading_y
    xr =  dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    # Normalise to chord
    xc = xr / chord

    inside = np.zeros(X.shape, dtype=bool)
    valid  = (xc >= 0) & (xc <= 1)

    # NACA symmetric formula
    xt = xc[valid]
    yt = (t / 0.2) * (0.2969*np.sqrt(xt)
                      - 0.1260*xt
                      - 0.3516*xt**2
                      + 0.2843*xt**3
                      - 0.1015*xt**4)

    # Camber line
    yc = np.where(xt < p,
                  m / p**2 * (2*p*xt - xt**2),
                  m / (1-p)**2 * ((1 - 2*p) + 2*p*xt - xt**2))

    yr_valid = yr[valid]
    in_airfoil = (yr_valid >= (yc - yt) * chord) & \
                 (yr_valid <= (yc + yt) * chord)

    inside[valid] = in_airfoil
    return inside


def naca4_smooth_outline(naca='2412', chord=60.0, aoa_deg=5.0,
                         leading_x=0.0, leading_y=0.0, n_points=300):
    """
    Return (x, y) arrays of a smooth, analytically-computed NACA airfoil outline
    suitable for plotting. Uses cosine spacing for better LE/TE resolution.
    """
    m = int(naca[0]) / 100.0
    p = int(naca[1]) / 10.0
    t = int(naca[2:]) / 100.0

    # Cosine spacing for better leading/trailing edge resolution
    beta = np.linspace(0, np.pi, n_points)
    xc   = 0.5 * (1 - np.cos(beta))          # 0 → 1

    # Thickness distribution (open trailing edge variant)
    yt = (t / 0.2) * (0.2969*np.sqrt(xc)
                      - 0.1260*xc
                      - 0.3516*xc**2
                      + 0.2843*xc**3
                      - 0.1015*xc**4)

    # Camber line & gradient
    if p > 0:
        yc  = np.where(xc < p,
                       m / p**2 * (2*p*xc - xc**2),
                       m / (1-p)**2 * ((1 - 2*p) + 2*p*xc - xc**2))
        dyc = np.where(xc < p,
                       2*m / p**2 * (p - xc),
                       2*m / (1-p)**2 * (p - xc))
    else:
        yc  = np.zeros_like(xc)
        dyc = np.zeros_like(xc)

    theta = np.arctan(dyc)

    # Upper / lower surface in normalised coords
    xu = xc  - yt * np.sin(theta)
    yu = yc  + yt * np.cos(theta)
    xl = xc  + yt * np.sin(theta)
    yl = yc  - yt * np.cos(theta)

    # Closed loop
    x_norm = np.concatenate([xu, xl[::-1]])
    y_norm = np.concatenate([yu, yl[::-1]])

    # Scale to chord
    x_chord = x_norm * chord
    y_chord = y_norm * chord

    # Rotate by angle of attack
    aoa = np.radians(aoa_deg)
    cos_a, sin_a = np.cos(aoa), np.sin(aoa)
    x_rot =  x_chord * cos_a + y_chord * sin_a
    y_rot = -x_chord * sin_a + y_chord * cos_a

    # Translate to leading-edge position
    x_out = x_rot + leading_x
    y_out = y_rot + leading_y

    return x_out, y_out


def naca4_boundary_and_angles(mask, center_x, center_y):
    """Return boundary indices and angles (measured from leading-edge centre)."""
    cyl = mask
    ny, nx = cyl.shape
    boundary = np.zeros_like(cyl, bool)
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        shifted = np.zeros_like(cyl)
        shifted[max(0,di):min(ny,ny+di),
                max(0,dj):min(nx,nx+dj)] = \
            cyl[max(0,-di):min(ny,ny-di),
                max(0,-dj):min(nx,nx-dj)]
        boundary |= cyl & ~shifted

    boundary_idx = np.argwhere(boundary)
    angles = np.degrees(
        np.arctan2(boundary_idx[:,0] - center_y,
                   boundary_idx[:,1] - center_x)
    ) % 360
    return boundary_idx, angles



#  Main simulation class

class ParticleAdvectionSimulation:
    """
    Simulate inviscid, incompressible flow particle advection around either:
      - a circular cylinder   (body='cylinder')
      - a NACA 4-digit airfoil (body='naca', naca_code='2412', aoa_deg=5)

    Computes lift and drag coefficients vs iterations, animates tracer particles,
    and efficiently stores flow-field snapshots for later POD analysis.
    """

    def __init__(self,
                 grid_size=100,
                 aspect_ratio=3,
                 max_particles=np.inf,
                 dt=0.1,
                 rho=1.0,
                 U_inf=1.0,
                 data_directory='flow_data',
                 # body selection
                 body='cylinder',
                 # NACA-specific
                 naca_code='2412',
                 aoa_deg=5.0):

        # Basic parameters
        self.s             = grid_size
        self.ar            = aspect_ratio
        self.dt            = dt
        self.rho           = rho
        self.U_inf         = U_inf
        self.max_particles = max_particles
        self.time_elapsed  = 0.0
        self.iteration     = 0

        self.body      = body.lower()
        self.naca_code = naca_code
        self.aoa_deg   = aoa_deg

        #  Body geometry 
        if self.body == 'cylinder':
            self.radius   = 10.0
            self.D        = 2 * self.radius
            self.center_x = self.s * self.ar / 4.0
            self.center_y = self.s / 2.0
        elif self.body == 'naca':
            # chord ~ 1/5 of domain width; leading edge at 1/4 of domain
            self.chord    = self.s * self.ar / 5.0
            self.D        = self.chord          # reference length for Cd/Cl
            self.center_x = self.s * self.ar * 0.25   # leading edge x
            self.center_y = self.s / 2.0
        else:
            raise ValueError(f"Unknown body type '{body}'. Choose 'cylinder' or 'naca'.")

        # Computational grid
        self.grid_y = np.arange(1, self.s + 1)
        self.grid_x = np.arange(1, self.s * self.ar + 1)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)

        # Flow fields
        self.p  = np.zeros((self.s, self.s * self.ar))
        self.vx = np.zeros_like(self.p)
        self.vy = np.zeros_like(self.p)

        # Derived (not stored between steps)
        self.vorticity          = np.zeros_like(self.p)
        self.velocity_magnitude = np.zeros_like(self.p)

        # Body mask & boundary
        self._build_body_mask()

        # Tracer particles
        self._init_particles()

        # Jacobi kernel
        self.J = np.array([[0,1,0],[1,0,1],[0,1,0]], float) / 4.0

        # Histories
        self.iterations = []
        self.times      = []
        self.Cd_list    = []
        self.Cl_list    = []
        self.Cp_history = []

        # Snapshots
        self.snapshot_iterations = []
        self.vx_snapshots        = []
        self.vy_snapshots        = []
        self.p_snapshots         = []

        # Angular bins (used for Cp distribution)
        self.theta_deg = np.linspace(0, 360, 72, endpoint=False)

        # Data directory
        self.data_directory = data_directory
        os.makedirs(self.data_directory, exist_ok=True)

    # Body mask helpers 

    def _build_body_mask(self):
        if self.body == 'cylinder':
            self.cyl_mask = (
                (self.X - self.center_x)**2 +
                (self.Y - self.center_y)**2
            ) <= self.radius**2
        else:
            # Smooth the raw boolean mask with a Gaussian to reduce staircase
            # edges, then re-threshold so interior/exterior stay crisp
            raw_mask = naca4_mask(
                self.X, self.Y,
                naca=self.naca_code,
                chord=self.chord,
                aoa_deg=self.aoa_deg,
                leading_x=self.center_x,
                leading_y=self.center_y
            )
            smoothed = gaussian_filter(raw_mask.astype(float), sigma=0.8)
            self.cyl_mask = smoothed > 0.5
        self._find_boundary()

    def _find_boundary(self):
        cyl  = self.cyl_mask
        ny, nx = cyl.shape
        boundary = np.zeros_like(cyl, bool)
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            shifted = np.zeros_like(cyl)
            shifted[max(0,di):min(ny,ny+di),
                    max(0,dj):min(nx,nx+dj)] = \
                cyl[max(0,-di):min(ny,ny-di),
                    max(0,-dj):min(nx,nx-dj)]
            boundary |= cyl & ~shifted

        self.boundary_idx    = np.argwhere(boundary)
        self.boundary_angles = np.degrees(
            np.arctan2(self.boundary_idx[:,0] - self.center_y,
                       self.boundary_idx[:,1] - self.center_x)
        ) % 360

    #  Particle initialisation 

    def _init_particles(self):
        self.px  = np.full((self.s,), 10.0)
        self.py  = np.arange(1, self.s + 1, dtype=float)
        self.pxo = self.px.copy()
        self.pyo = self.py.copy()

    #  Interpolation & RK4 

    def interp2d(self, field, x, y):
        interp = RegularGridInterpolator(
            (self.grid_y, self.grid_x),
            field,
            bounds_error=False,
            fill_value=0.0
        )
        pts = np.column_stack((y.flatten(), x.flatten()))
        return interp(pts).reshape(x.shape)

    def rk4(self, px, py, h):
        k1x = self.interp2d(self.vx, px, py)
        k1y = self.interp2d(self.vy, px, py)
        k2x = self.interp2d(self.vx, px + 0.5*h*k1x, py + 0.5*h*k1y)
        k2y = self.interp2d(self.vy, px + 0.5*h*k1x, py + 0.5*h*k1y)
        k3x = self.interp2d(self.vx, px + 0.5*h*k2x, py + 0.5*h*k2y)
        k3y = self.interp2d(self.vy, px + 0.5*h*k2x, py + 0.5*h*k2y)
        k4x = self.interp2d(self.vx, px +   h*k3x, py +   h*k3y)
        k4y = self.interp2d(self.vy, px +   h*k3x, py +   h*k3y)
        new_px = px + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)
        new_py = py + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)
        return new_px, new_py

    #  Physics 

    def compute_derived_quantities(self):
        dvdx = np.gradient(self.vy, axis=1)
        dudy = np.gradient(self.vx, axis=0)
        self.vorticity          = dvdx - dudy
        self.velocity_magnitude = np.sqrt(self.vx**2 + self.vy**2)

    def compute_forces(self):
        F  = np.zeros(2)
        ds = 1.0
        for i, j in self.boundary_idx:
            nx_c = (self.X[i,j] - self.center_x)
            ny_c = (self.Y[i,j] - self.center_y)
            norm = np.hypot(nx_c, ny_c) + 1e-12
            nx_c, ny_c = nx_c/norm, ny_c/norm
            F += -self.p[i,j] * np.array([nx_c, ny_c]) * ds

        q  = 0.5 * self.rho * self.U_inf**2
        Cd = F[0] / (q * self.D)
        Cl = F[1] / (q * self.D)
        return Cd, Cl

    def compute_pressure_coefficient(self):
        q       = 0.5 * self.rho * self.U_inf**2
        p_vals  = self.p[self.boundary_idx[:,0], self.boundary_idx[:,1]]
        Cp_vals = p_vals / q
        bins    = np.digitize(self.boundary_angles,
                              np.linspace(0, 360, len(self.theta_deg)+1))
        Cp_binned = np.zeros(len(self.theta_deg))
        counts    = np.zeros(len(self.theta_deg))
        for idx, b in enumerate(bins):
            if 1 <= b <= len(self.theta_deg):
                Cp_binned[b-1] += Cp_vals[idx]
                counts[b-1]    += 1
        mask = counts > 0
        Cp_binned[mask] /= counts[mask]
        return Cp_binned

    def update(self):
        # Zero inside body
        self.vx[self.cyl_mask] = 0.0
        self.vy[self.cyl_mask] = 0.0

        # Pressure solve (Jacobi iterations)
        rhs = 0.5 * (np.gradient(self.vx, axis=1) +
                     np.gradient(self.vy, axis=0))
        for _ in range(100):
            self.p = convolve2d(self.p, self.J, mode='same') - rhs
            self.p[0,:], self.p[-1,:] = self.p[1,:], self.p[-2,:]
            self.p[:,:5], self.p[:,-5:] = 1.0, 0.0

        # Velocity correction
        dpdx = np.gradient(self.p, axis=1)
        dpdy = np.gradient(self.p, axis=0)
        self.vx[1:-1,1:-1] -= dpdx[1:-1,1:-1]
        self.vy[1:-1,1:-1] -= dpdy[1:-1,1:-1]
        self.vx[self.cyl_mask] = 0.0
        self.vy[self.cyl_mask] = 0.0

        # Semi-Lagrangian advection
        backx, backy = self.rk4(self.X, self.Y, -self.dt)
        self.vx = self.interp2d(self.vx, backx, backy)
        self.vy = self.interp2d(self.vy, backx, backy)

        self.compute_derived_quantities()

        self.time_elapsed += self.dt
        self.iteration    += 1

        # Tracer particles
        self.px, self.py = self.rk4(self.px, self.py, self.dt)
        self.px = np.concatenate((self.px, self.pxo))
        self.py = np.concatenate((self.py, self.pyo))
        if self.px.size > self.max_particles:
            self.px = self.px[-int(self.max_particles):]
            self.py = self.py[-int(self.max_particles):]

        # Remove particles inside body or outside domain
        inside_body = self.interp2d(
            self.cyl_mask.astype(float), self.px, self.py) > 0.5
        in_domain   = ((self.px >= 0) &
                       (self.px < self.s * self.ar) &
                       (self.py >= 0) &
                       (self.py < self.s))
        keep = in_domain & ~inside_body
        self.px, self.py = self.px[keep], self.py[keep]

        # Forces & Cp
        Cd, Cl = self.compute_forces()
        Cp     = self.compute_pressure_coefficient()

        self.iterations.append(self.iteration)
        self.times.append(self.time_elapsed)
        self.Cd_list.append(Cd)
        self.Cl_list.append(Cl)
        self.Cp_history.append(Cp)

    #  I/O  (numpy .npz instead of h5py)

    def save_forces(self, filepath):
        # Replace .h5 extension with .npz
        filepath = os.path.splitext(filepath)[0] + '.npz'
        meta = {
            'body':      self.body,
            'rho':       self.rho,
            'U_inf':     self.U_inf,
            'D':         self.D,
            'timestamp': datetime.now().isoformat()
        }
        if self.body == 'naca':
            meta['naca_code'] = self.naca_code
            meta['aoa_deg']   = self.aoa_deg

        np.savez_compressed(
            filepath,
            iteration  = np.array(self.iterations),
            time       = np.array(self.times),
            Cd         = np.array(self.Cd_list),
            Cl         = np.array(self.Cl_list),
            theta_deg  = self.theta_deg,
            Cp_history = np.array(self.Cp_history),
            meta       = np.array([str(meta)])   # store dict as string
        )
        print(f"Forces saved -> {filepath}")

    def save_flow_field_hdf5(self, filepath):
        # Replace .h5 extension with .npz
        filepath = os.path.splitext(filepath)[0] + '.npz'

        vx = np.array(self.vx_snapshots, dtype=np.float32)
        vy = np.array(self.vy_snapshots, dtype=np.float32)
        p  = np.array(self.p_snapshots,  dtype=np.float32)

        if vx.shape[0] == 0:
            print("Warning: no snapshots to save.")
            return

        meta = {
            'body':            self.body,
            'grid_size':       self.s,
            'aspect_ratio':    self.ar,
            'cylinder_center': (self.center_x, self.center_y),
            'dt':              self.dt,
            'rho':             self.rho,
            'U_inf':           self.U_inf,
            'timestamp':       datetime.now().isoformat()
        }
        if self.body == 'cylinder':
            meta['cylinder_radius'] = self.radius
        else:
            meta['naca_code'] = self.naca_code
            meta['chord']     = self.chord
            meta['aoa_deg']   = self.aoa_deg

        np.savez_compressed(
            filepath,
            iterations = np.array(self.snapshot_iterations),
            vx         = vx,
            vy         = vy,
            p          = p,
            meta       = np.array([str(meta)])
        )
        print(f"Compressed flow fields saved -> {filepath}")

    #  Run 

    def run(self,
            LENGTH=3.0,
            HEIGHT=1.0,
            final_time=10.0,
            snapshot_interval=0.5,
            save_anim='advection.mp4',
            save_forces='forces.npz',
            save_fields='flow_fields.npz'):

        n_iters   = int(final_time / self.dt)
        snap_step = max(1, int(snapshot_interval / self.dt))

        dx_anim = LENGTH / (self.s * self.ar - 1)
        dy_anim = HEIGHT / (self.s - 1)

        #  Figure setup 
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_xlim(0, LENGTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x', color='white')
        ax.set_ylabel('y', color='white')
        body_label = ('Cylinder' if self.body == 'cylinder'
                      else f'NACA {self.naca_code}  (AoA={self.aoa_deg}°)')
        ax.set_title(f'Particle Advection — {body_label}', color='white')
        ax.tick_params(colors='white')

        # Draw body outline
        if self.body == 'cylinder':
            cx_a = (self.center_x - 1) * dx_anim
            cy_a = (self.center_y - 1) * dy_anim
            r_a  = self.radius * dx_anim
            body_patch = plt.Circle((cx_a, cy_a), r_a,
                                    edgecolor='cyan', facecolor='cyan',
                                    alpha=0.7)
            ax.add_patch(body_patch)
        else:
            # Draw smooth analytic airfoil outline (no jagged pixel edges)
            sx, sy = naca4_smooth_outline(
                naca       = self.naca_code,
                chord      = self.chord,
                aoa_deg    = self.aoa_deg,
                leading_x  = self.center_x,
                leading_y  = self.center_y,
                n_points   = 400,
            )
            # Convert from grid units to animation units
            sx_a = (sx - 1) * dx_anim
            sy_a = (sy - 1) * dy_anim
            airfoil_patch = plt.Polygon(
                np.column_stack((sx_a, sy_a)),
                closed=True, edgecolor='cyan', facecolor='cyan',
                alpha=0.7, linewidth=1.5, zorder=3,
            )
            ax.add_patch(airfoil_patch)

        scatter = ax.scatter([], [], s=0.5, c='white', alpha=0.75)

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            return scatter,

        def animate(frame):
            self.update()
            if self.iteration % snap_step == 0:
                self.snapshot_iterations.append(self.iteration)
                self.vx_snapshots.append(self.vx.copy())
                self.vy_snapshots.append(self.vy.copy())
                self.p_snapshots.append(self.p.copy())

            x_p = (self.px - 1) * dx_anim
            y_p = (self.py - 1) * dy_anim
            scatter.set_offsets(np.column_stack((x_p, y_p)))

            perc  = (frame + 1) / n_iters
            bar_n = int(perc * 40)
            bar   = '#' * bar_n + '-' * (40 - bar_n)
            print(f"\rIter {self.iteration}/{n_iters}: "
                  f"|{bar}| {perc*100:5.1f}% ", end='', flush=True)
            return scatter,

        ani = FuncAnimation(fig, animate, frames=n_iters,
                            init_func=init, blit=True,
                            interval=self.dt * 1000, repeat=False)
        ani.save(save_anim, writer='ffmpeg', fps=60, dpi=300)
        plt.close(fig)
        print(f"\nAnimation saved -> {save_anim}")

        #  Save data 
        self.save_forces(os.path.join(self.data_directory, save_forces))
        self.save_flow_field_hdf5(os.path.join(self.data_directory, save_fields))

        #  Plot force coefficients 
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.Cd_list, label='$C_D$')
        plt.plot(self.iterations, self.Cl_list, label='$C_L$')
        plt.xlabel('Iteration')
        plt.ylabel('Coefficient')
        plt.legend()
        plt.grid(True)
        plt.title(f'Force Coefficients vs Iteration — {body_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_directory,
                                 'force_coefficients.png'), dpi=300)
        plt.close()

        #  Plot average Cp 
        avg_Cp = np.mean(self.Cp_history, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(self.theta_deg, avg_Cp, '-o')
        plt.xlabel('Angle (deg)')
        plt.ylabel('Average $C_p$')
        plt.grid(True)
        plt.title(f'Average Pressure Coefficient — {body_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_directory,
                                 'pressure_coefficient.png'), dpi=300)
        plt.close()

        return ani


if __name__ == '__main__':

    print("  Inviscid Flow Particle Advection Simulation")
    print()
    print("Select geometry:")
    print("  1 — Circular cylinder")
    print("  2 — NACA airfoil")
    print()

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            break
        print("  Invalid input. Please enter 1 or 2.")

    if choice == '1':
        #  Cylinder 
        print("\n[Cylinder mode selected]\n")
        sim = ParticleAdvectionSimulation(
            grid_size       = 128,
            aspect_ratio    = 3,
            max_particles   = np.inf,
            dt              = 1.0,
            rho             = 1.0,
            U_inf           = 1.0,
            data_directory  = 'cylinder_flow_data',
            body            = 'cylinder',
        )
        sim.run(
            LENGTH            = 3.0,
            HEIGHT            = 1.0,
            final_time        = 1000.0,
            snapshot_interval = 0.5,
            save_anim         = 'cylinder_advection.mp4',
            save_forces       = 'forces.npz',
            save_fields       = 'flow_fields.npz',
        )

    else:
        #  NACA airfoil 
        print()
        naca_code = input(
            "Enter 4-digit NACA code (e.g. 2412) [default: 2412]: "
        ).strip() or '2412'

        while True:
            aoa_input = input(
                "Enter angle of attack in degrees (e.g. 5) [default: 5]: "
            ).strip() or '5'
            try:
                aoa_deg = float(aoa_input)
                break
            except ValueError:
                print("  Please enter a numeric value.")

        print(f"\n[NACA {naca_code}  AoA={aoa_deg}° ]\n")

        sim = ParticleAdvectionSimulation(
            grid_size       = 128,
            aspect_ratio    = 3,
            max_particles   = np.inf,
            dt              = 1.0,
            rho             = 1.0,
            U_inf           = 1.0,
            data_directory  = f'naca{naca_code}_{aoa_deg}_flow_data',
            body            = 'naca',
            naca_code       = naca_code,
            aoa_deg         = aoa_deg,
        )
        sim.run(
            LENGTH            = 3.0,
            HEIGHT            = 1.0,
            final_time        = 1000.0,
            snapshot_interval = 0.5,
            save_anim         = f'naca{naca_code}_{aoa_deg}_advection.mp4',
            save_forces       = 'forces.npz',
            save_fields       = 'flow_fields.npz',
        )