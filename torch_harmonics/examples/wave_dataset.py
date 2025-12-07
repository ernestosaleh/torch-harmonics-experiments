import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics.quadrature import legendre_gauss_weights
import matplotlib.pyplot as plt

# Create a random number generator
rng = np.random.default_rng()

def sph2cartR(theta, phi, R):
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

class WaveDataset(torch.utils.data.Dataset):
    """
    Dataset that returns (u0, u_t) of the analytical spherical wave
    evaluated at user-provided XYZ points on a sphere of radius R.

    Parameters
    ----------
    t        : float or 1D array-like of times at which to evaluate the solution
    f_handle : callable (x,y,z) -> initial displacement on sphere
    g_handle : callable (x,y,z) -> initial velocity on sphere
    c        : float, wave speed along surface
    R        : float, sphere radius
    dims     : Number of latitude and longitude points, by default (384, 768)
    lmax     : int, maximum spherical-harmonic degree (bandlimit)

    Optional behavior
    -----------------
    - num_examples: how many samples the dataset exposes (will cycle through t if array)
    - device: 'cuda' or 'cpu' for SHT ops
    - interp_mode: 'nearest' (default) or 'bilinear' for sampling from (φ,θ) grid to XYZ
    """

    def __init__(self,
                 t0,
                 dt,
                 c,
                 R,
                 dims = (384, 768),
                 t_range = None,
                 lmax = None,
                 device='cpu',
                 num_examples=256,
                 family="gaussian",
                 normalize = True,
                 history = 1,
                 nfuture=0): 
        
        super().__init__()
        self.device = torch.device(device)
        self.dt = dt
        self.num_examples = int(num_examples)
        self.family = family
        self.normalize = normalize
        self.history = history # number of past timesteps, must be >0
        self.nfuture = nfuture
        
        # Time settings
        if t_range == None:
            self.t_vec = np.arange(t0, t0+dt*(history+1e-8), dt)
        else:
            assert isinstance(t_range, tuple) and len(t_range) == 2
            self.t_vec = np.arange(t_range[0], t_range[1]+10e-8, dt)
            
        # Save basic params
        self.c = float(c)
        self.R = float(R)

        # Resolution exact for bandlimit
        if lmax:
            self.nlat = self.lmax + 1
            self.nlon = 2*self.lmax + 1
        else:
            self.nlat = dims[0]
            self.nlon = dims[1]

        # ---- Build Legendre–Gauss grid (exact for bandlimit) ----
        mu, w_mu = legendre_gauss_weights(self.nlat)
        self.quad_weights = w_mu.reshape(-1, 1)
        self.theta_q = np.arccos(mu).flip(0)                         # [0, pi]
        self.phi_q   = np.linspace(0.0, 2*np.pi, self.nlon+1)[:-1]  # [0, 2π)

        PH, TH = np.meshgrid(self.phi_q, self.theta_q, indexing='xy')  # shapes (nlat, nlon)
        
        self._TH = TH
        self._PH = PH

        # Cartesian grid on sphere (nlat, nlon, 3)
        XYZq = sph2cartR(TH, PH, self.R)
        self._X, self._Y, self._Z = XYZq[..., 0].T, XYZq[..., 1].T, XYZq[..., 2].T
        

        # ---- Build SHT operators (torch-harmonics) ----
        self.sht  = RealSHT(self.nlat, self.nlon, grid="legendre-gauss").to(self.device)
        self.isht = InverseRealSHT(self.nlat, self.nlon, grid="legendre-gauss").to(self.device)
        
        self.lmax = lmax or self.sht.lmax -1
        self.mmax = lmax or self.sht.mmax


        # Save spectral coeffs and angular frequencies ω_l
        # We assume spectral layout with last dim = l in [0..lmax]
        Ls = np.arange(0, self.lmax+1, dtype=np.float32)
        wl = (self.c / self.R) * np.sqrt(Ls * (Ls + 1.0))  # (lmax+1,)
        self.wl =torch.as_tensor(wl, dtype=torch.float32, device=self.device)
        
        # Precompute mask for zero-frequency modes
        zi = (self.wl.abs() < 1e-15)          # bool mask on l-axis; we’ll broadcast to spectral shape
        self.zi = zi
        
        # Normalization factors:
        if self.normalize:
            f_handle = gen_random_gaussian_handle()
            inp0 = torch.tensor(f_handle(self._X, self._Y, self._Z).T)
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

        # ---- Grids for plotting --------------------------------------
        # Prepare grids for interpolation from (θ,φ) to given XYZ
        # For 'nearest' we only need indexers; for 'bilinear' we’ll compute neighbors on the fly.
        self.theta_q_t = torch.as_tensor(self.theta_q, dtype=torch.float32, device=self.device)  # (nlat,)
        self.phi_q_t   = torch.as_tensor(self.phi_q,   dtype=torch.float32, device=self.device)  # (nlon,)

        # Precompute XYZ → (θ,φ) for sampling
        #self._theta_pts, self._phi_pts = cart2sph(self.XYZ[:,0], self.XYZ[:,1], self.XYZ[:,2])
        self._theta_pts_t = torch.as_tensor(self.theta_q, dtype=torch.float32, device=self.device)
        self._phi_pts_t   = torch.as_tensor(self.phi_q,   dtype=torch.float32, device=self.device)
        

    def _func2spec(self, f_handle, g_handle):
        # ---- Sample initial conditions on the grid -----------------------------
        # Call user handles (NumPy functions are fine)
        fq = f_handle(self._X, self._Y, self._Z).T           # shape (nlat, nlon)
        gq = g_handle(self._X, self._Y, self._Z).T 
        # shape (nlat, nlon)
        # Store initial profile for visualization
        
        # To torch (float32)
        fq_t = torch.as_tensor(fq, dtype=torch.float32, device=self.device)
        gq_t = torch.as_tensor(gq, dtype=torch.float32, device=self.device)

        # ---- Forward SHT (real → complex) --------------------------------------
        # Expected shape from torch-harmonics: (m_max, l_max+1) or (nlat, nlon) → spectral
        flm = self.sht(fq_t)  # complex tensor (lmax+1, mmax)
        glm = self.sht(gq_t)  # complex tensor (lmax+1, mmax)

        flm = flm.to(torch.complex128)   # higher precision for stable closed-form
        glm = glm.to(torch.complex128)
        
        return(flm, glm)


    def update_time_vec(self, t_vec):
        """Update for another solution withoug recomputing time spectral analysis"""
        self.t_vec = np.atleast_1d(np.array(t_vec, dtype=np.float32))

    def __len__(self):
        return self.num_examples

    def _evolve_spectral_beta(self, t_val: float, 
                         flm: torch.Tensor, glm: torch.Tensor):
        """
        Closed-form spectral evolution:
        u_lm(t) = f_lm cos(ω_l t) + g_lm/ω_l sin(ω_l t); handle ω_l=0 separately.
        """
        t = torch.as_tensor(t_val, dtype=torch.float32, device=self.device)

        # Broadcast cos/sin over spectral layout.
        # We assume last dim corresponds to l (size lmax+1). If torch-harmonics packs differently,
        # adapt the broadcasting to match (most builds use (m, l) with l last).
        cos_term = torch.cos(self.wl * t).view(-1,1)     # (lmax+1,)
        sin_term = torch.sin(self.wl * t).view(-1,1)      # (lmax+1,)
        inv_wl   = torch.zeros_like(self.wl)
        inv_wl[~self.zi] = 1.0 / self.wl[~self.zi]
        inv_wl =inv_wl.view(-1,1) 

        # Align shapes to (l, m): broadcast along m automatically
        ulm = flm * cos_term + glm * (inv_wl * sin_term)
        # Zero-frequency correction: ulm[:, l=0] = f_lm + g_lm * t
        # Create an index for l=0 only if lmax>=0
        if self.lmax >= 0:
            ulm = ulm.clone()
            ulm[0, :] = flm[0, :] + glm[0, :] * t

        return ulm  # complex128    
    
    def _evolve_spectral(self, t_val, flm: torch.Tensor, glm: torch.Tensor):
        """
        Closed-form spectral evolution for arbitrary time inputs.

        Parameters
        ----------
        t_val : float or array-like
            Single time or list/array/tensor of times.
        flm : Tensor (l_max+1, m_max)
            Initial displacement spectral coeffs.
        glm : Tensor (l_max+1, m_max)
            Initial velocity spectral coeffs.

        Returns
        -------
        ulm_out : Tensor
            If t_val is scalar: (l_max+1, m_max)
            If t_val is vector of length H: (H, l_max+1, m_max)
        """

        # ---- Convert t_val to 1D tensor on correct device ----
        t = torch.as_tensor(t_val, dtype=torch.float32, device=self.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)           # shape (1,)
        history = t.shape[0]

        # ---- time broadcast: (H,1,1) ----
        t = t.view(history, 1, 1)        # broadcastable over (l,m)

        # ---- spectral constants ----
        # wl: shape (l_max+1,)
        wl = self.wl.view(1, self.lmax+1, 1)      # -> (1,L,1)
        inv_wl = torch.zeros_like(wl)
        inv_wl[:, ~self.zi, :] = 1.0 / wl[:, ~self.zi, :]

        # ---- flm & glm: (1, L, M) for broadcasting ----
        flm = flm.unsqueeze(0)   # (1,L,M)
        glm = glm.unsqueeze(0)   # (1,L,M)

        # ---- time-dependent factors: (H, L, 1) ----
        coswt = torch.cos(wl * t)   # (H,L,1)
        sinwt = torch.sin(wl * t)   # (H,L,1)

        # ---- general ℓ evolution ----
        ulm = flm * coswt + glm * (inv_wl * sinwt)   # (H,L,M)

        # ---- ℓ = 0 correction (ω₀ = 0): u₀(t) = f₀ + g₀ t ----
        # Broadcasts automatically: t.shape = (H,1,1)
        ulm[:, 0, :] = flm[:, 0, :] + glm[:, 0, :] * t[:, 0, 0].unsqueeze(-1)

        return ulm            # shape (H,L,M)

    
    def _spec2grid(self, ulm):
        """
        Inverse SHT to field on (theta_q, phi_q) grid (float32).
        """
        u_grid = self.isht(ulm.to(torch.complex64))  # back to spatial grid
        return u_grid  # shape (nlat, nlon), real float32
    
    def time2grid(self, f_handle, g_handle, t_vals):
        # Returns the solution over an specific time t_val
        flm, glm = self._func2spec(f_handle, g_handle)
        ulm = self._evolve_spectral(t_vals, flm, glm)
        tar = self._spec2grid(ulm)
        if self.normalize:
            tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)
        # unsqueeze to get a tensor of shape (vars, height, width)
        return(tar)
    
    def get_rand_sample(self):        
        # Fetchs a random gaussian solution and solves for for a random time t_init
         # If normalize is none, then takes the value of self.normalize         
                
        if self.family == "gaussian":
            f_handle = gen_random_gaussian_handle() 
        else:
            raise ValueError("Unsupported family: expected 'gaussian'.") 
        # For now, the wave is stationary. 
        g_handle = gen_velocity_handle()
        
        inp = f_handle(self._X, self._Y, self._Z).T  
        
        # generate a random t_init a generates its state inp.
        # if len(self.t_vec)==int(self.history+1):
        #     t_val = self.t_vec[-1]
        # else:
        # select the initial state at a random time within t_range
        idx_init = int(rng.integers(0, len(self.t_vec)-self.history))
        t_inits = self.t_vec[idx_init:idx_init+self.history]
        
        flm, glm = self._func2spec(f_handle, g_handle)
        
        ulm = self._evolve_spectral(t_inits, flm, glm)
        inp = self._spec2grid(ulm)
        
        t_val = self.t_vec[int(idx_init+self.history)]   

        
        if self.normalize:
            inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
        
        tar = self.time2grid(f_handle, g_handle, t_val + self.nfuture*self.dt)
        
        inp = torch.as_tensor(inp, dtype=torch.float32)
        tar = torch.as_tensor(tar, dtype=torch.float32)
        
        return inp, tar  
        
    
    def __len__(self):
        length = self.num_examples
        return length

    def __getitem__(self, idx):
        """
        Returns
        -------
        inp : (N,) torch.float32   initial displacement at XYZ
        tar : (N,) torch.float32   solution at time t[idx % len(t_vec)] at XYZ
        """
        # choose last time (cycle through t_vec)
        
        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self.get_rand_sample()
        
        return inp.clone(), tar.clone()
    
    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples
        
    def set_nfuture(self, nfuture=0):
        self.nfuture = nfuture
           
    
    # ---------------- Utils to compute l2loss on a sphere ----------------
    def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
        """Integrate the solution on the grid."""
        dlon = 2 * torch.pi / self.nlon
        radius = 1 if dimensionless else self.radius
        
        if polar_opt > 0:
            out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
        else:
            out = torch.sum(ugrid * self.quad_weights * dlon * radius**2, dim=(-2, -1))
        return out
    
    
    def plot_griddata(
        self, data, fig=None, cmap='twilight_shifted',
        vmax=None, vmin=None, projection='3d', title=None,
        antialiased=False, ax=None
    ):
        """Plotting routine for data on the grid. Requires cartopy for 3d plots."""

        lats = -1*(self.theta_q-np.pi/2)
        lons = (self.phi_q-np.pi)

        if data.is_cuda:
            data = data.cpu()
            lons = lons.cpu()
            lats = lats.cpu()

        Lons, Lats = np.meshgrid(lons.reshape(-1,1), lats.reshape(-1,1))

        # If no axis is provided, we create one that fills the figure
        if ax is None and fig is None:
            fig = plt.figure()

        if projection == 'mollweide':

            if ax is None:
                ax = fig.add_subplot(1, 1, 1, projection=projection)

            im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, vmax=vmax, vmin=vmin)
            ax.grid(True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(title)

            # (let the caller decide about colorbars)
            # plt.colorbar(im, orientation='horizontal')

        elif projection == '3d':

            import cartopy.crs as ccrs

            if ax is None:
                proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)
                ax = fig.add_subplot(1, 1, 1, projection=proj)

            Lons_deg = Lons*180/math.pi
            Lats_deg = Lats*180/math.pi

            im = ax.pcolormesh(
                Lons_deg, Lats_deg, data,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                antialiased=antialiased,
                vmax=vmax, vmin=vmin
            )
            ax.set_title(title, y=1.05)

        elif projection == 'robinson':

            import cartopy.crs as ccrs

            if ax is None:
                proj = ccrs.Robinson(central_longitude=0.0)
                ax = fig.add_subplot(1, 1, 1, projection=proj)

            Lons_deg = Lons*180/math.pi
            Lats_deg = Lats*180/math.pi

            im = ax.pcolormesh(
                Lons_deg, Lats_deg, data,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                antialiased=antialiased,
                vmax=vmax, vmin=vmin
            )
            ax.set_title(title, y=1.05)

        else:
            raise NotImplementedError

        return im

    
    def generate_video_beta(self, state_fs, t_vec = None, projection = "3d"):
        
        f_handle = state_fs[0]
        g_handle = state_fs[1]
        flm, glm = self._func2spec(f_handle, g_handle)
        
        if t_vec == None:
            t_vec = self.t_vec
        
        print("Video generator starting ...")
        # Start video generation ----------------
        fig = plt.figure(figsize=(8, 6), dpi=72)

        moviewriter = animation.writers['pillow'](fps=20)
        moviewriter.setup(fig, "wave"+'.gif', dpi=72)
        print("Video generator started")
        # --------------------------------
        for i,t_val in enumerate(t_vec):
            # Save video frame ----------
            ulm = self._evolve_spectral(t_val, flm, glm)
            tar = self._spec2grid(ulm)
            inp = self.time2grid(f_handle, g_handle, t_val)
            self.plot_griddata(inp[0], fig)
            print( f"step {i} out of {len(t_vec)}")
            plt.draw()
            moviewriter.grab_frame()

        moviewriter.finish()
        
        from matplotlib import animation


    def generate_video(self, state_fs, t_vec=None, projection="3d",
                    filename="wave.mp4", fps=20, vmin=-1, vmax=1):
        
        f_handle = state_fs[0]
        g_handle = state_fs[1]
        flm, glm = self._func2spec(f_handle, g_handle)
        
        if t_vec is None:
            t_vec = self.t_vec
        
        print("Video generator starting ...")

        # Create figure
        fig = plt.figure(figsize=(8, 6), dpi=72)

        # Use ffmpeg writer for video instead of pillow (GIF)
        Writer = animation.writers['ffmpeg']
        moviewriter = Writer(fps=fps, metadata=dict(artist='WaveSolver'), bitrate=1800)

        print(f"Video generator started. Saving to {filename}")

        # Use context manager for safety
        with moviewriter.saving(fig, filename, dpi=72):
            for i, t_val in enumerate(t_vec):
                # Evolve spectral state and get grid fields
                ulm = self._evolve_spectral(t_val, flm, glm)
                tar = self._spec2grid(ulm)
                inp = self.time2grid(f_handle, g_handle, t_val)

                # Clear previous content
                fig.clf()

                # ---- Plot the new frame ----
                # First set up the axis manually
                ax = fig.add_subplot(111, projection=projection)

                # Your plot_griddata() will automatically plot into the current axis
                im = self.plot_griddata(inp[0], fig,
                                        projection=projection,
                                        vmin=vmin, vmax=vmax
                                        )
                ax.set_axis_off()

                # After it plots, retrieve the plotted mappable:
                # For 3D surfaces:
                #   im = ax.collections[-1]
                # For 2D pcolormesh:
                #   im = ax.images[-1]
                # We handle both robustly:
                if hasattr(ax, "collections") and len(ax.collections) > 0:
                    im = ax.collections[-1]
                elif hasattr(ax, "images") and len(ax.images) > 0:
                    im = ax.images[-1]

                print(f"step {i+1} / {len(t_vec)}")
                
                # ---- STANDARDIZED COLORBAR ----
               # im must be a mappable (mesh, surface, image...)
                cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.1)
                cbar.set_label("u-value", fontsize=12)

                # Title
                ax.set_title(f"t = {t_val:.3f}", fontsize=14)
                
                plt.draw()
                moviewriter.grab_frame()

        plt.close(fig)
        print("Video generation finished.")




# ---------------- Family of functions generation ----------------

def gen_random_gaussian_handle():
    # Generate a random point on the unit sphere
    theta = np.random.uniform(0, 2*np.pi)  # azimuth angle
    phi = np.arccos(np.random.uniform(-1, 1))  # polar angle

    x0 = np.sin(phi) * np.cos(theta)
    y0 = np.sin(phi) * np.sin(theta)
    z0 = np.cos(phi)

    # generate a single random sigma in (0.1, 0.999)
    sigma = float(np.random.uniform(0.1, 0.999))
    
    def gaussian(x, y, z):
        sigma = 0.5
        dot = x * x0 + y * y0 + z * z0  # shape like x
        cos_gamma = np.clip(dot, -1.0, 1.0)
        gamma = np.arccos(cos_gamma)    # angular distance in [0, π]
        return np.exp(-0.5 * (gamma / sigma)**2)
    
    return gaussian

def gen_velocity_handle():
    g_handle = lambda x, y, z: 0 * x
    return g_handle