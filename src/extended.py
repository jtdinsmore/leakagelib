import numpy as np
from scipy.signal import convolve
from .funcs import override_matplotlib_defaults
from .settings import *
from .spectrum import EnergyDependence

VMAX = 0.5
INITIAL_RATE = 1e-3

def fit_extended(source, psfs, spectrum, is_obs, qs_obs, us_obs, initial_source_pol=None, inertia=None, num_iter=5000, max_rate=1e-2, report_frequency=50, regularize_coeff=1, energy_dependence=None):
    '''Fit for the source Q and U given an observed Q and U. The resulting source Q and U will be stored in 
    self.u_map and self.u_map.
    
    ARGUMENTS:
        - source: The Source object containing the resolved I flux. Any source polarization stored in the object is ignored
        - psf: List of PSFs for each detector
        - spectrum: The Spectrum object of the source
        - is_obs, qs_obs, us_obs; List of observed I, q (normalized), and u (normalized) maps to match. One for each detector
        - initial_source_pol: Array to initialize the gradient descent method from. If None, the observed polarization in DU1 is used
        - inertia: ``Inertia`` map for gradient descent. High inertia in a given pixel means the source polarization is moved less. If None, inertia is set to the square root of the number of counts.
        - num_iter: Number of iterations to use
        - max_rate: Maximum gradient descent rate at beginning
        - report_frequency: Frequency at which to save a snapshot of the descent algorithm's current state. None for no snapshots.
        - regularize_coeff: Strength of the regularization term. Zero is no regularization.

    RETURNS: The source q and u images and a matplotlib animation containing a gif of the algorithm's progress. If `report_frequency` was None, the animation is set to None.
    '''

    if len(psfs) != 3:
        raise Exception("You must provide 3 PSFs")

    if len(is_obs) != 3 or len(qs_obs) != 3 or len(us_obs) != 3:
        raise Exception("You must provide 3 observations of I, Q, and U.")

    # Get the leakage parameters
    if energy_dependence is None:
        energy_dependence = EnergyDependence.default(source.use_nn)
    detector_params = []
    for det in range(3):
        detector_params.append(energy_dependence.get_params(spectrum))

    # Initialization
    n_pixels = np.prod(is_obs[0].shape)
    if initial_source_pol is None:
        initial_source_pol = np.array([np.mean(qs_obs, axis=0), np.mean(us_obs, axis=0)])
        # initial_source_pol += np.random.randn(*initial_source_pol.shape) * 0.1
        initial_source_pol *= spectrum.get_avg_one_over_mu(source.use_nn)
    source_pol = initial_source_pol
    last_gradient = None
    rate = INITIAL_RATE

    laplacian_kernel = np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0],
    ])/4

    if report_frequency is not None:
        import matplotlib.pyplot as plt
        from matplotlib import animation, gridspec

        override_matplotlib_defaults()

        fig = plt.figure(figsize=(13,12))

        gs = gridspec.GridSpec(2, 2)
        ax_q = fig.add_subplot(gs[0,0])
        ax_u = fig.add_subplot(gs[0,1])
        ax_z = fig.add_subplot(gs[1,:])

        line = np.arange(len(is_obs[0])).astype(float)
        line -= np.mean(line)

        q_mesh = ax_q.pcolormesh(line, line, source_pol[0], vmax=VMAX, vmin=-VMAX, cmap="RdBu")
        u_mesh = ax_u.pcolormesh(line, line, source_pol[1], vmax=VMAX, vmin=-VMAX, cmap="RdBu")

        fig.suptitle("Iteration: 0")

        ax_q.set_title("$q_\\mathrm{src}$")
        ax_u.set_title("$u_\\mathrm{src}$")
        ax_q.set_ylabel("y [pixels]")
        for axs in [ax_q, ax_u]:
            axs.set_xlabel("x [pixels]")
            axs.set_aspect("equal")
            axs.set_xlim(line[-1], line[0])
            axs.set_ylim(line[0], line[-1])

        z_line = ax_z.plot([], [], color='k')[0]
        ax_z.set_xlabel("Iteration number")
        ax_z.set_ylabel("Z")

        fig.tight_layout()
        fig.colorbar(u_mesh, ax=axs)

        saved_frames = []
        saved_zs = []
        rates = []
        saved_iterations = []

    min_z = None
    min_source_pol = None
    if inertia is None:
        inertia = np.array([np.sqrt(source.source), np.sqrt(source.source)])
        inertia /= np.mean(inertia)
    inv_inertia = 1 / inertia

    for iter_num in range(num_iter):
        if np.any(np.isnan(source_pol)):
            raise Exception("Fitter unexpectedly encoutered NaN.")
        gradient = np.zeros_like(source_pol)
        z = 0
        source.polarize_array(source_pol)

        for det, params in enumerate(detector_params):
            leak_i, leak_q, leak_u = source.compute_leakage(psfs[det], spectrum, normalize=True)
            leak_i *= np.sum(source.source) / np.sum(leak_i) # This thing makes the analytic gradient incorrect but I don't think it matters because it ought to affect every pixel by about the same amount, effectively scaling the gradient rather than turning it and who cares about that.

            delta_q = leak_q - qs_obs[det]
            delta_u = leak_u - us_obs[det]

            z += np.sum(delta_q**2 + delta_u**2)

            i0_qq = np.flip(
                + params["mu"] * psfs[det].psf
                + params["mu_sigma_plus"] * psfs[det].d_zs
                + params["mu_k_plus"] * psfs[det].d_zk
                + params["mu_k_cross"] * psfs[det].d_xk / 2
            )
            i0_uu = np.flip(
                + params["mu"] * psfs[det].psf
                + params["mu_sigma_plus"] * psfs[det].d_zs
                + params["mu_k_plus"] * psfs[det].d_zk
                - params["mu_k_cross"] * psfs[det].d_xk / 2
            )
            i0_qu = np.flip(
                + params["mu_k_cross"] * psfs[det].d_yk / 2
            )
            q0 = np.flip(
                + params["mu_sigma_minus"] * psfs[det].d_qs
                + params["mu_k_minus"] * psfs[det].d_qk
            )
            u0 = np.flip(
                + params["mu_sigma_minus"] * psfs[det].d_us
                + params["mu_k_minus"] * psfs[det].d_uk
            )

            gradient[0] += (
                + 2 * convolve(delta_q / leak_i, i0_qq, mode="same")
                + 2 * convolve(delta_u / leak_i, i0_qu, mode="same")
                - convolve(delta_q * leak_q / leak_i, q0, mode="same")
                - convolve(delta_u * leak_u / leak_i, q0, mode="same")
            ) * source.source

            gradient[1] += (
                + 2 * convolve(delta_u / leak_i, i0_uu, mode="same")
                + 2 * convolve(delta_q / leak_i, i0_qu, mode="same")
                - convolve(delta_u * leak_u / leak_i, u0, mode="same")
                - convolve(delta_q * leak_q / leak_i, u0, mode="same")
            ) * source.source

        # Regularization
        laplacian_q = convolve(source_pol[0], laplacian_kernel, mode="same")
        laplacian_u = convolve(source_pol[1], laplacian_kernel, mode="same")

        z += regularize_coeff * np.sum(laplacian_q**2 + laplacian_u**2)

        gradient[0] += 2 * regularize_coeff * (
            convolve(laplacian_q, np.flip(laplacian_kernel), mode="same")
        )

        gradient[1] += 2 * regularize_coeff * (
            convolve(laplacian_u, np.flip(laplacian_kernel), mode="same")
        )

        # Update
        last_source_pol = np.copy(source_pol)
        source_pol -= rate * gradient * inv_inertia
        source_pol = np.clip(source_pol, -1, 1)

        if iter_num > num_iter / 2 and (min_z is None or z < min_z):
            min_source_pol = np.copy(source_pol)
            min_z = z
        
        z_norm = np.sqrt(z) / (2 * n_pixels)

        if iter_num > 100:
            delta_x = (source_pol - last_source_pol).reshape(-1)
            delta_gradient = (gradient - last_gradient).reshape(-1)
            rate = (delta_x @ delta_gradient) / (delta_gradient @ delta_gradient)
            rate = min(max_rate, np.abs(rate))

        last_gradient = np.copy(gradient)
        
        if report_frequency is not None and iter_num % report_frequency == 0:
            saved_frames.append(np.copy(source_pol))
            saved_iterations.append(iter_num)
            saved_zs.append(z_norm)
            rates.append(rate)

    if report_frequency is None:
        anim = None
    else:
        def animate(i):
            fig.suptitle(f"Iteration {i * report_frequency}\t Error per pixel {saved_zs[i]*100:.2f}%\tRate {rates[i]*1e3:.2f}e-3")
            q_mesh.set_array(saved_frames[i][0])
            u_mesh.set_array(saved_frames[i][1])

            if i >= 1:
                z_line.set_data(saved_iterations[:i], saved_zs[:i])
                min_i = max(0, i - 30)
                ax_z.set_xlim(saved_iterations[min_i], saved_iterations[i])
                min_y = np.nanmin(saved_zs[min_i:i])
                max_y = np.nanmax(saved_zs[min_i:i])
                if max_y > min_y:
                    ax_z.set_ylim(min_y, max_y)
        anim = animation.FuncAnimation(fig,animate,frames=len(saved_frames),interval=150,blit=False,repeat=True)

    return min_source_pol[0], min_source_pol[1], anim