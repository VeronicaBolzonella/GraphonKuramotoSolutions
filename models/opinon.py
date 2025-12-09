import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import diffrax
import numpy as np

class OpinionGraph():
    def __init__(self, n, alpha=5.0, gamma=1.0, u0=None):
        self.n = n
        self.alpha = alpha # peer pressure strength
        self.gamma_base = gamma # external pull strength
        
        if u0 is not None:
            self.u = u0
        else:
            # random starting opinions in [0, 1]
            self.u = jr.uniform(jr.PRNGKey(99), (n,))
            
        self.x = None
        self.D = None # distance matrix
        self.A = None # adjacency matrix of couplings weigths
        
        self.O_ext = jnp.ones(n) * 0.5 # opinion of external forces
        self.Gamma = jnp.ones(n) * gamma 

    def positions(self):
        """Position nodes on a domain [0, 1], with periodicity"""
        self.x = jnp.linspace(0, 1, self.n, endpoint=False)
        
    def distances(self):
        """Compute circular distances on domain [0, 1]"""
        D_row = self.x[:, None]
        D_col = self.x[None, :]
        D_total = jnp.abs(D_row - D_col)
        self.D = jnp.minimum(D_total, 1.0 - D_total)
        
    def R(self, d, k, lam):
        """Exponential kernel graphon"""
        local_coupling = k * jnp.exp(-d/lam)
        return local_coupling
    
    def distance_based_graphon(self, k=1.0, lam=0.1):
        """Build contact network based on distance"""
        self.A = self.R(self.D, k, lam)
        return self.A

    def set_external_field(self, O_func=None, Gamma_func=None):
        """
        Opinion field that defines how media and influencers behave
        Args:
            O_func: function taking x (position) returning target opinion.
            Gamma_func: function taking x (position) returning strength.
        """
        if self.x is None: self.positions()
        
        if O_func:
            self.O_ext = O_func(self.x)
        else:
            self.O_ext = jnp.ones(self.n) * 0.5 # default 

        if Gamma_func:
            self.Gamma = Gamma_func(self.x)
        else:
            self.Gamma = jnp.ones(self.n) * self.gamma_base # default 

    def solve(self, T=10.0, dt=0.01, phi_type='linear', epsilon=0.2):
        """
        Simulate dynamics in a time frame T.

        Args:
            T (float, optional): end time. Defaults to 10.0.
            dt (float, optional): step size. Defaults to 0.01.
            phi_type (str, optional): interaction kernel type, only linear implemented. Defaults to 'linear'.
            epsilon (float, optional): _description_. Defaults to 0.2.
        """
        def dynamics(t, u, args):
            """
            Opinion Dynamics:
            du_i/dt = alpha  * Sum A_ij * phi(|u_j-u_i|) * (u_j - u_i)
                      + Gamma_i * (O_ext_i - u_i)
            """
            N = self.n
            
            # matrix of opinion differences 
            diff_matrix = u[None, :] - u[:, None]
            
            # Interaction Kernel phi(u_j - u_i)
            if phi_type != 'linear':
                raise ValueError("Only linear kernel implemented")
            
            social = (self.alpha / N) * jnp.sum(self.A * diff_matrix, axis=1)
            external = self.Gamma * (self.O_ext - u)
            return social + external
        
        term = diffrax.ODETerm(dynamics)
        solver = diffrax.Tsit5() 
        
        ts = jnp.linspace(0, T, int(T/dt) + 1)
        sol = diffrax.diffeqsolve(
            term, solver, t0=0, t1=T, dt0=dt,
            y0=self.u, saveat=diffrax.SaveAt(ts=ts)
        )
        return sol

    def plot_graphon(self, title="Graphon Adjacency"):
        plt.figure(figsize=(5,5))
        plt.imshow(self.A, cmap='gray', origin='lower')
        plt.colorbar(label='Weight')
        plt.title(title)
        plt.xlabel('Node j')
        plt.ylabel('Node i')
        plt.show()

    def plot_final_state(self, sol):
        """Plots the spatial configuration of opinions at the final time"""
        u_final = sol.ys[-1]

        x_plot = np.array(self.x)
        u_final = np.array(sol.ys[-1])
        O_ext_plot = np.array(self.O_ext)

        if O_ext_plot.ndim == 0:
            O_ext_plot = np.full_like(x_plot, O_ext_plot)

        
        plt.figure(figsize=(10, 6))
        
        # plot individuals
        plt.scatter(self.x, u_final, c='blue', alpha=0.6, label='Individual Opinions $u^*$')
        
        # plot external forces target
        plt.plot(self.x, self.O_ext, 'r--', label='Media Target $O_{ext}$')
        
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Social Space Position ($x$)')
        plt.ylabel('Opinion Space ($u$)')
        plt.title('Population Steady State')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_time_evolution(self, sol):
        """Plots trajectories of opinions over time"""
        ys = sol.ys # [T, N]
        ts = sol.ts
        
        plt.figure(figsize=(10, 6))
        step = max(1, self.n // 50) # avoid clutter for large N
        plt.plot(ts, ys[:, ::step], alpha=0.5)
        
        plt.xlabel('Time')
        plt.ylabel('Opinion')
        plt.title(f'Opinion Evolution (N={self.n}, alpha={self.alpha})')
        plt.grid(True, alpha=0.3)
        plt.show()