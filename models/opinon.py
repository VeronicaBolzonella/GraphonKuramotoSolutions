import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import diffrax
import numpy as np

"""
This file contains the implementation of the graph and numerical simulation of the Opinion Dynamics Models discussed in 
the final paper. Visualizations of the results can be found in the opinion_visualization.ipynb notebook.
"""


class OpinionGraph():
    def __init__(self, n:int=200, alpha:float=5.0, gamma:float=1.0, u0=None):
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
        
    def R(self, d:float, k:float, lam:float):
        """Exponential graphon"""
        local_coupling = k * jnp.exp(-d/lam)
        return local_coupling
    
    def distance_based_graphon(self, k:float=1.0, lam:float=0.1):
        """Build contact network based on distance"""
        self.A = self.R(self.D, k, lam)
        return self.A
    
    def random_graphon(self, 
                       key:jr.PRNGKey, 
                       weight_scale:float=1.0, 
                       sigma:float=0.1, 
                       mu:float=0.5):
        """
        Builds a random network adjacency matrix.

        Args:
            key (jr.PRNGkey): seed for reproducibility
            p (float, optional): Probability of an edge between any two nodes. Defaults to 0.1.
            weight_scale (float, optional): Maximum weight for edges. Defaults to 1.0.
            sigma (float, optional): _description_. Defaults to 0.1.
            mu (float, optional): _description_. Defaults to 0.5.

        Returns:
            A (Array): [N, N] adjacency matrix
        """
        n = self.n
        # non symmetric adjacency matrix
        A = jr.normal(key, shape=(n, n)) * sigma + mu
        A = jnp.clip(A, 0.0, 1.0) * weight_scale
        
        # Zero diagonal (no self-loops)
        A = A.at[jnp.diag_indices(n)].set(0.0)
        
        self.A = A
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
        Simulate dynamics in during time T.

        Args:
            T (float, optional): end time. Defaults to 10.0.
            dt (float, optional): step size. Defaults to 0.01.
            phi_type (str, optional): interaction kernel type, only linear implemented. Defaults to 'linear'.
            epsilon (float, optional): _description_. Defaults to 0.2.
        """
        def dynamics(t, u, args):
            """
            Opinion Dynamics:
            du_i/dt = alpha  * Sum A_ij * phi(|u_j-u_i|) * (u_j - u_i) + Gamma_i * (O_ext_i - u_i)
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

    def plot_graphon(self):
        """Plots the graphon"""
        plt.figure(figsize=(10,10))
        plt.imshow(self.A, cmap='gray_r', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(label='Coupling weight')
        plt.title("Graphon")
        plt.xlabel('j')
        plt.ylabel('i')
        plt.show()

    def plot_adj(self):
        """Plots the adjacency matrix"""
        plt.figure(figsize=(10,10))
        plt.imshow(self.A, cmap='gray_r', origin='lower')
        plt.colorbar(label='Coupling weight')
        plt.title("Adjacency matrix")
        plt.xlabel('j')
        plt.ylabel('i')
        plt.show()

    def plot_final_state(self, sol):
        """Plots the opinions at the final time"""
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
        step = max(1, self.n // 50) # dont plot all if N very large
        plt.plot(ts, ys[:, ::step], alpha=0.5)
        
        plt.xlabel('Time')
        plt.ylabel('Opinion')
        plt.title(f'Opinion Evolution (N={self.n}, alpha={self.alpha})')
        plt.grid(True, alpha=0.3)
        plt.show()