import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import diffrax

"""
This file contains the implementation of the graph and numerical simulation of a SIS model.
Note that this is not discussed in the final paper, but was implemented for exploratory purpuses.
"""

class SISGraph():
    def __init__(self, n:int=200, beta:float=2.0, gamma:float=1.0, u0=None):
        self.n = n
        self.beta = beta    # infection rate
        self.gamma = gamma  # recovery rate
        
        if u0 is not None:
            self.u = u0
        else:
            # Start with random infections
            self.u = jr.uniform(jr.PRNGKey(42), (n,)) * 0.1
            
        self.x = None
        self.D = None
        self.A = None
        
    def positions(self):
        """Position nodes on a circle"""
        self.x = 2 * jnp.pi * jnp.arange(self.n) / self.n
        
    def distances(self):
        """Compute circular distances"""
        D_row = self.x[:, None]
        D_col = self.x[None, :]
        D_total = jnp.abs(D_row - D_col)
        self.D = jnp.minimum(D_total, 2*jnp.pi - D_total)
        
    def R(self, d:float, k:float, lam:float):
        # Exponential coupling graphon
        local_coupling = k * jnp.exp(-d/lam)
        return local_coupling
    
    def distance_based_graphon(self, k=1.0, lam=0.5):
        """Build contact network"""
        self.A = self.R(self.D, k, lam)
        return self.A
    
    def random_graphon(self, 
                       key:jr.PRNGKey, 
                       p:float=0.1, 
                       weight_scale:float=1.0, 
                       sigma:float=0.1, 
                       mu:float=0.5):
        """
        Build a random network adjacency matrix.

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
        
        # Generate upper triangular random edges
        upper_tri = jr.normal(key, shape=(n, n)) * sigma + mu
        upper_tri = jnp.clip(upper_tri, 0.0, 1.0) * weight_scale

        # Zero diagonal
        upper_tri = jnp.triu(upper_tri, k=1)
        
        # Make symmetric adjacency (undirected network)
        A = jnp.triu(upper_tri, k=1)
        A = A + A.T
        
        self.A = A
        return self.A

        
    def simulate(self, T:float=20.0, dt:float=0.01):
        def dynamics(t, u, args):
            """
            Network SIS dynamics
            du_i/dt = beta(1-u_i)Sum_j A_ij u_j - gamma u_i
            """
            dx = 2 * jnp.pi / self.n
            infection_rate = self.beta * (1 - u) * (self.A @ u) * dx
            recovery_rate = self.gamma * u
            return infection_rate - recovery_rate
        
        term = diffrax.ODETerm(dynamics)
        solver = diffrax.Euler()
        
        ts = jnp.linspace(0, T, int(T/dt) + 1)
        sol = diffrax.diffeqsolve(
            term, solver, t0=0, t1=T, dt0=dt,
            y0=self.u, saveat=diffrax.SaveAt(ts=ts)
        )
        return sol
    
    def plot_graph(self):
        # plots a visualization of the discrete graph 
        x = self.x
        A=self.A
        N = len(x)
        pos = jnp.stack([jnp.cos(x), jnp.sin(x)], axis=1)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')

        for i in range(N):
            for j in range(i+1, N):
                weight = A[i,j]
                if weight > 1e-3:
                    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                            color='blue', alpha=min(1, float(weight)), linewidth=1)

        ax.scatter(pos[:,0], pos[:,1], color='red', s=50)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("SIS Graph")
        plt.show()

    def plot_graphon(self, A):
        # plots the graphon gray scale picture
        plt.figure(figsize=(6,6))
        plt.imshow(A, cmap='gray', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(label='Coupling weight')
        plt.title("Graphon adjacency matrix")
        plt.xlabel('j')
        plt.ylabel('i')
        plt.show()

