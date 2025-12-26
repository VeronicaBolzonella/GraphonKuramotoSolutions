import matplotlib.pyplot as plt
from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax

"""
This file contains the implementation of the graph and numerical simulation of a Kuramoto model.
Note that this is not discussed in the final paper, but was implemented for exploratory purpuses.
"""

class KuramotoGraph():
    def __init__(self, n:int=200, u0=None):
        if n < 3:
            raise ValueError("N must be at least 3 for a ring.")
        self.n = n
        if u0 is not None:
            self.u = u0
        else: 
            self.u = jr.uniform(jax.random.PRNGKey(42), shape=(self.n,))
        self.x = None # angular positions
        self.D = None # distances
        self.A = None # Aij = R(Dij) adjacency matrix

    def positions(self):
        n = jnp.arange(self.n)
        self.x = 2*jnp.pi*n / self.n

    def distances(self):
        # D [a x a]
        D_row = self.x[:, None]
        D_col = self.x[None, :]
        D_total = jnp.abs(D_row - D_col)
        self.D = jnp.minimum(D_total, 2*jnp.pi-D_total)

    def R(self, d:float, k:float, lam:float, alpha=0.0):
        # graphon with exponential coupling strength
        local_coupling = k * jnp.exp(-d/lam)
        global_coupling = alpha * k * (1 - jnp.exp(-d/lam))
        return local_coupling + global_coupling
    

    def distance_based_graphon(self, k:float=1.0, lam:float=0.5):
        """Builds network based on distance around ring"""
        self.A = self.R(self.D, k, lam)
        return self.A
    
    def random_graphon(self, 
                       key:jr.PRNGKey, 
                       p:float=0.1, 
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


    def update_phase(self, dt:float=0.01):
        phase_diffs = self.u[None, :] - self.u[:, None]
        S = jnp.sin(2 * jnp.pi * phase_diffs)
        du_dt = (self.A * S).sum(axis=1) / self.n
        self.u = jnp.mod(self.u + dt * du_dt, 1.0)

    def kuramoto_motion(self,
            N:Optional[int]=None,
            T:float=50.0, 
            dt:float=1e-2,
            u0:float=0.0) -> diffrax.Solution:
        dtype = jnp.float32

        u0 = jnp.atleast_1d(u0).astype(dtype)

        def f(t, u, args):
            phase_diffs = u[None, :] - u[:, None]
            S = jnp.sin(2 * jnp.pi * phase_diffs)
            dudt = (self.A * S).sum(axis=1) / self.n
            return dudt   

        if N is None:
            N = jnp.floor(T/dt).astype(int)
        
        t0 = jnp.asarray(0.0, dtype)
        t1 = jnp.asarray(T, dtype)
        ts = jnp.linspace(t0, t1, N + 1, dtype=dtype)
        dt0 = jnp.asarray(dt, dtype)

        control_term = diffrax.ODETerm(f)

        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            terms=control_term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=u0,
            args=None,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol
    
    def simulate(self,
                N:Optional[int]=None,
                T:float = 50.0, 
                dt:float = 1e-2):
        return self.kuramoto_motion(N=N, T=T, dt=dt, u0=self.u)
    
    def m_twisted_state(self, m=1):
        """Create an m-twisted state initial condition"""
        x = 2*jnp.pi*jnp.arange(self.n) / self.n
        return (m * x / (2*jnp.pi)) % 1.0


    def plot_graph(self, x, A):
        N = self.n

        title=f"Kuramoto Graph, N={self.n}"
        pos = jnp.stack([jnp.cos(x), jnp.sin(x)], axis=1)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')

        for i in range(N):
            for j in range(i+1, N):
                weight = A[i,j]
                if weight > 1e-3:
                    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                            color='blue', alpha=min(1, float(weight)+0.1), linewidth=1)

        ax.scatter(pos[:,0], pos[:,1], color='red', s=50)

        ax.set_title(title)
        plt.show()

    def plot_graphon(self, A):
        plt.figure(figsize=(6,6))
        plt.imshow(A, cmap='gray', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(label='Coupling weight')
        plt.title("Graphon adjacency matrix")
        plt.xlabel('j')
        plt.ylabel('i')
        plt.show()



