import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import diffrax

class SISGraph():
    def __init__(self, n, beta=2.0, gamma=1.0, u0=None):
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
        
    def R(self, d, k, lam):
        local_coupling = k * jnp.exp(-d/lam)
        return local_coupling
    
    def distance_based_graphon(self, k=1.0, lam=0.5):
        """Build contact network"""
        self.A = self.R(self.D, k, lam)
        return self.A
    
    def random_graphon(self, key, p=0.1, weight_scale=1.0, sigma=0.1, mu=0.5):
        """
        Build a random network adjacency matrix.
        
        Parameters
        ----------
        p : float
            Probability of an edge between any two nodes.
        weight_scale : float
            Maximum weight for edges (random uniform in [0, weight_scale]).
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

        
    def simulate(self, T=20.0, dt=0.1):
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
    
    def plot_graph(self, title="Kuramoto Graph"):
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
        ax.set_title(title)
        plt.show()

    def plot_graphon(self, title="Graphon adjacency matrix"):
        plt.figure(figsize=(6,6))
        plt.imshow(1-self.A, cmap='gray', origin='lower')
        plt.colorbar(label='Coupling weight')
        plt.title(title)
        plt.xlabel('j')
        plt.ylabel('i')
        plt.show()
