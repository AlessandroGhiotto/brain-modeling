import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import seaborn as sns

sns.set_theme(style="darkgrid")


def izhikevic_model(a, b, c, d, dt, T, I_ext, v0=-70, w0=-15):
    """Simulates the Izhikevich neuron model."""

    time = np.arange(0, T, dt)
    max_value = 30  # Maximum spike value

    V = np.zeros(len(time))
    V[0] = v0  # Initial membrane potential (mV)
    w = np.zeros(len(time))
    w[0] = w0  # Initial recovery variable

    if isinstance(I_ext, (int, float)):
        I = np.full(
            len(time), I_ext
        )  # Constant input current (used for the phase plane)
    else:
        I = I_ext  # Time-varying input current

    for t in range(1, len(time)):
        # check that the membrane potential is not higher than the maximum value
        if V[t - 1] < max_value:
            # membran potential evolution
            # calculate the increment
            dV = (0.04 * V[t - 1] + 5) * V[t - 1] + 140 - w[t - 1]
            # update the membrane potential (update the state variable)
            V[t] = V[t - 1] + (dV + I[t - 1]) * dt

            # recovery variable evolution
            dw = a * (b * V[t - 1] - w[t - 1])
            w[t] = w[t - 1] + dt * dw

        # everytime V[t-1]>max_value we have a spike
        else:
            # peak reached
            V[t - 1] = max_value
            # peak reached
            V[t] = c
            # reset recovery variable
            w[t] = w[t - 1] + d

    return time, V, w


def compute_nullclines(a, b, I):
    """Compute nullclines for V and w."""
    V_range = np.linspace(-85, 35, 100)

    # V-nullcline: dV/dt = 0 → I = w - (0.04*V^2 + 5V + 140)
    V_null = 0.04 * V_range**2 + 5 * V_range + 140 + I

    # w-nullcline: dw/dt = 0 → w = bV
    u_null = b * V_range

    return V_range, V_null, u_null


def compute_vector_field(a, b, I):
    """Compute the vector field in phase space."""
    V_vals = np.linspace(-85, 35, 20)
    u_vals = np.linspace(-20, 15, 20)

    V, w = np.meshgrid(V_vals, u_vals)

    dV = (0.04 * V**2 + 5 * V + 140 - w) + I
    dw = a * (b * V - w)

    magnitude = np.sqrt(dV**2 + dw**2)
    dV_norm = dV / (magnitude + 1e-6)
    du_norm = dw / (magnitude + 1e-6)

    return V, w, dV_norm, du_norm


def find_critical_point(a, b, I):
    """Solve system equations to find the critical point in phase space."""
    V_sym, u_sym = sp.symbols("V w")

    eq1 = 0.04 * V_sym**2 + 5 * V_sym + 140 - u_sym + I  # dV/dt = 0
    eq2 = a * (b * V_sym - u_sym)  # dw/dt = 0

    c_p = sp.solve((eq1, eq2), (V_sym, u_sym))

    # Filter only real solutions
    real_solutions = [
        (float(sol[0]), float(sol[1]))
        for sol in c_p
        if sol[0].is_real and sol[1].is_real
    ]

    if real_solutions:
        return real_solutions
    return None


def plot_time_evolution(a, b, c, d, dt, T, I_ext, v0=-70):
    """Plots the time evolution of the Izhikevich neuron model."""

    time, V, w = izhikevic_model(a, b, c, d, dt, T, I_ext=I_ext, v0=v0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # plot the stimulus
    axes[0].plot(time, I_ext)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Current (pA)")
    axes[0].set_title("Input Current")

    # plotting the model output
    axes[1].plot(time, V)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Membrane Potential (mV)")
    axes[1].set_title("Izhikevic model")

    plt.tight_layout()
    plt.show()


def plot_phase_plane(I_ext, a, b, c, d, v0=-70, w0=-15):
    """Plots the phase plane, nullclines, vector field, and voltage trace."""
    dt = 0.1
    T = 200

    time, V, w = izhikevic_model(a, b, c, d, dt, T, I_ext, v0, w0)

    V_range, V_null, u_null = compute_nullclines(a, b, I_ext)
    V_grid, u_grid, dV_grid, du_grid = compute_vector_field(a, b, I_ext)
    critical_points = find_critical_point(a, b, I_ext)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Voltage over time
    axes[0].plot(time, V, label="V(t)")
    axes[0].plot(time, w, label="w(t)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("State variable")
    axes[0].set_title("Time Evolution")
    axes[0].legend()

    # Phase plane: V vs. w
    axes[1].plot(V_range, V_null, label=r"$\dot v = 0$", color="blue")
    axes[1].plot(V_range, u_null, label=r"$\dot w = 0$", color="green")

    # Plot vector field
    axes[1].quiver(V_grid, u_grid, dV_grid, du_grid, color="gray", alpha=0.5, scale=30)

    # Plot neuron trajectory
    axes[1].plot(V, w, color="red", label="trajectory")

    # Mark critical point
    if critical_points:
        for cp in critical_points:
            cp_x, cp_y = cp
            axes[1].plot(cp_x, cp_y, "r*", markersize=8)

    # starting position
    axes[1].plot(V[0], w[0], "bo", label="start", markersize=8)

    axes[1].set_xlabel("V (Membrane Potential mV)")
    axes[1].set_ylabel("w (Recovery Variable)")
    axes[1].set_title("Phase Plane")
    axes[1].legend(loc="upper right")
    axes[1].grid()
    axes[1].set_xlim(-85, 35)
    axes[1].set_ylim(-20, 15)

    plt.show()
