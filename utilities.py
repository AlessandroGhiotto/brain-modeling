import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

sns.set_theme(style="darkgrid")

PARAMS = {
    "RS": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},
    "IB": {"a": 0.02, "b": 0.2, "c": -55, "d": 4},
    "CH": {"a": 0.02, "b": 0.2, "c": -50, "d": 2},
    "FS": {"a": 0.1, "b": 0.2, "c": -65, "d": 2},
    "LTS": {"a": 0.02, "b": 0.25, "c": -65, "d": 2},
    "TC": {"a": 0.02, "b": 0.25, "c": -65, "d": 0.05},
    "RZ": {"a": 0.1, "b": 0.26, "c": -65, "d": 2},
    "IT": {"a": 0.02, "b": -0.1, "c": -55, "d": 6},
}


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
    V_vals = np.linspace(-85, 35, 25)
    u_vals = np.linspace(-20, 15, 25)

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


def plot_tensor(y_data, ylabel: str = None, key=None):
    """Plot a single tensor. Used for plotting the current or the state variables."""

    x_data = np.linspace(0, 1000, len(y_data))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
        )
    )

    if ylabel:
        fig.update_layout(
            yaxis_title=ylabel,
        )

    if key == "error":
        fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis_title="Time (ms)",
        font=dict(family="Arial", size=12),
        width=400,
        height=250,
        margin=dict(
            l=30,  # left margin
            r=30,  # right margin
            t=20,  # top margin
            b=20,  # bottom margin
        ),
    )

    return fig


def plot_two_tensors(
    y_data1,
    y_data2,
    label1: str = "Tensor 1",
    label2: str = "Tensor 2",
    ylabel: str = None,
    key=None,
):
    """Plot two tensors. Used for plotting the the two state variables together."""
    x_data = np.linspace(0, len(y_data1), len(y_data1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data1,
            mode="lines",
            line=dict(color="#9649cb"),
            name=label1,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data2,
            mode="lines",
            line=dict(color="#f35b04"),
            name=label2,
        )
    )

    if ylabel:
        fig.update_layout(yaxis_title=ylabel)

    if key == "error":
        fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis_title="Time (ms)",
        font=dict(family="Arial", size=12),
        width=400,
        height=400,
        margin=dict(l=30, r=35, t=75, b=10),
        legend=dict(x=0.5, y=1.0, xanchor="center", yanchor="bottom", orientation="h"),
    )

    return fig


def plot_phase_plane(
    V_range,
    V_null,
    u_null,
    V_grid,
    u_grid,
    dV_grid,
    du_grid,
    V,
    w,
    critical_points=None,
):
    """Plot the phase plane of the Izhikevich model."""

    fig = go.Figure()

    # Plot nullclines
    fig.add_trace(
        go.Scatter(
            x=V_range,
            y=V_null,
            mode="lines",
            name="V nullcline",
            line=dict(color="#9649cb", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=V_range,
            y=u_null,
            mode="lines",
            name="w nullcline",
            line=dict(color="#f35b04", width=3),
        )
    )

    # Vector field (quiver plot)
    fig_quiver = ff.create_quiver(
        V_grid,
        u_grid,
        dV_grid,
        du_grid,
        scale=1.3,  # Adjusts vector length
        arrow_scale=0.3,  # Size of arrowheads
        name="Vector Field",
        line_width=1,
    )
    for trace in fig_quiver.data:
        fig.add_trace(trace)  # Add quiver traces to figure

    # Neuron trajectory
    fig.add_trace(
        go.Scatter(
            x=V,
            y=w,
            mode="lines",
            name="Trajectory",
            line=dict(color="#06d6a0", width=1),
        )
    )

    # Critical points
    if critical_points:
        cp_x, cp_y = zip(*critical_points)  # Unzip the list of tuples
        fig.add_trace(
            go.Scatter(
                x=cp_x,
                y=cp_y,
                mode="markers",
                marker=dict(color="red", size=8, symbol="star"),
                name="Critical points",
            )
        )

    # Starting position
    fig.add_trace(
        go.Scatter(
            x=[V[0]],
            y=[w[0]],
            mode="markers",
            marker=dict(color="lightblue", size=10),
            name="Start",
        )
    )

    # Layout adjustments
    fig.update_layout(
        title=dict(
            text="Phase Plane",
            x=0.05,  # move the title
            font=dict(size=20),  # Increases font size
        ),
        xaxis_title="V (Membrane Potential mV)",
        yaxis_title="w (Recovery Variable)",
        xaxis=dict(range=[-85, 40], showgrid=False, zeroline=False),
        yaxis=dict(range=[-20, 15], showgrid=False, zeroline=False),
        legend=dict(x=0.8, y=1),
        font=dict(family="Arial", size=12),
        width=500,
        height=700,
    )

    return fig
