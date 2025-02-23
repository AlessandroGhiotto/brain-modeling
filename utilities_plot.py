import numpy as np
import plotly.graph_objects as go


def plot_tensor(y_data, ylabel: str = None, key=None):

    if key == "input":
        y_data = np.round(y_data, 4)

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
        height=300,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=30,  # top margin
            b=20,  # bottom margin
        ),
    )

    return fig
