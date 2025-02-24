import numpy as np
import streamlit as st
from utilities import *


def main():
    st.title("Integrator", anchor=False)

    st.markdown(
        """The Integrator neuron doesn't fire immidieately when a current is supplied,
        but it integrates the input over time and fires a spike when the membrane potential
        reaches a certain threshold.
        """
    )

    st.latex(r"""a = 0.02, \;\; b = -0.1, \;\; c = -55, \;\; d = 6""")

    coefficient = st.slider(
        "Angular Coefficient of the Input Current",
        min_value=0.00,
        max_value=0.1,
        value=0.03,
        step=0.01,
    )

    T = 1000
    dt = 1
    time_series = np.arange(0, T, dt)
    I_ = coefficient * time_series

    a, b, c, d = PARAMS["IT"].values()
    time, V, w = izhikevic_model(a, b, c, d, dt, T, I_ext=I_)

    st.plotly_chart(
        plot_tensor(I_, "Current (mA)", key="input"),
        key="input_0",
    )

    st.plotly_chart(
        plot_tensor(V, "V - Membrane Potential (mV)", key="voltage"),
        key="voltage_0",
    )

    st.plotly_chart(
        plot_tensor(w, "w - Recovery Variable", key="recovery"),
        key="recovery_0",
    )

    ### Second part

    st.markdown(
        """
        ---
        Plot of the phase plane for a constant input current.
        """
    )

    current = st.slider(
        "Constant Input Current (mA)",
        min_value=0,
        max_value=30,
        value=10,
        step=1,
    )

    time, V, w = izhikevic_model(a, b, c, d, dt=1, T=300, I_ext=current)

    V_range, V_null, u_null = compute_nullclines(a, b, current)
    V_grid, u_grid, dV_grid, du_grid = compute_vector_field(a, b, current)
    critical_points = find_critical_point(a, b, current)

    cols = st.columns([1, 3])

    with cols[0]:
        st.plotly_chart(
            plot_two_tensors(V, w, "Membrane Potential V", "Recovery Variable w"),
            key="phase_plane_0",
        )

    with cols[1]:
        st.plotly_chart(
            plot_phase_plane(
                V_range,
                V_null,
                u_null,
                V_grid,
                u_grid,
                dV_grid,
                du_grid,
                V,
                w,
                critical_points,
            ),
            key="phase_plane_1",
        )


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="Brain Modeling",
        page_icon=":brain:",
        menu_items={
            "Report a bug": "https://github.com/AlessandroGhiotto/brain-modeling/issues",
        },
    )
    main()
