import numpy as np
import streamlit as st
from utilities import *


def main():
    st.title("Thalamo-Cortical (TC)", anchor=False)

    st.markdown(
        """TC neurons have two firing regimes: When
        at rest (v is around 60 mV) and then depolarized, they exhibit tonic firing. However, if a 
        negative current step is delivered so that the membrane potential is
        hyperpolarized (v is around 90 mV), the neurons fire a rebound
        burst of action potentials."""
    )

    st.latex(r"""a = 0.02, \;\; b = 0.25, \;\; c = -65, \;\; d = 0.05""")

    T = 1000
    dt = 1
    time_series = np.arange(0, T, dt)

    a, b, c, d = PARAMS["TC"].values()

    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(
            """**1. Tonic Firing**: the neuron is at rest 
            (v is around -60 mV) and then is depolarized."""
        )

        current = st.slider(
            "Input Current magnitude (mA)",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )

        I_ = np.zeros(len(time_series))
        I_[200:800] = current
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

    with cols[1]:
        st.markdown(
            """**2. Rebound Burst**: a negative current step is 
            delivered so that the membrane potential is hyperpolarized"""
        )

        current = st.slider(
            "Input Current magnitude (mA)",
            min_value=-30,
            max_value=0,
            value=-20,
            step=1,
        )

        I_ = np.zeros(len(time_series))
        I_[:200] = current
        time, V, w = izhikevic_model(a, b, c, d, dt, T, I_ext=I_)

        st.plotly_chart(
            plot_tensor(I_, "Current (mA)", key="input"),
            key="input_1",
        )
        st.plotly_chart(
            plot_tensor(V, "V - Membrane Potential (mV)", key="voltage"),
            key="voltage_1",
        )
        st.plotly_chart(
            plot_tensor(w, "w - Recovery Variable", key="recovery"),
            key="recovery_1",
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
        value=2,
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
