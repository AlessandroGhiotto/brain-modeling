import numpy as np
import streamlit as st
from utilities import *


def main():
    st.title("Fast Spiking (FS)", anchor=False)

    st.markdown(
        """FS neurons can fire periodic trains of action potentials 
        with extremely high frequency practically without any adaptation 
        (slowing down). In the model,
        this corresponds to a = 0.1 (fast recovery)."""
    )

    st.latex(r"""a = 0.1, \;\; b = 0.2, \;\; c = -65, \;\; d = 2""")

    current = st.slider(
        "Input Current magnitude (mA)",
        min_value=0,
        max_value=30,
        value=10,
        step=1,
    )

    T = 1000
    dt = 1
    time_series = np.arange(0, T, dt)
    I_ = np.zeros(len(time_series))
    I_[200:800] = current

    a, b, c, d = PARAMS["FS"].values()
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
