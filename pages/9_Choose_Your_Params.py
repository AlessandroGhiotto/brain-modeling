import numpy as np
import streamlit as st
import random
from utilities import *


def main():
    st.title("Custom Model", anchor=False)

    st.markdown(
        """Welcome to the parameter selection page! Here, you can customize the values of 
        a, b, c, and d to explore their impact on the model's dynamics. Adjusting these 
        parameters allows you to analyze how different settings influence the system's behavior, 
        making it a valuable tool for studying its evolution and responses. You can pick them
        in the sidebar on the left.
        """
    )

    ### PARAMETERS
    # defaults
    A, B, C, D = 0.02, 0.2, -65, 8.0
    CURRENT = 10

    if "slider_version" not in st.session_state:
        st.session_state["slider_version"] = 1

    def reset_sliders():
        st.session_state["slider_version"] = +random.randint(1, 100)

    a = st.sidebar.slider(
        "a",
        min_value=0.00,
        max_value=0.2,
        value=A,
        step=0.01,
        key=f"slider_{0+st.session_state['slider_version']}",
    )
    b = st.sidebar.slider(
        "b",
        min_value=0.1,
        max_value=0.3,
        value=B,
        step=0.01,
        key=f"slider_{1+st.session_state['slider_version']}",
    )
    c = st.sidebar.slider(
        "c",
        min_value=-80,
        max_value=-30,
        value=C,
        step=1,
        key=f"slider_{2+st.session_state['slider_version']}",
    )
    d = st.sidebar.slider(
        "d",
        min_value=0.0,
        max_value=10.0,
        value=D,
        step=0.1,
        key=f"slider_{3+st.session_state['slider_version']}",
    )

    # refresh the sliders
    st.sidebar.button("Reset Params", on_click=reset_sliders)

    current = st.slider(
        "Input Current magnitude (mA)",
        min_value=0,
        max_value=30,
        value=CURRENT,
        step=1,
    )

    T = 1000
    dt = 1
    time_series = np.arange(0, T, dt)
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
