import numpy as np
import streamlit as st
from utilities_plot import plot_tensor


def main():
    st.title("Bursting Behavior in the Izhikevich Model", anchor=False)

    T = 1000  # simulation len
    dt = 0.5  # time step for the integration
    time_series = np.arange(0, T, dt)
    I = 10  # current step amplitude
    I_ = np.zeros(len(time_series))
    I_[200:1500] = I  # we stop at 1.5s

    cols = st.columns([1, 3])

    with cols[0]:
        st.plotly_chart(
            plot_tensor(I_, "Current (mA)", key="input"),
            key="input_0",
        )

    with cols[1]:
        st.plotly_chart(
            plot_tensor(I_, "Current (mA)", key="input"),
            key="input_1",
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
