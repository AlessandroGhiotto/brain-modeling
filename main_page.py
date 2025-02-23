import streamlit as st


def main():
    st.title("Brain Modeling", anchor=False)
    st.write("Welcome to the Brain Modeling page!")
    st.markdown(
        r"""The [**Izhikevich model**](https://www.izhikevich.org/publications/spikes.pdf) is a computationally efficient model of neuron dynamics that captures a wide range of spiking and bursting behaviors observed in biological neurons. It is defined by two differential equations: """
    )

    st.latex(
        r"""
        \begin{cases}
        \dot{v} = 0.04v^2 + 5v + 140 - u + I \\
        \dot{u} = a(bv - u)
        \end{cases}"""
    )

    st.markdown(
        """where:  
        - v represents the membrane potential,  
        - u is a recovery variable,  
        - I is the external input current,  
        - a, b, c, d are parameters that define different neuron types.  
    When v reaches a threshold (e.g., 30 mV), it is reset:"""
    )

    st.latex(r"""v \leftarrow c, \quad u \leftarrow u + d """)

    st.markdown(
        r"""Here we have some of the behaviour that can be expressed by the Izhikevich model:

        ![image.png](code/izhikevich.png)"""
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Brain Modeling",
        page_icon=":brain:",
        menu_items={
            "Report a bug": "https://github.com/AlessandroGhiotto/brain-modeling/issues",
        },
    )
    main()
