import streamlit as st


def main():
    st.title("Brain Modeling - Project", anchor=False)
    st.markdown(
        """Author: &nbsp; Alessandro Ghiotto &nbsp;
        [![Personal Profile](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/AlessandroGhiotto) 
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Izhikevich Model", anchor=False)
    st.markdown(
        r"""The [**Izhikevich model**](https://www.izhikevich.org/publications/spikes.pdf) is a computationally efficient model of neuron dynamics that captures a wide range of spiking and bursting behaviors observed in biological neurons. It is defined by two differential equations: """
    )

    st.latex(
        r"""
        \begin{cases}
        \dot{v} = 0.04v^2 + 5v + 140 - w + I \\
        \dot{w} = a(bv - w)
        \end{cases}"""
    )
    st.markdown("""With the auxiliary after-spike reset condition:""")

    st.latex(
        r"""
        \text{if } v \geq 30 \text{ mV, then}
        \begin{cases}
        v \leftarrow c, \\
        w \leftarrow w + d
        \end{cases}"""
    )

    st.markdown(
        """where:  
        - v represents the membrane potential,  
        - w is a recovery variable,  
        - I is the external input current,  
        - a, b, c, d are parameters that define different neuron types."""
    )

    st.markdown(
        "Here we have some of the behaviour that can be expressed by the Izhikevich model:"
    )
    st.image(
        "https://github.com/AlessandroGhiotto/brain-modeling/blob/main/images/izhikevich.png?raw=true",
        caption="Credits [Eugene M. Izhikevich](https://www.izhikevich.org/publications/spikes.htm)",
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
