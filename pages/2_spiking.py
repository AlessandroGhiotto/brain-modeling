import streamlit as st


def main():
    st.title("Spiking", anchor=False)


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
