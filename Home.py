import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="As-The-Crow-Flies Navigation",
    page_icon="🧭",
)

def main():
    # Header and Introduction
    st.title("🧭 As-The-Crow-Flies Navigation")
    st.image('header_img.png', caption='Created with DALLE 3')  # Assuming you have a header image
    st.write("""
    Welcome to our demo on As-The-Crow-Flies (ATCF) navigation for cyclists. This alternative approach to turn-by-turn navigation provides a unique experience by offering the beeline to the destination. Dive into our two detailed sections to understand the simulation and the underlying city graph statistics.
    """)

    # Brief Introduction to the Paper
    st.subheader("Research Abstract")
    st.write("""
    ATCF navigation utilizes the least-angle strategy, offering a unique beeline route to the destination. While it presents an exciting alternative for cyclists, it's not without its challenges, such as occasionally running into dead ends. Our research dives deep into understanding the relationship between street network attributes and the user experience of this navigation method. Through extensive analysis across 1633 cities, we uncover the key characteristics of the ideal city for ATCF and present design implications for its future implementations.
    """)

    # Link to the Full Paper and Embedding
    st.subheader("Full Research Publication")
    st.write("""
    Interested in the comprehensive details, analyses, and findings of our research? You can access the full publication below:
    """)

    st.write("""
    Gian-Luca Savino, Ankit Kariryaa, and Johannes Schöning. 2022. Free as a Bird, but at What Cost? The Impact of Street Networks on the User Experience of As-The-Crow-Flies Navigation for Cyclists. Proc. ACM Hum.-Comput. Interact. 6, MHCI, Article 209 (September 2022), 24 pages. [https://doi.org/10.1145/3546744](https://doi.org/10.1145/3546744)
    """)

if __name__ == "__main__":
    main()
