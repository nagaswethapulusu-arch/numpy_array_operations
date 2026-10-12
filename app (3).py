"""
Streamlit app - NumPy Array Operations (Task 12)
Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NumPy Array Operations", page_icon="🔢", layout="centered")

st.title("🔢 NumPy Array Operations")
st.caption("Task 12 · Level 1 · AI & ML Track — vectorized addition, subtraction, multiplication and division")


def parse_array(text: str) -> np.ndarray:
    """Convert '1, 2, 3' into np.array([1., 2., 3.])."""
    values = [v.strip() for v in text.replace("\n", ",").split(",") if v.strip()]
    return np.array([float(v) for v in values])


col1, col2 = st.columns(2)
with col1:
    text_a = st.text_input("Array A (comma separated)", "10, 20, 30, 40, 50")
with col2:
    text_b = st.text_input("Array B (comma separated)", "2, 4, 5, 8, 10")

if st.button("Calculate", type="primary"):
    try:
        a = parse_array(text_a)
        b = parse_array(text_b)
    except ValueError:
        st.error("Please enter only numbers separated by commas.")
        st.stop()

    if a.size == 0 or b.size == 0:
        st.error("Both arrays need at least one number.")
        st.stop()

    st.subheader("Arrays and shapes")
    st.code(f"a = {a}\nb = {b}\n\na.shape = {a.shape}\nb.shape = {b.shape}")

    if a.shape != b.shape:
        st.error(
            f"Shapes {a.shape} and {b.shape} do not match, so element-wise "
            "operations are not possible. Make both arrays the same length."
        )
        st.stop()

    with np.errstate(divide="ignore", invalid="ignore"):
        results = {
            "Addition (a + b)": a + b,
            "Subtraction (a - b)": a - b,
            "Multiplication (a * b)": a * b,
            "Division (a / b)": a / b,
        }

    if np.any(b == 0):
        st.warning("Array B contains 0, so division gives inf/nan at those positions.")

    st.subheader("Results")
    df = pd.DataFrame({"a": a, "b": b, **results})
    st.dataframe(df, use_container_width=True)

    st.subheader("Chart")
    st.bar_chart(pd.DataFrame(results).replace([np.inf, -np.inf], np.nan))

with st.expander("What is vectorization?"):
    st.write(
        "Vectorization means applying an operation to a whole array at once "
        "instead of writing a Python `for` loop. `a + b` adds `a[i] + b[i]` for "
        "every index in compiled C code, which is shorter and much faster than "
        "looping over Python lists."
    )
