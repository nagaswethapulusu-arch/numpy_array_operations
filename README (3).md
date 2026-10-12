# Task 12 – NumPy Array Operations

**Level 1 · Day 12 · AI & ML Track**

## Objective
Understand vectorized numerical operations using NumPy.

## Files
| File | Purpose |
|---|---|
| `numpy_array_operations.ipynb` | Jupyter Notebook version (main submission) |
| `numpy_array_operations.py` | Same code as a plain Python script |
| `README.md` | Results, explanation and interview answers |

## How to run
```bash
pip install numpy
python numpy_array_operations.py
# or open the notebook:
jupyter notebook numpy_array_operations.ipynb
```

## Deliverable 1 – Two NumPy arrays
```python
a = np.array([10, 20, 30, 40, 50])
b = np.array([2, 4, 5, 8, 10])
```
Both have shape `(5,)`, so element-wise operations work.

## Deliverable 2 – Results of operations
| Operation | Code | Result |
|---|---|---|
| Addition | `a + b` | `[12 24 35 48 60]` |
| Subtraction | `a - b` | `[ 8 16 25 32 40]` |
| Multiplication | `a * b` | `[ 20  80 150 320 500]` |
| Division | `a / b` | `[5. 5. 6. 5. 5.]` |

## Deliverable 3 – Short explanation of vectorized operations
A vectorized operation applies an operation to an entire array at once, without writing an explicit Python `for` loop. When you write `a + b`, NumPy adds each pair `a[i] + b[i]` internally, in compiled C code, over contiguous memory of a single data type. This makes the code shorter, more readable and much faster than looping over Python lists.

## Interview Questions

**1. What is vectorization?**
Vectorization means performing operations on whole arrays instead of element by element in Python loops. The looping happens in optimized, pre-compiled C code (often using CPU SIMD instructions), so you write `a + b` instead of a loop.

**2. What happens when two NumPy arrays are added?**
NumPy adds them element-wise: the result at index `i` is `a[i] + b[i]`, and a new array is returned. The arrays must have the same shape or be broadcastable (e.g. `(5,)` + a scalar, or `(3,1)` + `(1,4)`). If the shapes are incompatible, NumPy raises `ValueError: operands could not be broadcast together`.

**3. Why is NumPy faster than ordinary Python lists for numerical operations?**
- NumPy arrays store elements of one fixed type in a contiguous block of memory; Python lists store pointers to separate Python objects.
- The loops run in compiled C instead of the slow Python interpreter, avoiding per-element type checks and object overhead.
- Contiguous memory is cache-friendly and allows SIMD (vectorized CPU instructions).

In the bonus test for 1,000,000 elements, NumPy was several times faster than a list comprehension (the exact speedup varies by machine).

## Notes
- `a * b` is **element-wise** multiplication. For matrix multiplication use `a @ b` or `np.dot`.
- `/` always returns floats, even when the division is exact (`5.` not `5`).
- Dividing by zero in NumPy gives `inf` or `nan` with a warning instead of raising an error.
