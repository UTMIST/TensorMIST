#include <pybind11/pybind11.h>
#include <limits>
#include "ops.h"

namespace py = pybind11;

namespace {

static_assert(sizeof(float) == 4, "requires four-byte floats");

// internal API: Python validates storage, shapes and output ownership.
// holding each buffer export keeps its owner alive and prevents array resizing.
// this helper only obtains pointers; it does not validate the buffer metadata.
struct FloatBuffer {
    py::buffer_info info;

    explicit FloatBuffer(py::buffer buffer) : info(buffer.request()) {}

    const float *read() const { return static_cast<const float *>(info.ptr); }
    float *write() const { return static_cast<float *>(info.ptr); }
};

constexpr size_t max_elements =
    static_cast<size_t>(std::numeric_limits<py::ssize_t>::max()) / sizeof(float);

void check_native_count(size_t count) {
    if (count > max_elements) {
        throw py::value_error("element count exceeds the native float32 buffer limit");
    }
}

void check_native_matrix(size_t rows, size_t cols, bool int_counters = false) {
    // some existing kernels use int counters even though their API uses size_t.
    if (int_counters && (rows > static_cast<size_t>(std::numeric_limits<int>::max())
                         || cols > static_cast<size_t>(std::numeric_limits<int>::max()))) {
        throw py::value_error("dimensions exceed the kernel's int loop-counter limit");
    }
    check_native_count(rows);
    check_native_count(cols);
    if (cols != 0 && rows > max_elements / cols) {
        throw py::value_error("matrix size exceeds the native float32 buffer limit");
    }
}

}  // namespace

// entry point when importing tensormist._C. The GIL stays held for every call.
// all functions write into caller-owned output buffers and return None.
PYBIND11_MODULE(_C, m) {
    m.doc() = "internal CPU bindings; python callers must validate shapes and buffers";

    m.def("add", [](py::buffer a, py::buffer b, py::buffer out, size_t m, size_t n) {
        check_native_matrix(m, n, true);
        FloatBuffer av(a), bv(b), ov(out);
        tm_add(av.read(), bv.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("b"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("add_backward", [](py::buffer grad_out, py::buffer grad_a,
                              py::buffer grad_b, size_t m, size_t n) {
        check_native_matrix(m, n, true);
        FloatBuffer go(grad_out), ga(grad_a), gb(grad_b);
        tm_add_backward(go.read(), ga.write(), gb.write(), m, n);
    }, py::arg("grad_out"), py::arg("grad_a"), py::arg("grad_b"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("subtract", [](py::buffer a, py::buffer b, py::buffer out, size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer av(a), bv(b), ov(out);
        tm_subtract(av.read(), bv.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("b"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("subtract_backward", [](py::buffer grad_out, py::buffer grad_a,
                                   py::buffer grad_b, size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer go(grad_out), ga(grad_a), gb(grad_b);
        tm_subtract_backward(go.read(), ga.write(), gb.write(), m, n);
    }, py::arg("grad_out"), py::arg("grad_a"), py::arg("grad_b"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("multiply", [](py::buffer a, py::buffer b, py::buffer out, size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer av(a), bv(b), ov(out);
        tm_multiply(av.read(), bv.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("b"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("multiply_backward", [](py::buffer a, py::buffer b, py::buffer grad_out,
                                   py::buffer grad_a, py::buffer grad_b,
                                   size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer av(a), bv(b), go(grad_out), ga(grad_a), gb(grad_b);
        tm_multiply_backward(av.read(), bv.read(), go.read(), ga.write(), gb.write(), m, n);
    }, py::arg("a"), py::arg("b"), py::arg("grad_out"), py::arg("grad_a"),
       py::arg("grad_b"), py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("scalar_multiply", [](py::buffer a, float scalar, py::buffer out,
                                 size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer av(a), ov(out);
        tm_scalar_multiply(av.read(), scalar, ov.write(), m, n);
    }, py::arg("a"), py::arg("scalar"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("scalar_multiply_backward", [](py::buffer grad_out, float scalar,
                                          py::buffer grad_a, size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer go(grad_out), ga(grad_a);
        tm_scalar_multiply_backward(go.read(), scalar, ga.write(), m, n);
    }, py::arg("grad_out"), py::arg("scalar"), py::arg("grad_a"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("matmul", [](py::buffer a, py::buffer b, py::buffer out,
                        size_t m, size_t k, size_t n) {
        check_native_matrix(m, k);
        check_native_matrix(k, n);
        check_native_matrix(m, n);
        FloatBuffer av(a), bv(b), ov(out);
        tm_matmul(av.read(), bv.read(), ov.write(), m, k, n);
    }, py::arg("a"), py::arg("b"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("k").noconvert(), py::arg("n").noconvert());

    m.def("matmul_backward", [](py::buffer a, py::buffer b, py::buffer grad_out,
                                 py::buffer grad_a, py::buffer grad_b,
                                 size_t m, size_t k, size_t n) {
        check_native_matrix(m, k);
        check_native_matrix(k, n);
        check_native_matrix(m, n);
        FloatBuffer av(a), bv(b), go(grad_out), ga(grad_a), gb(grad_b);
        tm_matmul_backward(av.read(), bv.read(), go.read(), ga.write(), gb.write(), m, k, n);
    }, py::arg("a"), py::arg("b"), py::arg("grad_out"), py::arg("grad_a"),
       py::arg("grad_b"), py::arg("m").noconvert(), py::arg("k").noconvert(),
       py::arg("n").noconvert());

    m.def("transpose", [](py::buffer a, py::buffer out, size_t rows, size_t cols) {
        check_native_matrix(rows, cols);
        FloatBuffer av(a), ov(out);
        tm_transpose(av.read(), ov.write(), rows, cols);
    }, py::arg("a"), py::arg("out"),
       py::arg("rows").noconvert(), py::arg("cols").noconvert());

    // rows/cols describe the original input; grad_out has shape (cols, rows).
    m.def("transpose_backward", [](py::buffer grad_out, py::buffer grad_a,
                                    size_t rows, size_t cols) {
        check_native_matrix(rows, cols);
        FloatBuffer go(grad_out), ga(grad_a);
        tm_transpose_backward(go.read(), ga.write(), rows, cols);
    }, py::arg("grad_out"), py::arg("grad_a"),
       py::arg("rows").noconvert(), py::arg("cols").noconvert(),
       "Transpose grad_out of shape (cols, rows) into grad_a of shape (rows, cols).");

    m.def("relu", [](py::buffer a, py::buffer out, size_t m, size_t n) {
        check_native_matrix(m, n, true);
        FloatBuffer av(a), ov(out);
        tm_relu(av.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("relu_backward", [](py::buffer a, py::buffer grad_out,
                               py::buffer grad_in, size_t m, size_t n) {
        check_native_matrix(m, n, true);
        FloatBuffer av(a), go(grad_out), gi(grad_in);
        tm_relu_backward(av.read(), go.read(), gi.write(), m, n);
    }, py::arg("a"), py::arg("grad_out"), py::arg("grad_in"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("bias_add", [](py::buffer a, py::buffer bias, py::buffer out,
                          size_t m, size_t n) {
        check_native_matrix(m, n, true);
        FloatBuffer av(a), bv(bias), ov(out);
        tm_bias_add(av.read(), bv.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("bias"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("bias_add_backward", [](py::buffer grad_out, py::buffer grad_a,
                                   py::buffer grad_bias, size_t m, size_t n) {
        check_native_matrix(m, n, true);
        FloatBuffer go(grad_out), ga(grad_a), gb(grad_bias);
        tm_bias_add_backward(go.read(), ga.write(), gb.write(), m, n);
    }, py::arg("grad_out"), py::arg("grad_a"), py::arg("grad_bias"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("sum", [](py::buffer a, py::buffer out, size_t n) {
        check_native_count(n);
        FloatBuffer av(a), ov(out);
        tm_sum(av.read(), ov.write(), n);
    }, py::arg("a"), py::arg("out"), py::arg("n").noconvert());

    m.def("sum_backward", [](py::buffer grad_out, py::buffer grad_a, size_t n) {
        check_native_count(n);
        FloatBuffer go(grad_out), ga(grad_a);
        tm_sum_backward(go.read(), ga.write(), n);
    }, py::arg("grad_out"), py::arg("grad_a"), py::arg("n").noconvert());

    m.def("sum_rows", [](py::buffer a, py::buffer out, size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer av(a), ov(out);
        tm_sum_rows(av.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("sum_rows_backward", [](py::buffer grad_out, py::buffer grad_a,
                                   size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer go(grad_out), ga(grad_a);
        tm_sum_rows_backward(go.read(), ga.write(), m, n);
    }, py::arg("grad_out"), py::arg("grad_a"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("sum_cols", [](py::buffer a, py::buffer out, size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer av(a), ov(out);
        tm_sum_cols(av.read(), ov.write(), m, n);
    }, py::arg("a"), py::arg("out"),
       py::arg("m").noconvert(), py::arg("n").noconvert());

    m.def("sum_cols_backward", [](py::buffer grad_out, py::buffer grad_a,
                                   size_t m, size_t n) {
        check_native_matrix(m, n);
        FloatBuffer go(grad_out), ga(grad_a);
        tm_sum_cols_backward(go.read(), ga.write(), m, n);
    }, py::arg("grad_out"), py::arg("grad_a"),
       py::arg("m").noconvert(), py::arg("n").noconvert());
}
