/**
 * @file  bindings.cpp
 * @brief The python module framecor._framecor
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "framecor.h"

namespace py = pybind11;

namespace {

using Frames = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using Ints = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

class PyCorrelator {
public:
    PyCorrelator(int height, int width, std::vector<int> down, std::vector<int> across, int window,
                 int search, const std::string &backend, int batch, int statSize, int device, int threads)
    {
        p.height = height;
        p.width = width;
        p.down = std::move(down);
        p.across = std::move(across);
        p.window = window;
        p.search = search;
        p.batch = batch;
        p.statSize = statSize;
        framecor::checkParams(p);
        const std::string name = backend == "auto" ? (framecor::hasCuda() ? "cuda" : "cpu") : backend;
        if (name == "cuda")
            impl = framecor::makeCuda(p, device);
        else if (name == "cpu")
            impl = framecor::makeCpu(p, threads);
        else
            throw std::invalid_argument("framecor: the backend must be auto, cuda or cpu, not " + backend);
    }

    void setReference(const Frames &frame)
    {
        if (frame.ndim() != 2 || frame.shape(0) != p.height || frame.shape(1) != p.width)
            throw std::invalid_argument("framecor: the reference frame must be height x width");
        const uint8_t *data = frame.data();
        py::gil_scoped_release release;
        impl->setReference(data);
        ready = true;
    }

    py::dict correlate(const Frames &frames, std::optional<Ints> gross)
    {
        if (!ready) throw std::runtime_error("framecor: set_reference first");
        if (frames.ndim() != 3 || frames.shape(1) != p.height || frames.shape(2) != p.width)
            throw std::invalid_argument("framecor: the frames must be frames x height x width");
        const py::ssize_t n = frames.shape(0), ny = p.down.size(), nx = p.across.size(), nwin = ny * nx;

        // the gross offsets of each frame and window: given per frame (n x 2), or per window
        std::vector<int> g;
        if (gross) {
            const Ints &a = *gross;
            if (a.ndim() == 2 && a.shape(0) == n && a.shape(1) == 2) {
                g.resize(size_t(n) * nwin * 2);
                for (py::ssize_t f = 0; f < n; f++)
                    for (py::ssize_t k = 0; k < nwin; k++) {
                        g[2 * (f * nwin + k)] = a.at(f, 0);
                        g[2 * (f * nwin + k) + 1] = a.at(f, 1);
                    }
            }
            else if (a.ndim() == 4 && a.shape(0) == n && a.shape(1) == ny && a.shape(2) == nx && a.shape(3) == 2)
                g.assign(a.data(), a.data() + a.size());
            else
                throw std::invalid_argument("framecor: gross must be frames x 2 or frames x down x across x 2");
        }

        py::array_t<float> offset({n, ny, nx, py::ssize_t(2)}), peak({n, ny, nx}), snr({n, ny, nx}),
            cov({n, ny, nx, py::ssize_t(3)});
        const framecor::Outputs out{offset.mutable_data(), peak.mutable_data(), snr.mutable_data(),
                                    cov.mutable_data()};
        if (n > 0) {
            const uint8_t *data = frames.data();
            py::gil_scoped_release release;
            impl->correlate(data, int(n), g.empty() ? nullptr : g.data(), out);
        }
        py::dict d;
        d["offsets"] = offset;
        d["peak"] = peak;
        d["snr"] = snr;
        d["cov"] = cov;
        return d;
    }

    std::string backend() const { return impl->backend(); }
    py::tuple shape() const { return py::make_tuple(p.down.size(), p.across.size()); }
    const framecor::Params &params() const { return p; }

private:
    framecor::Params p;
    std::unique_ptr<framecor::Correlator> impl;
    bool ready = false;
};

} // namespace

PYBIND11_MODULE(_framecor, m)
{
    m.doc() = "Dense normalized cross-correlation of video frames with a reference frame";
    m.attr("__version__") = FRAMECOR_VERSION;
    m.def("has_cuda", &framecor::hasCuda, "Whether the CUDA backend is built and a GPU is available");

    py::class_<PyCorrelator>(m, "Correlator",
        "The correlator of 8-bit frames with a reference frame, on a grid of square windows\n\n"
        "height, width: the size of the frames\n"
        "down, across: the first pixels of the rows and of the columns of windows\n"
        "window: the size of the windows; search: the half search range, pixels\n"
        "backend: 'cuda', 'cpu' or 'auto' (CUDA if available)\n"
        "batch: the windows correlated together; stat_size: the SNR window\n"
        "device: the GPU; threads: the CPU threads (0 for the default)")
        .def(py::init<int, int, std::vector<int>, std::vector<int>, int, int, const std::string &, int, int, int, int>(),
             py::arg("height"), py::arg("width"), py::arg("down"), py::arg("across"), py::kw_only(),
             py::arg("window") = 64, py::arg("search") = 48, py::arg("backend") = "auto",
             py::arg("batch") = 4096, py::arg("stat_size") = 21, py::arg("device") = 0, py::arg("threads") = 0)
        .def("set_reference", &PyCorrelator::setReference, py::arg("frame"),
             "Set the reference frame (height x width, uint8)")
        .def("correlate", &PyCorrelator::correlate, py::arg("frames"), py::arg("gross") = py::none(),
             "Correlate the frames (frames x height x width, uint8) with the reference frame\n\n"
             "gross: integer offsets (down, across) of the search windows, frames x 2 or\n"
             "frames x down x across x 2 (default: none)\n\n"
             "Returns a dict of float32 arrays: offsets (frames x down x across x 2: down, across,\n"
             "the motion of the frame; NaN if not valid), peak (the normalized correlation),\n"
             "snr, cov (the covariance of the offsets: down, across, cross)")
        .def_property_readonly("backend", &PyCorrelator::backend)
        .def_property_readonly("shape", &PyCorrelator::shape, "the number of windows (down, across)")
        .def_property_readonly("window", [](const PyCorrelator &c) { return c.params().window; })
        .def_property_readonly("search", [](const PyCorrelator &c) { return c.params().search; });
}
// end of file
