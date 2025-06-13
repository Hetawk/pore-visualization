#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "fast_renderer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_renderer_cpp, m)
{
    m.doc() = "High-performance C++ renderer for 3D pore visualization";

    // Point3D class
    py::class_<PoreViz::Point3D>(m, "Point3D")
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def_readwrite("x", &PoreViz::Point3D::x)
        .def_readwrite("y", &PoreViz::Point3D::y)
        .def_readwrite("z", &PoreViz::Point3D::z)
        .def("__repr__", [](const PoreViz::Point3D &p)
             { return "Point3D(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ", " + std::to_string(p.z) + ")"; });

    // Sphere class
    py::class_<PoreViz::Sphere>(m, "Sphere")
        .def(py::init<const PoreViz::Point3D &, float, const std::array<float, 4> &, float>(),
             py::arg("center"), py::arg("radius"), py::arg("color"), py::arg("opacity") = 1.0f)
        .def_readwrite("center", &PoreViz::Sphere::center)
        .def_readwrite("radius", &PoreViz::Sphere::radius)
        .def_readwrite("color", &PoreViz::Sphere::color)
        .def_readwrite("opacity", &PoreViz::Sphere::opacity);

    // Bond class
    py::class_<PoreViz::Bond>(m, "Bond")
        .def(py::init<const PoreViz::Point3D &, const PoreViz::Point3D &, float, const std::array<float, 4> &>(),
             py::arg("start"), py::arg("end"), py::arg("thickness"), py::arg("color"))
        .def_readwrite("start", &PoreViz::Bond::start)
        .def_readwrite("end", &PoreViz::Bond::end)
        .def_readwrite("thickness", &PoreViz::Bond::thickness)
        .def_readwrite("color", &PoreViz::Bond::color);

    // ViewParameters class
    py::class_<PoreViz::ViewParameters>(m, "ViewParameters")
        .def(py::init<>())
        .def_readwrite("elevation", &PoreViz::ViewParameters::elevation)
        .def_readwrite("azimuth", &PoreViz::ViewParameters::azimuth)
        .def_readwrite("zoom", &PoreViz::ViewParameters::zoom)
        .def_readwrite("camera_position", &PoreViz::ViewParameters::camera_position)
        .def_readwrite("target_position", &PoreViz::ViewParameters::target_position)
        .def_readwrite("lighting_intensity", &PoreViz::ViewParameters::lighting_intensity)
        .def_readwrite("background_color", &PoreViz::ViewParameters::background_color);

    // FastRenderer class
    py::class_<PoreViz::FastRenderer>(m, "FastRenderer")
        .def(py::init<bool>(), py::arg("use_threading") = true)
        .def("set_spheres", &PoreViz::FastRenderer::setSpheres)
        .def("set_bonds", &PoreViz::FastRenderer::setBonds)
        .def("add_sphere", &PoreViz::FastRenderer::addSphere)
        .def("add_bond", &PoreViz::FastRenderer::addBond)
        .def("clear_data", &PoreViz::FastRenderer::clearData)
        .def("set_view_parameters", &PoreViz::FastRenderer::setViewParameters)
        .def("get_view_parameters", &PoreViz::FastRenderer::getViewParameters)
        .def("update_view", &PoreViz::FastRenderer::updateView)
        .def("set_camera_position", &PoreViz::FastRenderer::setCameraPosition)
        .def("set_lighting", &PoreViz::FastRenderer::setLighting)
        .def("set_background_color", &PoreViz::FastRenderer::setBackgroundColor)
        .def("render_frame", &PoreViz::FastRenderer::renderFrame)
        .def("render_to_buffer", [](PoreViz::FastRenderer &self, py::array_t<float> buffer, int width, int height)
             {
            py::buffer_info buf_info = buffer.request();
            self.renderToBuffer(static_cast<float*>(buf_info.ptr), width, height); })
        .def("needs_update", &PoreViz::FastRenderer::needsUpdate)
        .def("mark_updated", &PoreViz::FastRenderer::markUpdated)
        .def("set_threading", &PoreViz::FastRenderer::setThreading)
        .def("optimize_for_size", &PoreViz::FastRenderer::optimizeForSize)
        .def("get_sphere_count", &PoreViz::FastRenderer::getSphereCount)
        .def("get_bond_count", &PoreViz::FastRenderer::getBondCount)
        .def("pick_spheres", &PoreViz::FastRenderer::pickSpheres)
        .def("screen_to_world", &PoreViz::FastRenderer::screenToWorld)
        .def("world_to_screen", &PoreViz::FastRenderer::worldToScreen);

    // Utility functions
    m.def("convert_pore_data_to_spheres", &PoreViz::convertPoreDataToSpheres,
          py::arg("diameters"), py::arg("volumes"), py::arg("positions"), py::arg("colormap") = "viridis");
    m.def("generate_atomic_bonds", &PoreViz::generateAtomicBonds,
          py::arg("spheres"), py::arg("max_distance") = 2.0f, py::arg("thickness") = 0.5f);
    m.def("distance_squared", &PoreViz::distanceSquared);
    m.def("normalize", &PoreViz::normalize);
    m.def("cross", &PoreViz::cross);
    m.def("dot", &PoreViz::dot);
}
