#ifndef FAST_RENDERER_HPP
#define FAST_RENDERER_HPP

#include <vector>
#include <memory>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <execution>

namespace PoreViz
{

    struct Point3D
    {
        float x, y, z;
        Point3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    };

    struct Sphere
    {
        Point3D center;
        float radius;
        std::array<float, 4> color; // RGBA
        float opacity;

        Sphere(const Point3D &c, float r, const std::array<float, 4> &col, float op = 1.0f)
            : center(c), radius(r), color(col), opacity(op) {}
    };

    struct Bond
    {
        Point3D start, end;
        float thickness;
        std::array<float, 4> color;

        Bond(const Point3D &s, const Point3D &e, float t, const std::array<float, 4> &col)
            : start(s), end(e), thickness(t), color(col) {}
    };

    struct ViewParameters
    {
        float elevation;
        float azimuth;
        float zoom;
        Point3D camera_position;
        Point3D target_position;
        float lighting_intensity;
        std::array<float, 3> background_color;

        ViewParameters()
            : elevation(20.0f), azimuth(45.0f), zoom(1.0f),
              camera_position(0, 0, 10), target_position(0, 0, 0),
              lighting_intensity(0.8f), background_color({1.0f, 1.0f, 1.0f}) {}
    };

    class FastRenderer
    {
    private:
        std::vector<Sphere> spheres_;
        std::vector<Bond> bonds_;
        ViewParameters view_params_;
        std::mutex data_mutex_;
        std::atomic<bool> needs_update_;

        // Performance optimization
        bool use_threading_;
        size_t num_threads_;

        // Spatial optimization
        struct SpatialGrid
        {
            std::vector<std::vector<size_t>> grid;
            float cell_size;
            Point3D min_bounds, max_bounds;
            int grid_size_x, grid_size_y, grid_size_z;
        };

        std::unique_ptr<SpatialGrid> spatial_grid_;

        // Private methods
        void buildSpatialGrid();
        void updateBounds();
        std::vector<size_t> getVisibleSpheres(const ViewParameters &view) const;
        std::vector<size_t> getVisibleBonds(const ViewParameters &view) const;
        void renderSphere(const Sphere &sphere, float *buffer, int width, int height);
        void renderBond(const Bond &bond, float *buffer, int width, int height);

    public:
        FastRenderer(bool use_threading = true);
        ~FastRenderer() = default;

        // Data management
        void setSpheres(const std::vector<Sphere> &spheres);
        void setBonds(const std::vector<Bond> &bonds);
        void addSphere(const Sphere &sphere);
        void addBond(const Bond &bond);
        void clearData();

        // View control
        void setViewParameters(const ViewParameters &params);
        ViewParameters getViewParameters() const;
        void updateView(float elevation, float azimuth, float zoom);
        void setCameraPosition(const Point3D &position, const Point3D &target);
        void setLighting(float intensity);
        void setBackgroundColor(float r, float g, float b);

        // Rendering
        std::vector<float> renderFrame(int width, int height);
        void renderToBuffer(float *buffer, int width, int height);
        bool needsUpdate() const { return needs_update_.load(); }
        void markUpdated() { needs_update_.store(false); }

        // Performance tuning
        void setThreading(bool enabled, size_t num_threads = 0);
        void optimizeForSize(size_t expected_spheres, size_t expected_bonds);

        // Statistics
        size_t getSphereCount() const { return spheres_.size(); }
        size_t getBondCount() const { return bonds_.size(); }
        float getLastRenderTime() const;

        // Frustum culling and LOD
        void enableFrustumCulling(bool enabled);
        void setLODLevels(const std::vector<float> &distances);

        // Interactive features
        std::vector<size_t> pickSpheres(int screen_x, int screen_y, float radius);
        Point3D screenToWorld(int screen_x, int screen_y, float depth);
        std::array<float, 2> worldToScreen(const Point3D &world_point, int width, int height);
    };

    // Utility functions for data conversion
    std::vector<Sphere> convertPoreDataToSpheres(
        const std::vector<float> &diameters,
        const std::vector<float> &volumes,
        const std::vector<std::array<float, 3>> &positions,
        const std::string &colormap = "viridis");

    std::vector<Bond> generateAtomicBonds(
        const std::vector<Sphere> &spheres,
        float max_distance = 2.0f,
        float thickness = 0.5f);

    // Mathematical utilities
    float distanceSquared(const Point3D &a, const Point3D &b);
    Point3D normalize(const Point3D &v);
    Point3D cross(const Point3D &a, const Point3D &b);
    float dot(const Point3D &a, const Point3D &b);

} // namespace PoreViz

#endif // FAST_RENDERER_HPP
