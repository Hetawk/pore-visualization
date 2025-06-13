#include "fast_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

namespace PoreViz
{

    FastRenderer::FastRenderer(bool use_threading)
        : use_threading_(use_threading),
          num_threads_(use_threading ? std::thread::hardware_concurrency() : 1),
          needs_update_(true)
    {
        if (num_threads_ == 0)
            num_threads_ = 4; // fallback
        spatial_grid_ = std::make_unique<SpatialGrid>();
    }

    void FastRenderer::setSpheres(const std::vector<Sphere> &spheres)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        spheres_ = spheres;
        needs_update_.store(true);
        buildSpatialGrid();
    }

    void FastRenderer::setBonds(const std::vector<Bond> &bonds)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        bonds_ = bonds;
        needs_update_.store(true);
    }

    void FastRenderer::addSphere(const Sphere &sphere)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        spheres_.push_back(sphere);
        needs_update_.store(true);
    }

    void FastRenderer::addBond(const Bond &bond)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        bonds_.push_back(bond);
        needs_update_.store(true);
    }

    void FastRenderer::clearData()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        spheres_.clear();
        bonds_.clear();
        spatial_grid_->grid.clear();
        needs_update_.store(true);
    }

    void FastRenderer::setViewParameters(const ViewParameters &params)
    {
        view_params_ = params;
        needs_update_.store(true);
    }

    ViewParameters FastRenderer::getViewParameters() const
    {
        return view_params_;
    }

    void FastRenderer::updateView(float elevation, float azimuth, float zoom)
    {
        view_params_.elevation = elevation;
        view_params_.azimuth = azimuth;
        view_params_.zoom = zoom;
        needs_update_.store(true);
    }

    void FastRenderer::setCameraPosition(const Point3D &position, const Point3D &target)
    {
        view_params_.camera_position = position;
        view_params_.target_position = target;
        needs_update_.store(true);
    }

    void FastRenderer::setLighting(float intensity)
    {
        view_params_.lighting_intensity = std::clamp(intensity, 0.0f, 2.0f);
        needs_update_.store(true);
    }

    void FastRenderer::setBackgroundColor(float r, float g, float b)
    {
        view_params_.background_color = {
            std::clamp(r, 0.0f, 1.0f),
            std::clamp(g, 0.0f, 1.0f),
            std::clamp(b, 0.0f, 1.0f)};
        needs_update_.store(true);
    }

    void FastRenderer::buildSpatialGrid()
    {
        if (spheres_.empty())
            return;

        updateBounds();

        // Calculate optimal grid size
        float volume = (spatial_grid_->max_bounds.x - spatial_grid_->min_bounds.x) *
                       (spatial_grid_->max_bounds.y - spatial_grid_->min_bounds.y) *
                       (spatial_grid_->max_bounds.z - spatial_grid_->min_bounds.z);

        float optimal_cell_size = std::cbrt(volume / spheres_.size()) * 2.0f;
        spatial_grid_->cell_size = std::max(optimal_cell_size, 1.0f);

        spatial_grid_->grid_size_x = static_cast<int>(
                                         (spatial_grid_->max_bounds.x - spatial_grid_->min_bounds.x) / spatial_grid_->cell_size) +
                                     1;
        spatial_grid_->grid_size_y = static_cast<int>(
                                         (spatial_grid_->max_bounds.y - spatial_grid_->min_bounds.y) / spatial_grid_->cell_size) +
                                     1;
        spatial_grid_->grid_size_z = static_cast<int>(
                                         (spatial_grid_->max_bounds.z - spatial_grid_->min_bounds.z) / spatial_grid_->cell_size) +
                                     1;

        int total_cells = spatial_grid_->grid_size_x * spatial_grid_->grid_size_y * spatial_grid_->grid_size_z;
        spatial_grid_->grid.clear();
        spatial_grid_->grid.resize(total_cells);

        // Assign spheres to grid cells
        for (size_t i = 0; i < spheres_.size(); ++i)
        {
            const auto &sphere = spheres_[i];
            int x = static_cast<int>((sphere.center.x - spatial_grid_->min_bounds.x) / spatial_grid_->cell_size);
            int y = static_cast<int>((sphere.center.y - spatial_grid_->min_bounds.y) / spatial_grid_->cell_size);
            int z = static_cast<int>((sphere.center.z - spatial_grid_->min_bounds.z) / spatial_grid_->cell_size);

            x = std::clamp(x, 0, spatial_grid_->grid_size_x - 1);
            y = std::clamp(y, 0, spatial_grid_->grid_size_y - 1);
            z = std::clamp(z, 0, spatial_grid_->grid_size_z - 1);

            int index = x + y * spatial_grid_->grid_size_x + z * spatial_grid_->grid_size_x * spatial_grid_->grid_size_y;
            spatial_grid_->grid[index].push_back(i);
        }
    }

    void FastRenderer::updateBounds()
    {
        if (spheres_.empty())
        {
            spatial_grid_->min_bounds = Point3D(-10, -10, -10);
            spatial_grid_->max_bounds = Point3D(10, 10, 10);
            return;
        }

        spatial_grid_->min_bounds = spheres_[0].center;
        spatial_grid_->max_bounds = spheres_[0].center;

        for (const auto &sphere : spheres_)
        {
            spatial_grid_->min_bounds.x = std::min(spatial_grid_->min_bounds.x, sphere.center.x - sphere.radius);
            spatial_grid_->min_bounds.y = std::min(spatial_grid_->min_bounds.y, sphere.center.y - sphere.radius);
            spatial_grid_->min_bounds.z = std::min(spatial_grid_->min_bounds.z, sphere.center.z - sphere.radius);

            spatial_grid_->max_bounds.x = std::max(spatial_grid_->max_bounds.x, sphere.center.x + sphere.radius);
            spatial_grid_->max_bounds.y = std::max(spatial_grid_->max_bounds.y, sphere.center.y + sphere.radius);
            spatial_grid_->max_bounds.z = std::max(spatial_grid_->max_bounds.z, sphere.center.z + sphere.radius);
        }

        // Add some padding
        float padding = 1.0f;
        spatial_grid_->min_bounds.x -= padding;
        spatial_grid_->min_bounds.y -= padding;
        spatial_grid_->min_bounds.z -= padding;
        spatial_grid_->max_bounds.x += padding;
        spatial_grid_->max_bounds.y += padding;
        spatial_grid_->max_bounds.z += padding;
    }

    std::vector<float> FastRenderer::renderFrame(int width, int height)
    {
        std::vector<float> buffer(width * height * 4); // RGBA
        renderToBuffer(buffer.data(), width, height);
        return buffer;
    }

    void FastRenderer::renderToBuffer(float *buffer, int width, int height)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Clear buffer with background color
        for (int i = 0; i < width * height; ++i)
        {
            buffer[i * 4 + 0] = view_params_.background_color[0]; // R
            buffer[i * 4 + 1] = view_params_.background_color[1]; // G
            buffer[i * 4 + 2] = view_params_.background_color[2]; // B
            buffer[i * 4 + 3] = 1.0f;                             // A
        }

        // Get visible objects
        auto visible_spheres = getVisibleSpheres(view_params_);
        auto visible_bonds = getVisibleBonds(view_params_);

        // Render spheres in parallel if threading is enabled
        if (use_threading_ && visible_spheres.size() > 100)
        {
            // Use manual threading instead of std::execution for better compatibility
            const size_t num_spheres = visible_spheres.size();
            const size_t spheres_per_thread = (num_spheres + num_threads_ - 1) / num_threads_;

            std::vector<std::thread> threads;
            threads.reserve(num_threads_);

            for (size_t t = 0; t < num_threads_; ++t)
            {
                size_t start_idx = t * spheres_per_thread;
                size_t end_idx = std::min(start_idx + spheres_per_thread, num_spheres);

                if (start_idx < end_idx)
                {
                    threads.emplace_back([&, start_idx, end_idx]()
                                         {
                        for (size_t i = start_idx; i < end_idx; ++i)
                        {
                            const auto &sphere = spheres_[visible_spheres[i]];
                            renderSphere(sphere, buffer, width, height);
                        } });
                }
            }

            // Wait for all threads to complete
            for (auto &thread : threads)
            {
                thread.join();
            }
        }
        else
        {
            for (size_t sphere_idx : visible_spheres)
            {
                const auto &sphere = spheres_[sphere_idx];
                renderSphere(sphere, buffer, width, height);
            }
        }

        // Render bonds
        for (size_t bond_idx : visible_bonds)
        {
            const auto &bond = bonds_[bond_idx];
            renderBond(bond, buffer, width, height);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        // Store render time for performance monitoring (unused for now)
        (void)duration;
    }

    void FastRenderer::renderSphere(const Sphere &sphere, float *buffer, int width, int height)
    {
        // Simple sphere rendering - project to screen space and draw circle
        auto screen_pos = worldToScreen(sphere.center, width, height);
        float screen_x = screen_pos[0];
        float screen_y = screen_pos[1];

        // Calculate screen radius based on distance and sphere size
        float distance = std::sqrt(distanceSquared(sphere.center, view_params_.camera_position));
        float screen_radius = (sphere.radius * view_params_.zoom * width) / (distance * 2.0f);

        // Clamp to reasonable bounds
        screen_radius = std::clamp(screen_radius, 1.0f, static_cast<float>(std::min(width, height)) / 4.0f);

        int min_x = static_cast<int>(screen_x - screen_radius);
        int max_x = static_cast<int>(screen_x + screen_radius);
        int min_y = static_cast<int>(screen_y - screen_radius);
        int max_y = static_cast<int>(screen_y + screen_radius);

        min_x = std::max(0, min_x);
        max_x = std::min(width - 1, max_x);
        min_y = std::max(0, min_y);
        max_y = std::min(height - 1, max_y);

        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                float dx = x - screen_x;
                float dy = y - screen_y;
                float dist_sq = dx * dx + dy * dy;

                if (dist_sq <= screen_radius * screen_radius)
                {
                    // Simple alpha blending
                    int pixel_idx = (y * width + x) * 4;
                    float alpha = sphere.opacity * (1.0f - dist_sq / (screen_radius * screen_radius));

                    buffer[pixel_idx + 0] = buffer[pixel_idx + 0] * (1.0f - alpha) + sphere.color[0] * alpha;
                    buffer[pixel_idx + 1] = buffer[pixel_idx + 1] * (1.0f - alpha) + sphere.color[1] * alpha;
                    buffer[pixel_idx + 2] = buffer[pixel_idx + 2] * (1.0f - alpha) + sphere.color[2] * alpha;
                }
            }
        }
    }

    void FastRenderer::renderBond(const Bond &bond, float *buffer, int width, int height)
    {
        // Render bond as a line between two points
        auto start_screen = worldToScreen(bond.start, width, height);
        auto end_screen = worldToScreen(bond.end, width, height);

        // Simple line drawing using Bresenham's algorithm (simplified)
        int x0 = static_cast<int>(start_screen[0]);
        int y0 = static_cast<int>(start_screen[1]);
        int x1 = static_cast<int>(end_screen[0]);
        int y1 = static_cast<int>(end_screen[1]);

        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        int x = x0, y = y0;

        while (true)
        {
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                int pixel_idx = (y * width + x) * 4;
                float alpha = 0.7f; // Bond alpha

                buffer[pixel_idx + 0] = buffer[pixel_idx + 0] * (1.0f - alpha) + bond.color[0] * alpha;
                buffer[pixel_idx + 1] = buffer[pixel_idx + 1] * (1.0f - alpha) + bond.color[1] * alpha;
                buffer[pixel_idx + 2] = buffer[pixel_idx + 2] * (1.0f - alpha) + bond.color[2] * alpha;
            }

            if (x == x1 && y == y1)
                break;

            int e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                x += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y += sy;
            }
        }
    }

    std::vector<size_t> FastRenderer::getVisibleSpheres(const ViewParameters &view) const
    {
        std::vector<size_t> visible;
        visible.reserve(spheres_.size());

        // Simple frustum culling - check distance from camera
        float max_distance = 100.0f / view.zoom; // Adjust based on zoom

        for (size_t i = 0; i < spheres_.size(); ++i)
        {
            float dist_sq = distanceSquared(spheres_[i].center, view.camera_position);
            if (dist_sq <= max_distance * max_distance)
            {
                visible.push_back(i);
            }
        }

        return visible;
    }

    std::vector<size_t> FastRenderer::getVisibleBonds(const ViewParameters &view) const
    {
        std::vector<size_t> visible;
        visible.reserve(bonds_.size());

        // Simple culling for bonds
        float max_distance = 100.0f / view.zoom;

        for (size_t i = 0; i < bonds_.size(); ++i)
        {
            Point3D mid_point = {
                (bonds_[i].start.x + bonds_[i].end.x) * 0.5f,
                (bonds_[i].start.y + bonds_[i].end.y) * 0.5f,
                (bonds_[i].start.z + bonds_[i].end.z) * 0.5f};

            float dist_sq = distanceSquared(mid_point, view.camera_position);
            if (dist_sq <= max_distance * max_distance)
            {
                visible.push_back(i);
            }
        }

        return visible;
    }

    std::array<float, 2> FastRenderer::worldToScreen(const Point3D &world_point, int width, int height)
    {
        // Simple orthographic projection
        float x = world_point.x * view_params_.zoom + width * 0.5f;
        float y = world_point.y * view_params_.zoom + height * 0.5f;
        return {x, y};
    }

    // Performance tuning methods
    void FastRenderer::setThreading(bool enabled, size_t num_threads)
    {
        use_threading_ = enabled;
        if (num_threads > 0)
        {
            num_threads_ = num_threads;
        }
        else
        {
            num_threads_ = std::thread::hardware_concurrency();
        }
    }

    void FastRenderer::optimizeForSize(size_t expected_spheres, size_t expected_bonds)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        spheres_.reserve(expected_spheres);
        bonds_.reserve(expected_bonds);
    }

    float FastRenderer::getLastRenderTime() const
    {
        // Return a dummy value for now - in a real implementation,
        // you'd store the actual render time
        return 16.67f; // ~60 FPS
    }

    // Frustum culling and LOD methods
    void FastRenderer::enableFrustumCulling(bool enabled)
    {
        // Implementation for frustum culling enable/disable
        // For now, this is a placeholder
        (void)enabled;
    }

    void FastRenderer::setLODLevels(const std::vector<float> &distances)
    {
        // Implementation for LOD level setting
        // For now, this is a placeholder
        (void)distances;
    }

    // Interactive features methods
    std::vector<size_t> FastRenderer::pickSpheres(int screen_x, int screen_y, float radius)
    {
        std::vector<size_t> picked_spheres;

        // Simple picking implementation - find spheres near screen coordinates
        for (size_t i = 0; i < spheres_.size(); ++i)
        {
            auto screen_pos = worldToScreen(spheres_[i].center, 800, 600); // Default screen size
            float dist_sq = (screen_pos[0] - screen_x) * (screen_pos[0] - screen_x) +
                            (screen_pos[1] - screen_y) * (screen_pos[1] - screen_y);
            if (dist_sq <= radius * radius)
            {
                picked_spheres.push_back(i);
            }
        }

        return picked_spheres;
    }

    Point3D FastRenderer::screenToWorld(int screen_x, int screen_y, float depth)
    {
        // Simple reverse projection
        float normalized_x = (2.0f * screen_x) / 800.0f - 1.0f;
        float normalized_y = 1.0f - (2.0f * screen_y) / 600.0f;

        return Point3D(
            normalized_x * view_params_.zoom,
            normalized_y * view_params_.zoom,
            depth);
    }

    // Utility functions for data conversion
    std::vector<Sphere> convertPoreDataToSpheres(
        const std::vector<float> &diameters,
        const std::vector<float> &volumes,
        const std::vector<std::array<float, 3>> &positions,
        const std::string &colormap)
    {
        std::vector<Sphere> spheres;
        size_t count = std::min({diameters.size(), volumes.size(), positions.size()});

        for (size_t i = 0; i < count; ++i)
        {
            Point3D center(positions[i][0], positions[i][1], positions[i][2]);
            float radius = diameters[i] * 0.5f;

            // Simple color mapping based on volume
            float normalized_volume = volumes[i] / *std::max_element(volumes.begin(), volumes.end());
            std::array<float, 4> color;

            if (colormap == "viridis")
            {
                color[0] = 0.267f + 0.533f * normalized_volume; // R
                color[1] = 0.004f + 0.867f * normalized_volume; // G
                color[2] = 0.329f + 0.344f * normalized_volume; // B
                color[3] = 1.0f;                                // A
            }
            else
            {
                // Default to blue-to-red gradient
                color[0] = normalized_volume;
                color[1] = 0.5f;
                color[2] = 1.0f - normalized_volume;
                color[3] = 1.0f;
            }

            spheres.emplace_back(center, radius, color, 1.0f);
        }

        return spheres;
    }

    std::vector<Bond> generateAtomicBonds(
        const std::vector<Sphere> &spheres,
        float max_distance,
        float thickness)
    {
        std::vector<Bond> bonds;

        for (size_t i = 0; i < spheres.size(); ++i)
        {
            for (size_t j = i + 1; j < spheres.size(); ++j)
            {
                float dist = std::sqrt(distanceSquared(spheres[i].center, spheres[j].center));
                float combined_radius = spheres[i].radius + spheres[j].radius;

                if (dist <= max_distance && dist > combined_radius)
                {
                    std::array<float, 4> bond_color = {0.7f, 0.7f, 0.7f, 1.0f}; // Gray
                    bonds.emplace_back(spheres[i].center, spheres[j].center, thickness, bond_color);
                }
            }
        }

        return bonds;
    }

    // Utility functions implementation
    float distanceSquared(const Point3D &a, const Point3D &b)
    {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    Point3D normalize(const Point3D &v)
    {
        float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (length > 0.0f)
        {
            return Point3D(v.x / length, v.y / length, v.z / length);
        }
        return Point3D(0, 0, 0);
    }

    Point3D cross(const Point3D &a, const Point3D &b)
    {
        return Point3D(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
    }

    float dot(const Point3D &a, const Point3D &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

} // namespace PoreViz
