#include <vector>
#include <iostream>
#include <optional>
#include <limits>

#include "geometry.h"
#include "vec3.h"

struct camera {
    vec3f origin;
    vec3f direction;
    double fov;
};

struct ray {
    vec3f origin, direction;
};

struct sphere {
    vec3f center;
    double radius;
};

vec3f pixel_ray(camera view, vec2i resolution, vec2i pixel)
{
    // distance to imaginary frustrum
    auto α = 1.;
    // half of the fov angle
    auto θ = view.fov * pi / 180.;
    // pixel angle size
    auto px_θ = vec2f{θ / resolution.x, θ / resolution.y};
    px_θ.x *= pixel.x;
    px_θ.y *= -pixel.y;

    auto yaw = view.direction.z == 0
                   ? 90. * pi / 180.
                   : std::atan2(view.direction.x, view.direction.z);

    auto ray = view.direction;
    // rotate counter to the yaw to align with z
    ray = {ray.x * std::cos(yaw) - ray.z * std::sin(yaw), ray.y,
           ray.x * std::sin(yaw) + ray.z * std::cos(yaw)};

    // pitch
    ray = {
        0,
        ray.z * std::sin(px_θ.y) + ray.y * std::cos(px_θ.y),
        ray.z * std::cos(px_θ.y) - ray.y * std::sin(px_θ.y),
    };

    // restore the yaw
    yaw += px_θ.x;
    ray = {ray.x * std::cos(-yaw) - ray.z * std::sin(-yaw), ray.y,
           ray.x * std::sin(-yaw) + ray.z * std::cos(-yaw)};

    return ray;
}

std::optional<vec3f> intersect(ray r, sphere s)
{
    auto to_sphere = (s.center - r.origin);
    auto s_dir = to_sphere.normalized();
    auto angle_opposite = (s_dir - r.direction).length() / 2;
    auto sin = angle_opposite / s_dir.length();
    auto angle = 2. * std::asin(sin);

    auto cast = std::sin(angle) * to_sphere.length();
    if (cast < s.radius) {
        auto closest_point_d = std::cos(angle) * to_sphere.length();
        auto intersection_depth
            = std::sin(std::acos(cast / s.radius)) * s.radius;
        auto intersection_d = closest_point_d - intersection_depth;

        if (intersection_d > 0)
            return {r.origin + r.direction * (intersection_d)};
    }
    return {};
};

vec3f diffuse(vec3f light, const sphere& sphere, vec3f point)
{
    auto surface_normal = (point - sphere.center).normalized();

    double transmission = 1;
    auto light_dir = (light - point).normalized();

    auto base = vec3f{1, 1, 1};

    auto surface_dot_point = surface_normal.x * light_dir.x
                             + surface_normal.y * light_dir.y
                             + surface_normal.z * light_dir.z;

    return base * transmission * std::max(0., surface_dot_point);
}

using scene = std::vector<sphere>;

vec3f trace(ray view_ray, scene spheres)
{
    auto light = vec3f{-15, 8, 35};
    auto view_color = vec3f{};
    auto nearest = std::numeric_limits<double>::max();
    
    for (const auto& sphere : spheres) {
        auto intersection = intersect(view_ray, sphere);
        if (not intersection) continue;

        auto color = diffuse(light, sphere, *intersection);
        auto shadow_ray
            = ray{*intersection, (light - *intersection).normalized()};

        for (const auto& other : spheres) {
            if (&other == &sphere) continue;
            if (intersect(shadow_ray, other)) color *= 0.001;
        }

        auto distance = (*intersection - view_ray.origin).length();
        if (nearest > distance) {
            nearest = distance;
            view_color = color;
        }
    }

    return view_color;
}

int main()
{
    // auto resolution = vec2i{100, 50};
    auto resolution = vec2i{400, 180};
    auto view = camera{{1, 5, 0}, vec3f{-0.1, -0.1, 1}.normalized(), 35.};

    auto spheres = scene{
        {{-5, -4.5, 30}, 1},
        {{3, -3.25, 40}, 2.5},
        {{-1, 0, 60}, 8},
        {{-8, 6, 45}, 1},
        {{0, -1005, 0}, 1000},
    };

    for (int y(0); y < resolution.y; ++y) {
        for (int x(0); x < resolution.x; ++x) {
            auto ray_direction
                = pixel_ray(view, resolution, vec2i{x, y} - resolution * 0.5);
            auto pixel = trace({view.origin, ray_direction}, spheres);

            std::string c = " ";
            if (pixel.length() >= 0.99) c = "@";
            else if (pixel.length() >= 0.75) c = "%";
            else if (pixel.length() >= 0.40) c = "*";
            else if (pixel.length() >= 0.10) c = ":";
            else if (pixel.length() > 0) c = "·";
            std::cout << c;
        }
        std::cout << "\n";
    }
}
