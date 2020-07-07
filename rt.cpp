#include <vector>
#include <iostream>
#include <optional>
#include <variant>
#include <chrono>
#include <limits>
#include <sstream>

#include "geometry.h"
#include "vec3.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

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
    vec3f color{1, 1, 1};
};

using object = std::variant<sphere>;

using scene = std::vector<object>;
vec3f trace(const ray& view_ray, const scene& objects);

struct intersection;
using material = vec3f (*)(const ray&, const object&, const intersection&,
                           const scene&);

struct intersection {
    vec3f p;
    vec3f surface_normal;
    material mat;
};

vec3f random_direction()
{
    while (true) {
        auto v = vec3f{drand48(), drand48(), drand48()} * 2 - vec3f{1, 1, 1};
        if (v.length2() < 1)
            return v;
    }
}

vec3f reflect(const vec3f& v, const vec3f& normal)
{
    return v - normal * 2.0 * v.dot(normal);
}

vec3f mirror_diffuse(const ray& observer, const object& obj,
                     const intersection& intersection, const scene& objects)
{
    auto reflection =
        reflect(observer.direction, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection};

    return std::get<sphere>(obj).color * (trace(reflected_ray, objects) * 0.7);
}

vec3f matte_diffuse(const ray&, const object& obj,
                    const intersection& intersection, const scene& objects)
{
    auto new_direction =
        intersection.p + intersection.surface_normal + random_direction();
    auto normal = (new_direction - intersection.p).normalized();
    auto diffuse_ray = ray{intersection.p, normal};

    return std::get<sphere>(obj).color * (trace(diffuse_ray, objects) * 0.5);
}

vec3f glossy_diffuse(const ray& observer, const object& obj,
                     const intersection& intersection, const scene& objects)
{
    auto reflection =
        reflect(observer.direction, intersection.surface_normal).normalized();
    auto diffused =
        intersection.p + intersection.surface_normal + random_direction();
    auto diffusion = (diffused - intersection.p).normalized();

    auto gloss_ray = ray{intersection.p,
                         (reflection * 0.85 + diffusion * 0.15).normalized()};

    return std::get<sphere>(obj).color * (trace(gloss_ray, objects) * 0.7);
}

std::optional<intersection> intersect(const ray& r, const sphere& s)
{
    auto to_sphere = (s.center - r.origin);
    auto s_dir = to_sphere.normalized();
    auto angle_opposite = (s_dir - r.direction).length() / 2;
    auto sin = angle_opposite / s_dir.length();
    auto angle = 2. * std::asin(sin);

    auto cast = std::sin(angle) * to_sphere.length();
    if (cast < s.radius) {
        auto closest_point_d = std::cos(angle) * to_sphere.length();
        auto intersection_depth =
            std::sin(std::acos(cast / s.radius)) * s.radius;
        auto intersection_distance = closest_point_d - intersection_depth;

        auto intersection_point =
            r.origin + r.direction * intersection_distance;
        auto surface_normal = (intersection_point - s.center).normalized();

        if (intersection_distance > 0)
            return {{intersection_point, surface_normal, glossy_diffuse}};
    }
    return {};
}

std::optional<intersection> intersect(const ray& ray, const object& obj)
{
    return std::visit([&ray](auto& obj) { return intersect(ray, obj); }, obj);
}

vec3f diffuse(const ray& observer, const object& obj,
              const intersection& intersection, const scene& objects)
{
    return intersection.mat(observer, obj, intersection, objects);
}

vec3f trace(const ray& view_ray, const scene& objects)
{
    auto light = vec3f{-15, 8, 35};
    auto view_color = vec3f{.9, .95, 1};
    auto nearest = std::numeric_limits<double>::max();

    for (const auto& obj : objects) {
        auto intersection = intersect(view_ray, obj);
        if (not intersection)
            continue;

        auto distance = (intersection->p - view_ray.origin).length();
        if (nearest <= distance)
            continue;
        nearest = distance;

        auto color = diffuse(view_ray, obj, *intersection, objects);
        view_color = color;
    }

    return view_color;
}

// auto shadow_ray =
//     ray{*intersection, (light - *intersection).normalized()};

// for (const auto &other : objects) {
//     if (&other == &sphere)
//         continue;
//     if (intersect(shadow_ray, other))
//         color *= 0.1;
// }

vec3f pixel_ray(camera view, vec2i resolution, vec2f pixel)
{
    // distance to imaginary frustrum
    // auto α = 1.;
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

vec3<unsigned char> rgb_light(vec3f light)
{
    return vec3<unsigned char>(light * 255.);
}

vec3f supersample(const camera& view, const vec2i& resolution,
                  const scene& objects, vec2i pixel)
{
    constexpr auto supersampling{8};

    vec3f color{};

    for (int sample{}; sample < supersampling; ++sample) {
        auto sample_offset = vec2f{drand48(), drand48()};
        auto ray_direction =
            pixel_ray(view, resolution, vec2f{pixel} + sample_offset);
        auto view_ray = ray{view.origin, ray_direction};
        color += trace(view_ray, objects);
    }

    return color * (1.0 / supersampling);
}

void sfml_popup(camera view, scene scene)
{
    auto resolution = vec2i{400, 400};
    float upscaling = 2.5;
    sf::RenderWindow window{
        sf::VideoMode{(unsigned int)(resolution.x * upscaling),
                      (unsigned int)(resolution.y * upscaling)},
        "ray"};
    window.setPosition({1800, 0});
    sf::Event event;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> out;
    out.reserve(4 * resolution.y * resolution.x);

    for (int y(0); y < resolution.y; ++y) {
        for (int x(0); x < resolution.x; ++x) {
            auto pixel = supersample(view, resolution, scene,
                                     vec2i{x, y} - resolution * 0.5);

            auto [r, g, b] = vec3<int>(rgb_light(pixel));

            out.push_back(r);
            out.push_back(g);
            out.push_back(b);
            out.push_back(255);
        }
    }

    std::cerr << (std::chrono::high_resolution_clock::now() - t0).count() /
                     1000000.
              << "ms\n";

    sf::Texture texture;
    texture.create((unsigned int)resolution.x, (unsigned int)resolution.y);
    texture.update(out.data());
    sf::Sprite view_sprite;
    view_sprite.setTexture(texture);
    view_sprite.setScale({upscaling, upscaling});
    window.clear();
    window.draw(view_sprite);
    window.display();

    while (window.isOpen()) {
        while (window.pollEvent(event))
            if (event.type == sf::Event::Closed)
                window.close();
    }
}

void output_colored(std::ostream& out, vec3<unsigned char> rgb, std::string s)
{
    auto [r, g, b] = vec3<int>(rgb);
    out << "\033[48;2;" << r / 2 << ";" << g / 2 << ";" << b / 2 << "m";
    out << "\033[38;2;" << r << ";" << g << ";" << b << "m";
    out << s;
    out << "\033[m";
    out << "\033[m";
}

void text_output(camera view, scene scene)
{
    auto resolution = vec2i{152, 76};

    std::ostringstream out;

    for (int y(0); y < resolution.y; ++y) {
        for (int x(0); x < resolution.x; ++x) {
            auto pixel = supersample(view, resolution, scene,
                                     vec2i{x, y} - resolution * 0.5);

            std::string c = " ";
            if (pixel.length() >= 0.99)
                c = "@";
            else if (pixel.length() >= 0.75)
                c = "%";
            else if (pixel.length() >= 0.40)
                c = "*";
            else if (pixel.length() >= 0.10)
                c = ":";
            else if (pixel.length() > 0)
                c = "·";

            output_colored(out, rgb_light(pixel), c);
        }
        out << "\n";
    }

    std::cout.write(out.str().data(), out.str().size());
    fflush(stdout);
}

int main()
{
    auto view = camera{{1, 5, 0}, vec3f{-0.1, -0.1, 1}.normalized(), 35};

    auto objects = scene{
        {sphere{{-5, -4.5, 30}, 1, {0.6, 1, 0.8}}},
        {sphere{{3, -3.25, 40}, 2.5, {1, 0.2, 0.2}}},
        {sphere{{-1, 2, 60}, 8, {1, 0.70, 0.25}}},
        {sphere{{-8, 6, 45}, 1, {1, 1, 1}}},
        {sphere{{0, -1000000006., 0}, 1000000000.}},
    };

    // text_output(view, objects);
    sfml_popup(view, objects);
}
