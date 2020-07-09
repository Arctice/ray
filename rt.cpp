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

#include "lib/pool.hpp"

struct camera {
    vec3f origin;
    vec3f direction;
    double fov;
};

struct ray {
    vec3f origin, direction;
    std::uint8_t depth;
};

struct intersection;
struct sphere;
using object = std::variant<sphere>;
using scene = std::vector<object>;

using material = vec3f (*)(const ray&, const object&, const intersection&,
                           const scene&);

vec3f matte_diffuse(const ray&, const object&, const intersection&,
                    const scene&);

struct sphere {
    vec3f center;
    double radius;
    vec3f color{1, 1, 1};
    material surface{matte_diffuse};
};

vec3f trace(ray& view_ray, const scene& objects);

struct intersection {
    vec3f p;
    vec3f surface_normal;
    material mat;
};

vec3f random_direction()
{
    while (true) {
        auto v = vec3f{drand48(), drand48(), drand48()} * 2 - vec3f{1, 1, 1};
        if (v.length2() < 1) return v;
    }
}

vec3f reflect(const vec3f& v, const vec3f& normal)
{
    return v - normal * 2.0 * v.dot(normal);
}

vec3f light_source(const ray& observer, const object& obj,
                  const intersection& intersection, const scene& objects)
{
    return std::get<sphere>(obj).color;
}

vec3f mirror_diffuse(const ray& observer, const object& obj,
                     const intersection& intersection, const scene& objects)
{
    auto reflection =
        reflect(observer.direction, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection, observer.depth};

    return std::get<sphere>(obj).color * trace(reflected_ray, objects);
}

vec3f matte_diffuse(const ray& observer, const object& obj,
                    const intersection& intersection, const scene& objects)
{
    auto new_direction =
        intersection.p + intersection.surface_normal + random_direction();
    auto normal = (new_direction - intersection.p).normalized();
    auto diffuse_ray = ray{intersection.p, normal, observer.depth};

    auto reflectivity{0.60};
    auto transferance = std::get<sphere>(obj).color * 2.0 * reflectivity
                        * std::max(0., intersection.surface_normal.dot(normal));

    return trace(diffuse_ray, objects) * transferance;

    // return std::get<sphere>(obj).color * (trace(diffuse_ray, objects) * 0.5);
}

vec3f glossy_diffuse(const ray& observer, const object& obj,
                     const intersection& intersection, const scene& objects)
{
    auto reflection
        = reflect(observer.direction, intersection.surface_normal).normalized();
    auto diffused
        = intersection.p + intersection.surface_normal + random_direction();
    auto diffusion = (diffused - intersection.p).normalized();

    auto normal = (reflection * 0.75 + diffusion * 0.25).normalized();
    auto gloss_ray = ray{intersection.p, normal, observer.depth};

    return std::get<sphere>(obj).color * (trace(gloss_ray, objects) * 0.7);
}

std::optional<intersection> intersect(const ray& r, const sphere& s)
{
    auto r2 = s.radius * s.radius;
    auto to_sphere = (s.center - r.origin);
    auto projection = r.direction.dot(to_sphere);
    auto cast = r.direction * projection;
    auto closest_to_sphere_sq = (to_sphere - cast).length2();
    if (projection < 0 or closest_to_sphere_sq >= r2) return {};

    auto intersection_depth = std::sqrt(r2 - closest_to_sphere_sq);
    auto intersection_distance = projection - intersection_depth;
    if (intersection_distance < 0) return {};

    auto intersection_point = r.origin + r.direction * intersection_distance;
    auto surface_normal = (intersection_point - s.center).normalized();
    return {{intersection_point, surface_normal, s.surface}};
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

vec3f trace(ray& view_ray, const scene& objects)
{
    auto view_color = vec3f{};
    if (view_ray.depth > 128) return view_color;
    view_ray.depth++;

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

vec3f gamma_correction(vec3f light)
{
    return {std::sqrt(light.x), std::sqrt(light.y), std::sqrt(light.z)};
}

vec3<unsigned char> rgb_light(vec3f light)
{

    return vec3<unsigned char>(gamma_correction(light) * 255.);
}

vec3f supersample(const camera& view, const vec2i& resolution,
                  const scene& objects, vec2i pixel)
{
    constexpr auto supersampling{12};

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
    auto resolution = vec2i{800, 800};
    // float scaling = 1;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> out;
    out.resize(4 * resolution.y * resolution.x);

    thread_pool pool;
    std::atomic<int> done_count{};

    for (int y(0); y < resolution.y; ++y) {
        pool.enqueue_work([y, &done_count, &out, &view, &resolution, &scene]() {
            for (int x(0); x < resolution.x; ++x) {
                auto pixel = supersample(view, resolution, scene,
                                         vec2i{x, y} - resolution * 0.5);

                auto [r, g, b] = vec3<int>(rgb_light(pixel));

                out[4 * (y * resolution.x + x)] = r;
                out[4 * (y * resolution.x + x) + 1] = g;
                out[4 * (y * resolution.x + x) + 2] = b;
                out[4 * (y * resolution.x + x) + 3] = 255;
            }done_count++;
            });
        }

        while (done_count != resolution.y) {};

        std::cerr << (std::chrono::high_resolution_clock::now() - t0).count()
                         / 1000000.
                  << "ms\n";

        sf::Image img;
        img.create((unsigned int)(resolution.x), (unsigned int)(resolution.y),
                   out.data());
        img.saveToFile("out.png");

        // sf::Texture texture;
        // texture.create((unsigned int)resolution.x, (unsigned
        // int)resolution.y); texture.update(out.data()); sf::Sprite
        // view_sprite; view_sprite.setTexture(texture);
        // view_sprite.setScale({scaling, scaling});

        // sf::RenderWindow window{
        //     sf::VideoMode{(unsigned int)(resolution.x * scaling),
        //                   (unsigned int)(resolution.y * scaling)},
        //     "ray"};
        // window.setPosition({1800, 0});

        // window.clear();
        // window.draw(view_sprite);
        // window.display();

        // sf::Event event;
        // while (window.isOpen()) {
        //     while (window.pollEvent(event))
        //         if (event.type == sf::Event::Closed)
        //             window.close();
        // }
    }

    void output_colored(std::ostream & out, vec3<unsigned char> rgb,
                        std::string s)
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
    // auto view = camera{{1, 5, 0}, vec3f{0.5918782901, -8.91237850058, 40}.normalized(), 0.000000001};
    auto view = camera{{1, 5, 0}, vec3f{-0.1, -0.1, 1}.normalized(), 34};

    auto objects = scene{
        {sphere{{-6, -5, 35}, 1, {0.6, 1, 0.8}, mirror_diffuse}},
        {sphere{{3, -3.5, 40}, 2.5, {1, 0.2, 0.2}, mirror_diffuse}},
        {sphere{{-1, 2, 60}, 8, {1, 0.70, 0.25}, glossy_diffuse}},
        {sphere{{-8, 6, 45}, 1, {1, 1, 1}, matte_diffuse}},
        {sphere{{-3, -5.5, 42}, .2, {1, 1, 1}, light_source}},
        {sphere{{0, -1000000006., 0},
                1000000000.,
                {.85, .85, .95},
                glossy_diffuse}},
        {sphere{{0, 1000000000., 0}, 999999900., {.9, .95, 1}, light_source}},
    };

    // text_output(view, objects);
    sfml_popup(view, objects);
}

// std::optional<intersection> intersect(const ray& r, const sphere& s)
// {
//     // R t = o + v*t;
//     // (x - sx)² + (y - sy)² + (z - sz)² = r²;
//     // b = o - s;
//     // (bx + vx*t)² + (...) - r2 = 0;
//     // (bx² + by² + bz² - r2)
//     //    + 2t(bx·vx + by·vy + bz·vz)
//     //    + (vx²t² + vy²t² + vz²t²);
//     // t²(v·v) + 2t(b·v) + (b·b) - r² = 0
//     auto r2 = s.radius * s.radius;
//     auto v = r.direction;
//     auto b = r.origin - s.center;
//     // At² + Bt + C = 0
//     auto A = v.dot(v);
//     auto B = 2 * b.dot(v);
//     auto C = b.dot(b) - r2;
//     auto q = B * B - 4 * A * C;
//     if (q < 0) return {};
//     q = std::sqrt(q);
//     auto t = (-B - q) / 2 * A;
//     if (t < 0) return {};
//     auto intersection_point = r.origin + r.direction * t;
//     auto surface_normal = (intersection_point - s.center).normalized();
//     return {{intersection_point, surface_normal, s.surface}};
// }
