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

const vec3f black = vec3f{};

struct camera {
    vec3f origin;
    vec3f direction;
    double fov;
};

struct ray {
    vec3f origin, direction;
    std::uint8_t depth;
};

struct point_light{
    vec3f position;
    vec3f intensity;
};

struct intersection;
struct sphere;
using object = std::variant<sphere, point_light>;
using scene = std::vector<object>;

using material = vec3f (*)(const vec3f&, const object&, const intersection&);

vec3f matte_reflect(const vec3f&, const object&, const intersection&);

struct sphere {
    vec3f center;
    double radius;
    vec3f color{1, 1, 1};
    material surface{matte_reflect};
};

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

// vec3f light_source(const vec3f& observer, const object& obj,
//                   const intersection& intersection, const scene& objects)
// {
//     return std::get<sphere>(obj).color;
// }

// vec3f mirror_reflect(const vec3f& observer, const object& obj,
//                      const intersection& intersection, const scene& objects)
// {
//     auto reflection =
//         reflect(observer.direction, intersection.surface_normal).normalized();
//     auto reflected_ray = ray{intersection.p, reflection, observer.depth};

//     return std::get<sphere>(obj).color * trace(reflected_ray, objects);
// }

ray matte_bounce(const vec3f& view, const object& obj,
                 const intersection& intersection)
{
    auto new_direction =
        intersection.p + intersection.surface_normal + random_direction();
    auto normal = (new_direction - intersection.p).normalized();
    auto diffuse_ray = ray{intersection.p, normal};
    return diffuse_ray;
}

vec3f matte_reflect(const vec3f& view, const object& obj,
                    const intersection& intersection)
{
    return std::get<sphere>(obj).color;
}

// vec3f glossy_reflect(const vec3f& observer, const object& obj,
//                      const intersection& intersection, const scene& objects)
// {
//     auto reflection
//         = reflect(observer.direction, intersection.surface_normal).normalized();
//     auto diffused
//         = intersection.p + intersection.surface_normal + random_direction();
//     auto diffusion = (diffused - intersection.p).normalized();

//     auto normal = (reflection * 0.75 + diffusion * 0.25).normalized();
//     auto gloss_ray = ray{intersection.p, normal, observer.depth};

//     return std::get<sphere>(obj).color * (trace(gloss_ray, objects) * 0.7);
// }

vec3f surface_reflect(const vec3f& view, const object& obj,
                      const intersection& intersection)
{
    return intersection.mat(view, obj, intersection);
}

std::optional<intersection> intersect(const ray&, const point_light&)
{
    return {};
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
    if (intersection_distance < -10e-10) return {};

    auto intersection_point = r.origin + r.direction * intersection_distance;
    auto surface_normal = (intersection_point - s.center).normalized();
    return {{intersection_point, surface_normal, s.surface}};
}

std::optional<intersection> intersect(const ray& ray, const object& obj)
{
    return std::visit([&ray](auto& obj) { return intersect(ray, obj); }, obj);
}

std::optional<std::pair<const object&, intersection>>
intersect(ray& ray, const scene& objects)
{
    if (ray.depth > 32) return {};
    ray.depth++;

    auto nearest = std::numeric_limits<double>::max();
    std::optional<intersection> result{};
    const object* intersected_object{};

    for (const auto& obj : objects) {
        auto intersection = intersect(ray, obj);
        if (not intersection) continue;

        auto distance = (intersection->p - ray.origin).length();
        if (nearest <= distance) continue;

        nearest = distance;
        result = {intersection};
        intersected_object = &obj;
    }

    if (result)
        return {std::pair<const object&, intersection>{*intersected_object,
                                                       *result}};
    else
        return {};
}

struct incident_light{
    vec3f light;
    vec3f normal;
};

incident_light sample_direct_lighting(const intersection& p,
                                      const scene& objects)
{
    // select one light
    auto light_count{0};
    for (auto& obj : objects) {
        light_count += std::visit(
            [](auto& obj) {
                return std::is_same_v<decltype(obj), const point_light&>;
            },
            obj);
    }
    if (light_count == 0) return {black, {}};

    light_count = int(float(light_count) * drand48()) + 1;

    const object* one_light;
    for (auto& obj : objects) {
        light_count -= std::visit(
            [](auto& obj) {
                return std::is_same_v<decltype(obj), const point_light&>;
            },
            obj);
        if (light_count <= 0) {
            one_light = &obj;
            break;
        }
    }

    auto light_source = std::get<point_light>(*one_light);
    
    // cast shadow ray
    auto towards_light = light_source.position - p.p;
    auto shadow_ray = ray{p.p, towards_light.normalized()};

    // check visibility
    auto visibility = intersect(shadow_ray, objects);
    if (visibility) {
        auto& [_, intersection] = *visibility;
        auto intersection_distance = (intersection.p - p.p).length2();
        if (intersection_distance < towards_light.length2()) return {black, {}};
    }

    // return incident light contribution
    auto L = light_source.intensity * (1. / towards_light.length2());

    L *= p.surface_normal.dot(shadow_ray.direction);

    return {L, shadow_ray.direction};
}

vec3f trace(ray view_ray, const scene& scene)
{
    vec3f light = black;
    auto remaining_light_transfer = vec3f{1, 1, 1};

    while (true) {
        auto found_intersection = intersect(view_ray, scene);
        if (not found_intersection) return light;
        auto& [obj, intersection] = *found_intersection;

        auto [direct_light, light_direction]
            = sample_direct_lighting(intersection, scene);

        direct_light = direct_light * remaining_light_transfer
                       * surface_reflect(light_direction, obj, intersection);

        auto reflectance
            = surface_reflect(view_ray.direction, obj, intersection);
        remaining_light_transfer = remaining_light_transfer * reflectance;

        light += direct_light;

        view_ray = matte_bounce(view_ray.direction, obj, intersection);
    }

    return light;
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
    return {std::pow(light.x, 1. / 2.2), std::pow(light.y, 1. / 2.2),
            std::pow(light.z, 1. / 2.2)};
}

vec3<unsigned char> rgb_light(vec3f light)
{
    auto clamp = [](auto c) { return std::min(1., std::max(0., c)); };
    light = {clamp(light.x), clamp(light.y), clamp(light.z)};
    return vec3<unsigned char>(gamma_correction(light) * 255.);
}

vec3f supersample(const camera& view, const vec2i& resolution,
                  const scene& objects, vec2i pixel)
{
    constexpr auto supersampling{16};

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
            }
            done_count++;
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
    // auto view = camera{{1, 5, 0}, vec3f{0.5918782901, -8.91237850058, 40}.normalized(), 0.000000001};
    auto view = camera{{1, 5, 0}, vec3f{-0.1, -0.1, 1}.normalized(), 30};

    auto objects = scene{
        {sphere{{-6, -5, 35}, 1, {0.6, 1, 0.8}}},
        {sphere{{3, -3.5, 40}, 2.5, {1, 0.2, 0.2}}},
        {sphere{{-1, 2, 60}, 8, {1, 0.70, 0.25}}},
        {sphere{{-8, 6, 45}, 1, {1, 1, 1}}},
        {sphere{{0, -1000000006., 0}, 1000000000., {.85, .85, .95}}},
        // {sphere{{0, 1000000000., 0}, 999999900., {.9, .95, 1},
        // light_source}},

        {point_light{{-4, -4, 45}, {50., 50., 200.}}},
        {point_light{{4, 6, 25}, {300., 50., 50.}}},
        {point_light{{0, 500, 0}, {100000., 100000., 100000.}}},
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
