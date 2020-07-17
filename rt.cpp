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
auto epsilon = 10e-10;

struct camera {
    vec3f origin;
    vec3f direction;
    double fov;
};

struct ray {
    vec3f origin, direction;

    vec3f distance(double t) const { return origin + direction * t; }
};

struct point_light {
    vec3f position;
    vec3f intensity;
};

struct intersection;
struct sphere;
struct triangle;
using object = std::variant<triangle, sphere, point_light>;
using scene = std::vector<object>;

struct reflection {
    ray reflected;
    vec3f light_transfer;
};

struct material {
    vec3f (*reflect)(const vec3f&, const vec3f&, const object&,
                     const intersection&);
    reflection (*scatter)(const vec3f&, const object&, const intersection&);
};

extern material matte;

struct sphere {
    vec3f center;
    double radius;
    vec3f color{1, 1, 1};
    material* surface{&matte};
};

struct triangle {
    vec3f A, B, C;
    vec3f color{1, 1, 1};
    material* surface{&matte};
};

struct intersection {
    vec3f p;
    vec3f surface_normal;
    material* material;
};


vec3f surface_color(const object& obj)
{
    return std::visit(
        [](auto& obj) {
            if constexpr (!std::is_same_v<decltype(obj), const point_light&>)
                return obj.color;
            else
                return vec3f{};
        },
        obj);
}


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

vec3f matte_reflect(const vec3f& view, const vec3f& light, const object& obj,
                    const intersection& intersection)
{
    return surface_color(obj);
}

reflection matte_scatter(const vec3f& view, const object& obj,
                         const intersection& intersection)
{
    auto new_direction
        = intersection.p + intersection.surface_normal + random_direction();
    auto normal = (new_direction - intersection.p).normalized();
    auto diffuse_ray = ray{intersection.p, normal};
    return {diffuse_ray, matte_reflect(view, {}, obj, intersection)};
}

vec3f specular_reflect(const vec3f& view, const vec3f& light, const object& obj,
                       const intersection& intersection)
{
    return {};
}

reflection mirror_scatter(const vec3f& view, const object& obj,
                          const intersection& intersection)
{
    auto reflection = reflect(view, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection};
    auto cosθ = reflection.dot(intersection.surface_normal);

    return {reflected_ray, surface_color(obj) / cosθ};
}

constexpr double dielectric_refraction = 1.52; // crown glass
// constexpr double dielectric_refraction = 1.333; // wat

double fresnel_dielectric(double cosθi, double ηi, double ηt)
{
    // ηi and ηt are the incident and transmitted
    // indices of medium refraction
    auto sinθi = std::sqrt(1.0 - cosθi * cosθi);
    auto sinθt = ηi / ηt * sinθi;
    // handle total internal reflection if sinθ > 1
    auto cosθt = std::sqrt(1.0 - sinθt * sinθt);

    auto r_par = (ηt * cosθi - ηi * cosθt) / (ηt * cosθi + ηi * cosθt);
    auto r_per = (ηi * cosθi - ηt * cosθt) / (ηi * cosθi + ηt * cosθt);

    auto Fr = (r_par * r_par + r_per * r_per) / 2.;
    return Fr;
}

reflection dielectric_scatter(const vec3f& view, const object& obj,
                              const intersection& intersection)
{
    auto reflection = reflect(view, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection};

    // dielectric fresnel reflectance
    // reflectance by light polarization
    // r∥ = Ƞₜ cosθᵢ - Ƞᵢ cosθₜ / Ƞₜ cosθᵢ + Ƞᵢ cosθₜ
    // r⟂ = Ƞᵢ cosθᵢ - Ƞₜ cosθₜ / Ƞᵢ cosθᵢ + Ƞₜ cosθₜ
    // fresnel reflectance for unpolarized light
    // Fᵣ = (r∥² + r⟂²) / 2
    // total energy transmitted = 1 - Fᵣ

    auto ηi = 1.;
    auto ηt = dielectric_refraction;
    auto cosθi = reflection.dot(intersection.surface_normal);
    auto Fr = fresnel_dielectric(cosθi, ηi, ηt);
    Fr /= std::abs(cosθi);

    auto light = surface_color(obj) * Fr;
    return {reflected_ray, light};
}

vec3f sqrt(vec3f v) { return {std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z)}; }

reflection conductor_scatter(const vec3f& view, const object& obj,
                             const intersection& intersection)
{
    auto reflection = reflect(view, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection};

    auto cosθ = reflection.dot(intersection.surface_normal);

    auto ηi = 1.;
    auto ηt = 0.32393;
    auto k = 2.5972;

    vec3f η = ηt / ηi;
    vec3f ηk = k / ηi;

    double cos2θ = cosθ * cosθ;
    double sin2θ = 1. - cosθ;

    vec3f η2 = η * η;
    vec3f ηk2 = ηk * ηk;

    vec3f t0 = η2 - ηk2 - sin2θ;
    vec3f a2plusb2 = sqrt(t0 * t0 + η2 * ηk2 * 4);
    vec3f t1 = a2plusb2 + cos2θ;
    vec3f a = sqrt((a2plusb2 + t0) * 0.5f);
    vec3f t2 = a * (2. * cosθ);

    vec3f rs = (t1 - t2) / (t1 + t2);
    vec3f t3 = a2plusb2 * cos2θ + sin2θ * sin2θ;
    vec3f t4 = t2 * sin2θ;
    vec3f rp = rs * (t3 - t4) / (t3 + t4);

    auto Fr = (rp + rs) * .5;
    Fr /= std::abs(cosθ);

    auto light = surface_color(obj) * Fr;
    return {reflected_ray, light};
}

reflection transmission_scatter(const vec3f& view, const object& obj,
                                const intersection& intersection)
{
    auto cosθ = (view * -1.).dot(intersection.surface_normal);
    bool entering = 0 < cosθ;

    auto η = dielectric_refraction;

    auto ηi = entering ? 1.0 : η;
    auto ηt = entering ? η : 1.0;

    η = ηi / ηt;
    auto sin2θi = 1. - cosθ * cosθ;
    auto sin2θt = η * η * sin2θi;
    if (sin2θt >= 1) {
        // total internal reflection
        return {{}, {}};
    }
    auto cosθt = std::sqrt(1. - sin2θt);
    vec3f refraction
        = view * η + intersection.surface_normal * (η * cosθ - cosθt);

    auto transmission = surface_color(obj);
    auto Fr = fresnel_dielectric(cosθt, ηi, ηt);
    auto light = transmission * (vec3f(1) - Fr);
    light /= std::abs(cosθ);

    auto refracted_ray = ray{intersection.p, refraction};

    return {refracted_ray, light};
}

reflection fresnel_scatter(const vec3f& view, const object& obj,
                           const intersection& intersection)
{
    auto cosθ = (view * -1.).dot(intersection.surface_normal);
    auto ηi = 1.;
    auto ηt = dielectric_refraction;
    auto F = fresnel_dielectric(cosθ, ηi, ηt);
    if (drand48() < F)
        return dielectric_scatter(view, obj, intersection);
    else
        return transmission_scatter(view, obj, intersection);
}

material matte{matte_reflect, matte_scatter};
material mirror{specular_reflect, mirror_scatter};
material specular_dielectric{specular_reflect, dielectric_scatter};
material specular_conductor{specular_reflect, conductor_scatter};
material specular_transmissive{specular_reflect, transmission_scatter};
material specular_fresnel{specular_reflect, fresnel_scatter};

vec3f surface_reflect(const vec3f& view, const vec3f& light, const object& obj,
                      const intersection& intersection)
{
    return intersection.material->reflect(view, light, obj, intersection);
}

reflection surface_scatter(const vec3f& view, const object& obj,
                           const intersection& intersection)
{
    return intersection.material->scatter(view, obj, intersection);
}

std::optional<intersection> intersect(const ray& R, const triangle& V)
{
    auto BA = V.B - V.A;
    auto CA = V.C - V.A;
    auto n = BA.cross(CA);
    auto q = R.direction.cross(CA);
    auto a = BA.dot(q);
    if (n.dot(R.direction) >= 0 or std::abs(a) <= epsilon) return {};

    auto s = (R.origin - V.A) / a;
    auto r = s.cross(BA);

    auto b = vec3f{s.dot(q), r.dot(R.direction), 0};
    b.z = 1.0 - b.x - b.y;
    if (b.x < 0. or b.y < 0. or b.z < 0.) return {};

    auto t = CA.dot(r);
    if (t < 0.) return {};

    auto isect_p = R.distance(t);
    return {{isect_p, n.normalized(), V.surface}};
}

std::optional<intersection> intersect(const ray& r, const sphere& s)
{
    auto r2 = s.radius * s.radius;
    auto to_sphere = (s.center - r.origin);
    auto projection = r.direction.dot(to_sphere);
    auto cast = r.direction * projection;
    auto closest_to_sphere_sq = (to_sphere - cast).length2();
    if (projection < 0 or closest_to_sphere_sq >= r2) return {};

    auto inside = (to_sphere.length2() - 10e-10 <= r2);

    auto intersection_depth = std::sqrt(r2 - closest_to_sphere_sq);
    auto intersection_distance
        = projection - intersection_depth * (inside ? -2 : 1);
    if (intersection_distance < -10e-10) return {};

    auto intersection_point = r.distance(intersection_distance);
    auto surface_normal
        = (intersection_point - s.center).normalized() * (inside ? -1 : 1);
    return {{intersection_point, surface_normal, s.surface}};
}

std::optional<intersection> intersect(const ray&, const point_light&)
{
    return {};
}

std::optional<intersection> intersect_object(const ray& ray, const object& obj)
{
    return std::visit([&ray](auto& obj) { return intersect(ray, obj); }, obj);
}

std::optional<std::pair<const object&, intersection>>
intersect(const ray& ray, const scene& objects)
{
    auto nearest = std::numeric_limits<double>::max();
    std::optional<intersection> result{};
    const object* intersected_object{};

    for (const auto& obj : objects) {
        auto intersection = intersect_object(ray, obj);
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

struct incident_light {
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
    auto depth{0};
    auto rr_threshold = .1;

    auto remaining_light_transfer = vec3f{1, 1, 1};
    vec3f light = black;

    while (++depth < 12) {
        auto found_intersection = intersect(view_ray, scene);
        if (not found_intersection) break;
        auto& [obj, intersection] = *found_intersection;

        auto [direct_light, light_direction]
            = sample_direct_lighting(intersection, scene);

        auto light_contribution = surface_reflect(
            view_ray.direction, light_direction, obj, intersection);
        direct_light *= remaining_light_transfer * light_contribution;

        auto [next_ray, transmission]
            = surface_scatter(view_ray.direction, obj, intersection);
        transmission
            *= std::abs(intersection.surface_normal.dot(next_ray.direction));
        remaining_light_transfer *= transmission;

        light += direct_light;
        view_ray = next_ray;

        auto maxb = std::max(
            remaining_light_transfer.x,
            std::max(remaining_light_transfer.y, remaining_light_transfer.z));
        if (maxb < rr_threshold) {
            auto q = std::max(.1, 1. - maxb);
            if (drand48() < q) break;
            remaining_light_transfer /= 1. - q;
        }
    }

    return light;
}

vec3f lens_ray(camera view, vec2f coordinates)
{
    // distance to imaginary frustrum
    // auto α = 1.;
    // half of the fov angle
    auto θ = view.fov * pi / 180.;
    // pixel angle size
    auto px_θ = vec2f{θ * coordinates.x, -θ * coordinates.y};

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

vec3f perspective_ray(camera view, vec2f coords)
{
    auto A = lens_ray(view, {0, -.5});
    auto B = lens_ray(view, {0, .5});
    auto C = lens_ray(view, {-.5, 0});
    auto D = lens_ray(view, {.5, 0});

    auto xs = D - C;
    auto ys = B - A;

    auto px = xs * (coords.x) + ys * (coords.y);
    auto ray = (view.direction + px).normalized();

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
    constexpr auto supersampling{12};

    vec3f color{};

    for (int sample{}; sample < supersampling; ++sample) {
        auto sample_offset = vec2f{drand48(), drand48()};
        auto ray_direction = perspective_ray(
            view, (vec2f{pixel} + sample_offset) / vec2f{resolution});
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
    matte.reflect = matte_reflect;
    matte.scatter = matte_scatter;

    auto view = camera{{0, 3, 25}, vec3f{-.05, -0.2, 1}.normalized(), 72};
    auto objects = scene{
        {sphere{{-4.5, -4.8, 40}, 1.2, {0.42, 1, 0.18}, &matte}},
        {sphere{{3, -3.5, 39}, 2.5, {1, 0.08, 0.3}, &mirror}},
        {sphere{{-1, 2, 60}, 8, {.83, .686, .21}, &specular_conductor}},
        {sphere{{-5, -3, 47}, 3., {.05, .6, .8}, &specular_fresnel}},
        {sphere{{-8.25, -4, 51}, 2, {1, .3, 0}, &matte}},
        {sphere{{-8, 6, 45}, 1, {1, 1, 1}, &matte}},
        {triangle{{6, -2, 54}, {11, 1, 52}, {12, -6.2, 57}, {.4, .5, .6}}},

        {sphere{{0, -1000000006., 0}, 1000000000., {.9, .9, .9}, &matte}},
        {sphere{{0, 0, 10000110}, 10000000., {.9, .9, .9}, &matte}},
        {sphere{{0, 0, -10000000}, 10000000., {.9, .9, .9}, &matte}},
        {sphere{{10000050, 0, 0}, 10000000., {.9, .9, .9}, &matte}},
        {sphere{{-10000050, 0, 0}, 10000000., {.9, .9, .9}, &matte}},
        {sphere{{0, 10000030, 0}, 10000000., {.9, .9, .9}, &matte}},

        {point_light{{-30, 10, 75}, {1000., 90, 1000.}}},
        {point_light{{-2, 10, 12}, {90, 1000., 1000.}}},
        {point_light{{25, 10, 75}, {1000., 1000., 90}}},
    };

    // text_output(view, objects);
    sfml_popup(view, objects);
}
