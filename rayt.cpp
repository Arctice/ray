#include <memory>
#include <vector>
#include <unordered_map>
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

using Float = double;
using vec2s = vec2<Float>;
using vec3s = vec3<Float>;

Float random_uniform() {
    thread_local static unsigned short int xsubi[3];
    return erand48(xsubi);
}

const vec3s black{};
template <class T> constexpr float epsilon_ = 7e-7;
template <> constexpr double epsilon_<double> = 10e-10;
constexpr Float epsilon = epsilon_<Float>;

struct camera {
    vec3s origin;
    vec3s direction;
    vec3s inverse_direction;
    vec2s world_size;
    vec2i resolution;
    int supersampling;
};

struct ray {
    vec3s origin, direction, inverse_direction;

    vec3s distance(Float t) const { return origin + direction * t; }
};

struct point_light {
    vec3s position;
    vec3s intensity;
};

struct cell;
struct triangle;
using object = std::variant<std::monostate, point_light, triangle, cell>;

struct reflection {
    ray reflected;
    vec3s light_transfer;
};

struct intersection;

struct material {
    vec3s (*reflect)(const vec3s&, const vec3s&, const object&,
                     const intersection&);
    reflection (*scatter)(const vec3s&, const object&, const intersection&);
};

extern material matte;

struct triangle {
    vec3s A, B, C;
    vec3s color{1, 1, 1};
    material* surface{&matte};
};

struct cell {
    vec3i grid_pos;
    vec3s color{1, 1, 1};
    material* surface{&matte};
};

struct grid {
    grid(vec3i size) : size(size), cells() {
        cells.resize(size.x * size.y * size.z, {std::monostate{}});
    }
    vec3i size;
    std::vector<object> cells;

    const object& operator[](vec3i v) const {
        auto at = v.y * size.z * size.x + v.x * size.z + v.z;
        return cells[at];
    }

    object& operator[](vec3i v) {
        auto at = v.y * size.z * size.x + v.x * size.z + v.z;
        return cells[at];
    }

    bool inbounds(vec3i v) const {
        return v.x >= 0 and v.y >= 0 and v.z >= 0 and v.x < size.x and
               v.z < size.y and v.y < size.z;
    }
};

struct scene {
    std::vector<object> lights;
    std::vector<object> objects;
    grid grid;
};

struct intersection {
    vec3s p;
    vec3s surface_normal;
    material* material;
};

vec3s surface_color(const object& obj) {
    return std::visit(
        [](auto& obj) {
            if constexpr (std::is_same_v<decltype(obj), const triangle&> ||
                          std::is_same_v<decltype(obj), const cell&>)
                return obj.color;
            else
                return vec3s{};
        },
        obj);
}

vec3s random_direction() {
    while (true) {
        auto v =
            vec3s{random_uniform(), random_uniform(), random_uniform()} * 2 -
            vec3s{1, 1, 1};
        if (v.length2() < 1)
            return v;
    }
}

vec3s reflect(const vec3s& v, const vec3s& normal) {
    return v - normal * 2.0 * v.dot(normal);
}

vec3s matte_reflect(const vec3s& view, const vec3s& light, const object& obj,
                    const intersection& intersection) {
    return surface_color(obj);
}

reflection matte_scatter(const vec3s& view, const object& obj,
                         const intersection& intersection) {
    auto new_direction =
        intersection.p + intersection.surface_normal + random_direction();
    auto normal = (new_direction - intersection.p).normalized();
    auto diffuse_ray = ray{intersection.p, normal};
    return {diffuse_ray, matte_reflect(view, {}, obj, intersection)};
}

vec3s specular_reflect(const vec3s& view, const vec3s& light, const object& obj,
                       const intersection& intersection) {
    return {};
}

reflection mirror_scatter(const vec3s& view, const object& obj,
                          const intersection& intersection) {
    auto reflection = reflect(view, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection};
    auto cosθ = reflection.dot(intersection.surface_normal);

    return {reflected_ray, surface_color(obj) / cosθ};
}

constexpr Float dielectric_refraction = 1.52; // crown glass
// constexpr Float dielectric_refraction = 1.333; // wat

Float fresnel_dielectric(Float cosθi, Float ηi, Float ηt) {
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

reflection dielectric_scatter(const vec3s& view, const object& obj,
                              const intersection& intersection) {
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

vec3s sqrt(vec3s v) { return {std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z)}; }

reflection conductor_scatter(const vec3s& view, const object& obj,
                             const intersection& intersection) {
    auto reflection = reflect(view, intersection.surface_normal).normalized();
    auto reflected_ray = ray{intersection.p, reflection};

    auto cosθ = reflection.dot(intersection.surface_normal);

    auto ηi = 1.;
    auto ηt = 0.32393;
    auto k = 2.5972;

    vec3s η = ηt / ηi;
    vec3s ηk = k / ηi;

    Float cos2θ = cosθ * cosθ;
    Float sin2θ = 1. - cosθ;

    vec3s η2 = η * η;
    vec3s ηk2 = ηk * ηk;

    vec3s t0 = η2 - ηk2 - sin2θ;
    vec3s a2plusb2 = sqrt(t0 * t0 + η2 * ηk2 * 4);
    vec3s t1 = a2plusb2 + cos2θ;
    vec3s a = sqrt((a2plusb2 + t0) * 0.5);
    vec3s t2 = a * (2. * cosθ);

    vec3s rs = (t1 - t2) / (t1 + t2);
    vec3s t3 = a2plusb2 * cos2θ + sin2θ * sin2θ;
    vec3s t4 = t2 * sin2θ;
    vec3s rp = rs * (t3 - t4) / (t3 + t4);

    auto Fr = (rp + rs) * .5;
    Fr /= std::abs(cosθ);

    auto light = surface_color(obj) * Fr;
    return {reflected_ray, light};
}

reflection transmission_scatter(const vec3s& view, const object& obj,
                                const intersection& intersection) {
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
    vec3s refraction =
        view * η + intersection.surface_normal * (η * cosθ - cosθt);

    auto transmission = surface_color(obj);
    auto Fr = fresnel_dielectric(cosθt, ηi, ηt);
    auto light = transmission * (vec3s(1) - Fr);
    light /= std::abs(cosθ);

    auto refracted_ray = ray{intersection.p, refraction};

    return {refracted_ray, light};
}

reflection fresnel_scatter(const vec3s& view, const object& obj,
                           const intersection& intersection) {
    auto cosθ = (view * -1.).dot(intersection.surface_normal);
    auto ηi = 1.;
    auto ηt = dielectric_refraction;
    auto F = fresnel_dielectric(cosθ, ηi, ηt);
    if (random_uniform() < F)
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

vec3s surface_reflect(const vec3s& view, const vec3s& light, const object& obj,
                      const intersection& intersection) {
    return intersection.material->reflect(view, light, obj, intersection);
}

reflection surface_scatter(const vec3s& view, const object& obj,
                           const intersection& intersection) {
    return intersection.material->scatter(view, obj, intersection);
}

std::optional<intersection> intersect(const ray& ray, const triangle& V) {
    auto BA = V.B - V.A;
    auto CA = V.C - V.A;
    auto n = BA.cross(CA);
    auto q = ray.direction.cross(CA);
    auto a = BA.dot(q);
    if (n.dot(ray.direction) >= 0 or std::abs(a) <= epsilon)
        return {};

    auto s = (ray.origin - V.A) / a;
    auto r = s.cross(BA);

    auto b = vec3s{s.dot(q), r.dot(ray.direction), 0};
    b.z = 1.0 - b.x - b.y;
    if (b.x < 0. or b.y < 0. or b.z < 0.)
        return {};

    auto t = CA.dot(r);
    if (t < 0.)
        return {};

    auto isect_p = ray.distance(t);
    return {{isect_p, n.normalized(), V.surface}};
}

auto intersect_plane(const ray& ray, const vec3s& plane_normal,
                     const vec3s& plane_point) {
    // (Ro+Rd*t)·n = 0
    // Ro·n / Rd·n = t
    return (plane_point - ray.origin).dot(plane_normal) /
           ray.direction.dot(plane_normal);
}

std::optional<intersection> intersect_floor(const ray& ray, const cell& S) {
    auto tile = S.grid_pos;
    auto t = (tile.y - ray.origin.y) * ray.inverse_direction.y;
    auto hit = ray.origin + ray.direction * t;
    auto cast = hit - vec3s{tile};
    if (t <= epsilon || cast.x < 0 || cast.x > 1 || cast.z < 0 || cast.z > 1)
        return {};
    Float sign = ray.origin.y > tile.y ? 1. : -1.;
    return {{hit, {0, sign, 0}, S.surface}};
}

std::optional<intersection> intersect(const ray& ray, const cell& S) {
    return intersect_floor(ray, S);
}

std::optional<intersection> intersect(const ray&, const point_light&) {
    return {};
}

std::optional<intersection> intersect(const ray&, const std::monostate&) {
    return {};
}

std::optional<intersection> intersect_object(const ray& ray,
                                             const object& obj) {
    return std::visit([&ray](auto& obj) { return intersect(ray, obj); }, obj);
}

struct bounding_box {
    vec3s min, max;

    bool intersect(const ray& ray) const {
        Float t0 = 0., t1 = 10.e10;
        auto z = ray.origin * ray.inverse_direction;
        for (auto d{0}; d < 3; ++d) {
            auto a = min[d] * ray.inverse_direction[d] - z[d];
            auto b = max[d] * ray.inverse_direction[d] - z[d];
            if (a > b)
                std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
        }

        return t0 <= t1;
    }

    bool intersect(const bounding_box& other) {
        return std::max(min.x, other.min.x) <= std::min(max.x, other.max.x) &&
               std::max(min.y, other.min.y) <= std::min(max.y, other.max.y) &&
               std::max(min.z, other.min.z) <= std::min(max.z, other.max.z);
    }

    bounding_box operator|(const bounding_box& other) const {
        return {vec3s{std::min(min.x, other.min.x),
                      std::min(min.y, other.min.y),
                      std::min(min.z, other.min.z)},
                vec3s{
                    std::max(max.x, other.max.x),
                    std::max(max.y, other.max.y),
                    std::max(max.z, other.max.z),
                }};
    }

    bounding_box operator|(const vec3s& rhs) const {
        return {vec3s{std::min(min.x, rhs.x), std::min(min.y, rhs.y),
                      std::min(min.z, rhs.z)},
                vec3s{
                    std::max(max.x, rhs.x),
                    std::max(max.y, rhs.y),
                    std::max(max.z, rhs.z),
                }};
    }

    vec3s centroid() const { return (max + min) / 2; };
};

bounding_box point_bounds(const vec3s& a, const vec3s& b) {
    return {{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)},
            {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}};
}

bounding_box object_bounds(const triangle& V) {
    vec3s min{std::min(std::min(V.A.x, V.B.x), V.C.x),
              std::min(std::min(V.A.y, V.B.y), V.C.y),
              std::min(std::min(V.A.z, V.B.z), V.C.z)};
    vec3s max{std::max(std::max(V.A.x, V.B.x), V.C.x),
              std::max(std::max(V.A.y, V.B.y), V.C.y),
              std::max(std::max(V.A.z, V.B.z), V.C.z)};
    return {min, max};
}

std::optional<std::pair<const object&, intersection>>
intersect_brute(ray& ray, const scene& scene) {
    ray.inverse_direction = vec3s{1} / ray.direction;
    std::optional<intersection> result{};
    const object* intersected_object{};
    auto nearest = std::numeric_limits<Float>::max();
    for (const auto& obj : scene.grid.cells) {
        auto intersection = intersect_object(ray, obj);
        if (not intersection)
            continue;
        auto distance = (intersection->p - ray.origin).length2();
        if (nearest <= distance)
            continue;
        nearest = distance;
        result = {intersection};
        intersected_object = &obj;
    }
    if (result)
        return {std::pair<const object&, intersection>{*intersected_object,
                                                       *result}};
    return {};
}

std::optional<std::pair<const object&, intersection>>
intersect(ray& ray, const scene& scene) {
    auto bounds = bounding_box{{0.}, vec3s{scene.grid.size}};
    if (!bounds.intersect(ray))
        return {};

    auto adv = vec3s{std::abs(ray.inverse_direction.x),
                     std::abs(ray.inverse_direction.y),
                     std::abs(ray.inverse_direction.z)};

    vec3s base{};
    for (auto d{0}; d < 3; ++d)
        base[d] = ray.direction[d] < 0 ? std::floor(ray.origin[d])
                                       : std::ceil(ray.origin[d]);
    base = (base - ray.origin) * ray.inverse_direction;

    vec3s t{base};
    vec3i steps{};

    auto speculated{ray};
    while (!scene.grid.inbounds(vec3i{speculated.origin})) {
        auto prev_t = std::min(std::min(t.x, t.z), t.y);
        speculated.origin = ray.distance(prev_t + epsilon);
        for (auto d{0}; d < 3; ++d)
            while (t[d] <= prev_t)
                t[d] = ++steps[d] * adv[d] + base[d];
    }

    while (scene.grid.inbounds(vec3i{speculated.origin})) {
        auto& obj = scene.grid[vec3i{speculated.origin}];
        auto hit = intersect(ray, std::get<cell>(obj));
        if (hit)
            return {{obj, *hit}};

        auto prev_t = std::min(std::min(t.x, t.z), t.y);
        speculated.origin = ray.distance(prev_t + epsilon);

        for (auto d{0}; d < 3; ++d)
            while (t[d] <= prev_t)
                t[d] = ++steps[d] * adv[d] + base[d];
    }

    return {};
}

struct incident_light {
    vec3s light;
    vec3s normal;
};

incident_light sample_direct_lighting(const intersection& p,
                                      const scene& scene) {
    // select one light
    auto light_count{scene.lights.size()};
    if (light_count == 0)
        return {black, {}};

    light_count = int(float(light_count) * random_uniform()) + 1;

    const object* one_light;
    for (auto& obj : scene.lights) {
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
    shadow_ray.inverse_direction = vec3s{1} / shadow_ray.direction;

    // check visibility
    auto visibility = intersect(shadow_ray, scene);
    if (visibility) {
        auto& [_, intersection] = *visibility;
        auto intersection_distance = (intersection.p - p.p).length2();
        if (intersection_distance < towards_light.length2())
            return {black, {}};
    }

    // return incident light contribution
    auto L = light_source.intensity * (1. / towards_light.length2());

    L *= p.surface_normal.dot(shadow_ray.direction);

    return {L, shadow_ray.direction};
}

vec3s trace(ray view_ray, const scene& scene) {
    auto depth{0};
    auto rr_threshold = .1;

    auto remaining_light_transfer = vec3s{1};
    vec3s light = black;
    const vec3s ambient = vec3s{.12};

    while (++depth < 12) {
        auto found_intersection = intersect(view_ray, scene);
        if (not found_intersection) {
            light += remaining_light_transfer * ambient;
            break;
        }
        auto& [obj, intersection] = *found_intersection;

        auto [direct_light, light_direction] =
            sample_direct_lighting(intersection, scene);

        auto light_contribution = surface_reflect(
            view_ray.direction, light_direction, obj, intersection);
        direct_light *= remaining_light_transfer * light_contribution;
        light += direct_light;

        auto [next_ray, transmission] =
            surface_scatter(view_ray.direction, obj, intersection);
        transmission *=
            std::abs(intersection.surface_normal.dot(next_ray.direction));
        remaining_light_transfer *= transmission;

        // roulette halt
        auto maxb = std::max(
            remaining_light_transfer.x,
            std::max(remaining_light_transfer.y, remaining_light_transfer.z));
        if (maxb < rr_threshold) {
            auto q = std::max(.1, 1. - maxb);
            if (random_uniform() < q)
                break;
            remaining_light_transfer /= 1. - q;
        }

        view_ray = next_ray;
        view_ray.inverse_direction = vec3s{1} / view_ray.direction;
    }

    return light;
}

ray orthogonal_ray(camera view, vec2s coordinates) {
    auto plane_a = view.direction.cross({0, 1, 0}).normalized();
    auto plane_b = view.direction.cross(plane_a).normalized();
    auto offset = coordinates * view.world_size;
    return {view.origin - plane_a * offset.x + plane_b * offset.y,
            view.direction, view.inverse_direction};
}

vec3s supersample(const camera& view, const scene& objects, vec2i pixel) {
    auto color{black};

    for (int sample{}; sample < view.supersampling; ++sample) {
        auto sample_offset = vec2s{random_uniform(), random_uniform()};
        auto ray = orthogonal_ray(view, (vec2s{pixel} + sample_offset) /
                                            vec2s{view.resolution});
        color += trace(ray, objects);
    }

    return color * (1.0 / view.supersampling);
}

vec3s gamma_correction(vec3s light) {
    return {(Float)std::pow(light.x, 1. / 2.2),
            (Float)std::pow(light.y, 1. / 2.2),
            (Float)std::pow(light.z, 1. / 2.2)};
}

vec3<unsigned char> rgb_light(vec3s light) {
    auto clamp = [](auto c) { return std::clamp<Float>(c, 0, 1); };
    light = {clamp(light.x), clamp(light.y), clamp(light.z)};
    return vec3<unsigned char>(gamma_correction(light) * 255.);
}

void img_draw(camera view, scene scene) {
    auto resolution = view.resolution;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> out{};
    out.resize(4 * resolution.y * resolution.x);

// #pragma omp parallel for schedule(monotonic : dynamic)
    for (int y = 0; y < resolution.y; ++y) {
        for (int x = 0; x < resolution.x; ++x) {
            auto pixel =
                supersample(view, scene, vec2i{x, y} - resolution * 0.5);

            auto [r, g, b] = vec3<int>(rgb_light(pixel));
            out[4 * (y * resolution.x + x)] = r;
            out[4 * (y * resolution.x + x) + 1] = g;
            out[4 * (y * resolution.x + x) + 2] = b;
            out[4 * (y * resolution.x + x) + 3] = 255;
        }
    }

    std::cerr << (std::chrono::high_resolution_clock::now() - t0).count() /
                     1000000.
              << "ms\n";

    sf::Image img;
    img.create((unsigned int)(resolution.x), (unsigned int)(resolution.y),
               out.data());
    img.saveToFile("out.png");
}

int main() {
    srand48(std::chrono::nanoseconds(
                std::chrono::high_resolution_clock().now().time_since_epoch())
                .count());
    srand48(0);

    grid tiles{{8}};
    for (int x{0}; x < tiles.size.x; ++x) {
        for (int y{0}; y < tiles.size.x; ++y) {
            for (int z{0}; z < tiles.size.z; ++z) {
                auto pos = vec3i{x, y, z};
                auto color =
                    vec3s{random_uniform(), random_uniform(), random_uniform()}
                        .normalized();
                auto material = &matte;
                tiles[pos] = cell{pos, color, material};
            }
        }
    }

    auto direction = vec3s{.95, -.4, .55}.normalized();
    auto origin = direction * -tiles.size.x + vec3s{tiles.size / 2};
    std::vector<object> lights = {
        point_light{vec3s{-1, 2, -2}.normalized() * 20, 500}};

    img_draw(
        camera{.origin = origin,
               .direction = direction,
               .inverse_direction = vec3s{1} / direction,
               .world_size = vec2s{vec2i{tiles.size.x, tiles.size.y}} * 1.5,
               .resolution = {800, 800},
               .supersampling = 3},
        {lights, {}, tiles});
}
