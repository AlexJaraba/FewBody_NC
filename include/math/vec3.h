#pragma once

#include <cmath>

/* ==============================================================================

    Small 3D vector type used throughout FewBodyNC.

    Vec3 replaces repeated std::vector<double>(3) allocations and 
    provides lightweight vector arithmetic for positions, velocities, momenta, forces, and tangent variables.

   ============================================================================== */

struct Vec3 {
    double x;
    double y;
    double z;

    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Addition
    [[nodiscard]] constexpr Vec3 operator+(const Vec3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    // Subtract
    [[nodiscard]] constexpr Vec3 operator-(const Vec3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    // Negative
    [[nodiscard]] constexpr Vec3 operator-() const noexcept {
        return {-x, -y, -z};
    }

    // Multiply
    [[nodiscard]] constexpr Vec3 operator*(double scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }

    // Divide
    [[nodiscard]] constexpr Vec3 operator/(double scalar) const noexcept {
        return {x / scalar, y / scalar, z / scalar};
    }

    // Compound Operators
    constexpr Vec3& operator+=(const Vec3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    constexpr Vec3& operator-=(const Vec3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    constexpr Vec3& operator*=(double scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    constexpr Vec3& operator/=(double scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // Norms
    [[nodiscard]] constexpr double norm2() const noexcept {
        return x*x + y*y + z*z;
    }
    [[nodiscard]] double norm() const noexcept {
        return std::sqrt(norm2());
    }

    [[nodiscard]] bool is_finite() const noexcept {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
    }
};

[[nodiscard]] constexpr inline double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
[[nodiscard]] constexpr inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
[[nodiscard]] constexpr inline Vec3 operator*(double scalar, const Vec3& vector) noexcept {
    return vector * scalar;
}