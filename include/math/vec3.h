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

    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Addition
    Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    // Subtract
    Vec3 operator-(const Vec3& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    // Multiply
    Vec3 operator*(double s) const {
        return {x * s, y * s, z * s};
    }

    // Divide
    Vec3 operator/(double s) const {
        return {x / s, y / s, z / s};
    }

    // Compound Operators
    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    Vec3& operator-=(const Vec3& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
    Vec3& operator*=(double s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    Vec3& operator/=(double s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // Norms
    double norm2() const {
        return x*x + y*y + z*z;
    }
    double norm() const {
        return std::sqrt(norm2());
    }
};

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline Vec3 operator *(double s, const Vec3& v) {
    return {s*v.x, s*v.y, s*v.z};
}