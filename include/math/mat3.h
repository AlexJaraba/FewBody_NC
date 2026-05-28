#pragma once

#include "math/vec3.h"

struct Mat3 {
    double xx, xy, xz; 
    double yx, yy, yz; 
    double zx, zy, zz; 
    
    Mat3() : xx(0.0), xy(0.0), xz(0.0), 
             yx(0.0), yy(0.0), yz(0.0), 
             zx(0.0), zy(0.0), zz(0.0) {}
    
    Mat3 operator+(const Mat3& rhs) const {
        Mat3 m;
        m.xx = xx + rhs.xx;
        m.xy = xy + rhs.xy;
        m.xz = xz + rhs.xz;
        m.yx = yx + rhs.yx;
        m.yy = yy + rhs.yy;
        m.yz = yz + rhs.yz;
        m.zx = zx + rhs.zx;
        m.zy = zy + rhs.zy;
        m.zz = zz + rhs.zz;

        return m;
    }

    Mat3 operator-(const Mat3& rhs) const {
        Mat3 m;
        m.xx = xx - rhs.xx;
        m.xy = xy - rhs.xy;
        m.xz = xz - rhs.xz;
        m.yx = yx - rhs.yx;
        m.yy = yy - rhs.yy;
        m.yz = yz - rhs.yz;
        m.zx = zx - rhs.zx;
        m.zy = zy - rhs.zy;
        m.zz = zz - rhs.zz;

        return m;
    }

    Mat3& operator+=(const Mat3& rhs) {
        xx += rhs.xx;
        xy += rhs.xy;
        xz += rhs.xz;
        yx += rhs.yx;
        yy += rhs.yy;
        yz += rhs.yz;
        zx += rhs.zx;
        zy += rhs.zy;
        zz += rhs.zz;

        return *this;
    }

    Mat3 operator-=(const Mat3& rhs) {
        Mat3 m;
        xx -= rhs.xx;
        xy -= rhs.xy;
        xz -= rhs.xz;
        yx -= rhs.yx;
        yy -= rhs.yy;
        yz -= rhs.yz;
        zx -= rhs.zx;
        zy -= rhs.zy;
        zz -= rhs.zz;

        return *this;
    }
};

inline Vec3 operator*(const Mat3& A, const Vec3& v) {
    return {
        A.xx * v.x + A.xy * v.y + A.xz * v.z,
        A.yx * v.x + A.yy * v.y + A.yz * v.z,
        A.zx * v.x + A.zy * v.y + A.zz * v.z
    };
}

inline Mat3 outer(const Vec3& a, const Vec3& b) {
    Mat3 m;

    m.xx = a.x * b.x;
    m.xy = a.x * b.y;
    m.xz = a.x * b.z;
    m.yx = a.y * b.x;
    m.yy = a.y * b.y;
    m.yz = a.y * b.z;
    m.zx = a.z * b.x;
    m.zy = a.z * b.y;
    m.zz = a.z * b.z;

    return m;
}