/*********************************************************************************
 * Copyright (c) 2021, Chengwei Zhu. All rights reserved.

 * <LOQUAT> Three-dimensional GPU-accelerated SPH solver for geotechnical modeling

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    1. Redistributions of source code must retain the copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. The origin of this software must not be misrepresented; you must
 *       not claim that you wrote the original software.
 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
 * EVENT SHALL THE HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE.
 *********************************************************************************/

/*
 This file declares some useful mathematical functions.
*/

#pragma once
#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include <ctime>
#include <cuda_runtime.h>
#include "types.h"
// Algebraic Operations related with int3
static __inline__ __host__ __device__ int3 make_int3(const int v) {	return{ v, v, v }; }
static __inline__ __host__ __device__ int3 make_int3(scalar3 v) { return{ int(v.x), int(v.y), int(v.z)}; }
static __inline__ __host__ __device__ int3 make_int3(int4 v) { return{ v.x, v.y, v.z }; }
static __inline__ __host__ __device__ int3 operator + (const int3 v1, const int3 v2) { return{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
static __inline__ __host__ __device__ bool operator >(const int3 v1, const int3 v2) { return(v1.x > v2.x && v1.y > v2.y && v1.z > v2.z); }
static __inline__ __host__ __device__ bool operator >=(const int3 v1, const int3 v2) { return(v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z); }
static __inline__ __host__ __device__ bool operator <(const int3 v1, const int3 v2) { return(v1.x < v2.x && v1.y < v2.y && v1.z < v2.z);}
static __inline__ __host__ __device__ bool operator <=(const int3 v1, const int3 v2) { return(v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z); }

// Algebraic Operations related with int4
static __inline__ __host__ __device__ int4 make_int4(int v) { return{ v, v, v, v }; }
static __inline__ __host__ __device__ int4 make_int4(int3 v) { return{ v.x, v.y, v.z,0 }; }
static __inline__ __host__ __device__ int4 make_int4(int3 v1, int v2) { return{ v1.x, v1.y, v1.z, v2 }; }
static __inline__ __host__ __device__ int4 make_int4(scalar3 v) { return{ int(v.x), int(v.y), int(v.z),0 }; }
static __inline__ __host__ __device__ int4 operator + (const int4 v1, const int3 v2) { return{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w }; }

// Algebraic Operations related with scalar3
static __inline__ __host__ __device__ scalar3 make_scalar3(scalar v) { return { v, v, v }; }
static __inline__ __host__ __device__ scalar3 make_scalar3(scalar v1, scalar v2, scalar v3) { return { v1, v2, v3 }; }
static __inline__ __host__ __device__ scalar3 make_scalar3(scalar2 v) { return { v.x, v.y, 0.0 }; }
static __inline__ __host__ __device__ scalar3 make_scalar3(scalar4 v) { return { v.x, v.y, v.z }; }
static __inline__ __host__ __device__ scalar3 operator + (const scalar3 v1, const scalar3 v2) {	return{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };}
static __inline__ __host__ __device__ scalar3 operator - (const scalar3 v1, const scalar3 v2) {return{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };}
static __inline__ __host__ __device__ scalar3 operator * (const scalar3 v1, const scalar v2) {return{ v1.x * v2, v1.y * v2, v1.z * v2 };}
static __inline__ __host__ __device__ scalar3 operator * (const scalar3 v1, const scalar3 v2) { return{ v1.x * v2.x, v1.y * v2.y, v1.z * v2.z }; }
static __inline__ __host__ __device__ scalar3 operator * (const int3 v1, const scalar v2) { return{ v1.x * v2, v1.y * v2, v1.z * v2 }; }
static __inline__ __host__ __device__ scalar3 operator / (const scalar3 v1, const scalar v2) {return{ v1.x / v2, v1.y / v2, v1.z / v2 };}

// Algebraic Operations related with scalar4
static __inline__ __host__ __device__ scalar4 make_scalar4(scalar v) { return { v, v, v, v }; }
static __inline__ __host__ __device__ scalar4 make_scalar4(scalar v1, scalar v2, scalar v3, scalar v4) { return { v1, v2, v3, v4 }; }
static __inline__ __host__ __device__ scalar4 make_scalar4(scalar2 v) { return { v.x, v.y, 0.0, 0.0 }; }
static __inline__ __host__ __device__ scalar4 make_scalar4(scalar3 v) { return { v.x, v.y, v.z, 0.0 }; }
static __inline__ __host__ __device__ scalar4 make_scalar4(const scalar3 v1, const scalar v2) {	return{ v1.x, v1.y, v1.z, v2 };}
static __inline__ __host__ __device__ scalar4 operator + (const scalar4 v1, const scalar4 v2) {	return{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w };}
static __inline__ __host__ __device__ scalar4 operator + (const scalar4 v1, const scalar3 v2) { return{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w }; }
static __inline__ __host__ __device__ scalar4 operator - (const scalar4 v1, const scalar4 v2) { return{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
static __inline__ __host__ __device__ scalar4 operator - (const scalar4 v1, const scalar3 v2) { return{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w }; }
static __inline__ __host__ __device__ scalar4 operator * (const scalar4 v1, const scalar v2) { return{ v1.x * v2, v1.y * v2, v1.z * v2, v1.w * v2 }; }
static __inline__ __host__ __device__ scalar4 operator / (const scalar4 v1, const scalar v2) { return{ v1.x / v2, v1.y / v2, v1.z / v2, v1.w / v2 }; }

// Algebraic Operations related with tensor2d
static __inline__ __host__ __device__ tensor2d make_tensor2d(scalar v) { return{ v, v, v, v, v, v }; }
static __inline__ __host__ __device__ tensor2d make_tensor2d(scalar v1, scalar v2, scalar v3, scalar v4, scalar v5, scalar v6) { return{ v1, v2, v3, v4, v5, v6 };}
static __inline__ __host__ __device__ tensor2d operator + (const tensor2d v1, const tensor2d v2) { return{ v1.xx + v2.xx, v1.yy + v2.yy, v1.zz + v2.zz, v1.xy + v2.xy, v1.xz + v2.xz, v1.yz + v2.yz };}
static __inline__ __host__ __device__ tensor2d operator - (const tensor2d v1, const tensor2d v2) { return{ v1.xx - v2.xx, v1.yy - v2.yy, v1.zz - v2.zz, v1.xy - v2.xy, v1.xz - v2.xz, v1.yz - v2.yz };}
static __inline__ __host__ __device__ tensor2d operator - (const tensor2d v1, const scalar3 v2) { return{ v1.xx - v2.x, v1.yy - v2.y, v1.zz - v2.z, v1.xy, v1.xz, v1.yz }; }
static __inline__ __host__ __device__ tensor2d operator * (const tensor2d v1, const scalar v2) { return{ v1.xx * v2, v1.yy * v2, v1.zz * v2, v1.xy * v2, v1.xz * v2, v1.yz * v2 }; }
static __inline__ __host__ __device__ tensor2d operator / (const tensor2d v1, const scalar v2) { return{ v1.xx / v2, v1.yy / v2, v1.zz / v2, v1.xy / v2, v1.xz / v2, v1.yz / v2 }; }

// Algebraic Operations related with tensor2dful
static __inline__ __host__ __device__ tensor2dful make_tensor2dful(scalar v) { return{ v, v, v, v, v, v, v, v, v };}
static __inline__ __host__ __device__ tensor2dful make_tensor2dful(scalar v1, scalar v2, scalar v3, scalar v4, scalar v5, scalar v6, scalar v7, scalar v8, scalar v9) {	return{ v1, v2, v3, v4, v5, v6, v7, v8, v9 }; }
static __inline__ __host__ __device__ tensor2dful operator + (const tensor2dful v1, const tensor2dful v2) { return{ v1.xx + v2.xx, v1.xy + v2.xy, v1.xz + v2.xz, v1.yx + v2.yx, v1.yy + v2.yy, v1.yz + v2.yz, v1.zx + v2.zx, v1.zy + v2.zy, v1.zz + v2.zz }; }
static __inline__ __host__ __device__ tensor2dful operator - (const tensor2dful v1, const tensor2dful v2) { return{ v1.xx - v2.xx, v1.xy - v2.xy, v1.xz - v2.xz, v1.yx - v2.yx, v1.yy - v2.yy, v1.yz - v2.yz, v1.zx - v2.zx, v1.zy - v2.zy, v1.zz - v2.zz }; }
static __inline__ __host__ __device__ tensor2dful operator * (const tensor2dful v1, const scalar v2) { return{ v1.xx * v2, v1.xy * v2, v1.xz * v2,	v1.yx * v2, v1.yy * v2, v1.yz * v2,	v1.zx * v2, v1.zy * v2, v1.zz * v2 }; }

// Matrix operation
static __inline__ __host__ __device__ scalar DotProduct(const scalar3 v1, const scalar3 v2){ return(v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }
static __inline__ __host__ __device__ scalar J2(const tensor2d v1, const tensor2d v2) { return(v1.xx * v2.xx + v1.yy * v2.yy + v1.zz * v2.zz + 2.0 * v1.xy * v2.xy + 2.0 * v1.xz * v2.xz + 2.0 * v1.yz * v2.yz); }
static __inline__ __host__ __device__ scalar J2(const tensor2d v) { return(v.xx * v.xx + v.yy * v.yy + v.zz * v.zz + 2.0 * v.xy * v.xy + 2.0 * v.xz * v.xz + 2.0 * v.yz * v.yz); }
static __inline__ __host__ __device__ scalar EulerianNorm(const scalar3 v) { return(sqrt(v.x * v.x + v.y * v.y + v.z * v.z)); }
static __inline__ __host__ __device__ scalar trace(const tensor2d v) { return(v.xx + v.yy + v.zz); }
static __inline__ __host__ __device__ scalar trace(const tensor2d v1, const tensor2d v2) {return(v1.xx * v2.xx + v1.yy * v2.yy + v1.zz * v2.zz + 2.0 * v1.xy * v2.xy + 2.0 * v1.xz * v2.xz + 2.0 * v1.yz * v2.yz); }
static __inline__ __host__ __device__ scalar3 DotProduct(const tensor2d v1, const scalar3 v2){ return{ v1.xx * v2.x + v1.xy * v2.y + v1.xz * v2.z, v1.xy * v2.x + v1.yy * v2.y + v1.yz * v2.z, v1.xz * v2.x + v1.yz * v2.y + v1.zz * v2.z }; }
static __inline__ __host__ __device__ scalar3 DotProduct(const tensor2dful v1, const scalar3 v2) { return{ v1.xx * v2.x + v1.xy * v2.y + v1.xz * v2.z, v1.yx * v2.x + v1.yy * v2.y + v1.yz * v2.z, v1.zx * v2.x + v1.zy * v2.y + v1.zz * v2.z }; }
static __inline__ __host__ __device__ scalar3 SkewTensor(const tensor2dful v) { return{ 0.5f * (v.yx - v.xy), 0.5f * (v.zx - v.xz), 0.5f * (v.zy - v.yz) }; }
static __inline__ __host__ __device__ scalar3 CrossProduct(const scalar3 v1, const scalar3 v2) { return{ v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,v1.x * v2.y - v1.y * v2.x }; }
static __inline__ __host__ __device__ tensor2d FullTensor2SymTensor(const tensor2dful v) { return{ v.xx, v.yy, v.zz, 0.5f * (v.xy + v.yx), 0.5f * (v.xz + v.zx), 0.5f * (v.yz + v.zy) }; }
static __inline__ __host__ __device__ tensor2d SymTensorIdentity() { return{ 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }; }
static __inline__ __host__ __device__ tensor2d SymTensor(const tensor2dful v) { return{ v.xx, v.yy, v.zz, 0.5f * (v.xy + v.yx), 0.5f * (v.xz + v.zx), 0.5f * (v.yz + v.zy) }; }
static __inline__ __host__ __device__ tensor2d DeviationTensor(tensor2d v) { return(v - SymTensorIdentity() * (trace(v) / 3.0)); }
static __inline__ __host__ __device__ tensor2dful Tensor2dFulIdentity() { return{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 }; }
static __inline__ __host__ __device__ tensor2dful TensorProduct(const scalar3 v1, const scalar3 v2){
	return{ v1.x * v2.x, v1.x * v2.y, v1.x * v2.z,
			v1.y * v2.x, v1.y * v2.y, v1.y * v2.z,
			v1.z * v2.x, v1.z * v2.y, v1.z * v2.z };
}
static __inline__ __host__ __device__ tensor2dful SkewTensor2FullTensor(const scalar3 v) {
	return{ 0.0, -v.z, v.y,
		    v.z, 0.0, -v.x,
		   -v.y, v.x, 0.0 };
}
static __inline__ __host__ __device__ tensor2dful SymTensor2FullTensor(const tensor2d v) {
	return{ v.xx, v.xy, v.xz,
			v.xy, v.yy, v.yz,
			v.xz, v.yz, v.zz };
}
static __inline__ __host__ __device__ tensor2dful LinearTransformation(const tensor2dful v1, const tensor2dful v2) {
	return{ v1.xx * v2.xx + v1.xy * v2.yx + v1.xz * v2.zx, v1.xx * v2.xy + v1.xy * v2.yy + v1.xz * v2.zy, v1.xx * v2.xz + v1.xy * v2.yz + v1.xz * v2.zz,
		    v1.yx * v2.xx + v1.yy * v2.yx + v1.yz * v2.zx, v1.yx * v2.xy + v1.yy * v2.yy + v1.yz * v2.zy, v1.yx * v2.xz + v1.yy * v2.yz + v1.yz * v2.zz,
		    v1.zx * v2.xx + v1.zy * v2.yx + v1.zz * v2.zx, v1.zx * v2.xy + v1.zy * v2.yy + v1.zz * v2.zy, v1.zx * v2.xz + v1.zy * v2.yz + v1.zz * v2.zz };
}
static __inline__ __host__ __device__ tensor2dful GetRodriguesMatrix(const scalar3 angle) {
	scalar norm_angle = EulerianNorm(angle);
	scalar3 rotat_axis = angle / norm_angle;
	tensor2dful K = SkewTensor2FullTensor(rotat_axis);
	return(Tensor2dFulIdentity() + K * sin(norm_angle) + LinearTransformation(K, K) * (1.0 - cos(norm_angle)));
}

// Functions related to geometry
static __inline__ __host__ __device__ scalar Dist2Plane(const scalar3 point, const scalar4 plane) {
	return((point.x * plane.x + point.y * plane.y + point.z * plane.z + plane.w) / sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z));
}
static __inline__ __host__ __device__ scalar AbsDist2Plane(const scalar3 point, const scalar4 plane) {
	return(abs(point.x * plane.x + point.y * plane.y + point.z * plane.z + plane.w) / sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z));
}

// Functions used in SPH
static __inline__ __host__ __device__ scalar GetKernel(const scalar par, const scalar q) { return(par * (2.0 - q) * (2.0 - q) * (2.0 - q) * (2.0 - q) * (q + 0.5)); }
static __inline__ __host__ __device__ scalar GetGradKernel(const scalar par, const scalar q) { return(par * (2.0f - q) * (2.0f - q) * (2.0f - q)); }

#endif
