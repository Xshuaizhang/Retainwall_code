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
 This file declares some consitutive models.
*/

#pragma once
#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <ctime>
#include <cuda_runtime.h>
#include "types.h"


/**
* Update stress tensor using Drucker-Prager constitutive model
* Please refer to the book Zhang et al. "Material Point Method" (张雄 等，《物质点法》)
*/
static __inline__ __device__ void cuMatModDP(const scalar deltat, const scalar4 omega, tensor2d dedt, tensor2d& sigma, scalar* par_material, bool& yld) {
    // Jaumann stress rate - rotation
    tensor2d jau_str_rate;
    tensor2dful spin = SkewTensor2FullTensor(make_scalar3(omega));
    tensor2dful stress_full = SymTensor2FullTensor(sigma);
    jau_str_rate = FullTensor2SymTensor(LinearTransformation(spin, stress_full) - LinearTransformation(stress_full, spin));

    // Elastic predict
    tensor2d str_rate;
    tensor2d str_pre; //predicted stress
    scalar bulk_modulus = par_material[0];
    scalar shear_modulus = par_material[1];
    scalar lame_par = bulk_modulus - 2.0 / 3.0 * shear_modulus;
    str_rate = jau_str_rate + dedt * (2.0 * shear_modulus) + SymTensorIdentity() * (lame_par * trace(dedt));
    str_pre = sigma + str_rate * deltat;

    // Obtain deviation stress
    scalar str_sph = (trace(str_pre)) / 3.0; //spheric stress str_sph
    tensor2d str_dev = DeviationTensor(str_pre);
    
    // Plastic correction
    scalar k_psi = par_material[3];
    scalar k_c = par_material[4];
    scalar k_phi = par_material[2];
    scalar tenlt = 0.0; //tensionlimit
    if (k_phi == 0.0) tenlt = 0.0;
    else tenlt = k_c / k_phi;

    scalar tenyld = str_sph - tenlt; // Tensile yielding
    if (tenyld >= 0.0) {
        scalar dlamda = tenyld / bulk_modulus;
        str_sph = tenlt;
        //yld = 1;
    }

    scalar tau = sqrtf(0.5 * trace(str_dev, str_dev)); // Equivalent shear stress
    scalar shr_yld = tau - (-str_sph * k_phi + k_c); // Shear yielding
    if (shr_yld > 0.0) {
        scalar dlamda = shr_yld / (shear_modulus + bulk_modulus * k_phi * k_psi);
        str_sph -= bulk_modulus * k_psi * dlamda;

        scalar ratio = tau < 1e-10 ? 1.0 : (k_c - k_phi * str_sph) / tau;
        str_dev = str_dev * ratio;
        yld = 1;
    }
    sigma = str_dev + SymTensorIdentity() * str_sph;
}

/**
* Update stress tensor using Mu(I) relation-enhanced Drucker-Prager constitutive model
* Please refer to zhu2022Lagrangian for more details
*/
static __inline__ __device__ void cuMatModMuIDP(const scalar deltat, const scalar4 omega, tensor2d dedt, tensor2d& sigma, scalar* par_material) {
    // Jaumann stress rate - rotation
    tensor2d jau_str_rate;
    tensor2dful spin = SkewTensor2FullTensor(make_scalar3(omega));
    tensor2dful stress_full = SymTensor2FullTensor(sigma);
    jau_str_rate = FullTensor2SymTensor(LinearTransformation(spin, stress_full) - LinearTransformation(stress_full, spin));

    // Elastic predict
    tensor2d str_rate;
    tensor2d str_pre; //predicted stress
    scalar bulk_modulus = par_material[0];
    scalar shear_modulus = par_material[1];
    scalar lame_par = bulk_modulus - 2.0 / 3.0 * shear_modulus;

    str_rate = jau_str_rate + dedt * (2.0 * shear_modulus) + SymTensorIdentity() * (lame_par * trace(dedt));
    str_pre = sigma + str_rate * deltat;

    // Obtain deviation stress
    scalar str_sph = (trace(str_pre)) / 3.0; //spheric stress str_sph
    tensor2d str_dev = DeviationTensor(str_pre);

    // Plastic correction
    scalar gamma = sqrt(2.0 * J2(DeviationTensor(dedt)));
    if (str_sph > 0.0) str_sph = 0.0;
    scalar mu_s = par_material[2];
    scalar delta_mu = par_material[3] - par_material[2];
    scalar I_0 = par_material[4];
    scalar rho_g = par_material[5]; //Grain density
    scalar d_grain = par_material[6];
    scalar lambda = par_material[7];
    scalar pressure = -str_sph;
    scalar mu = mu_s + delta_mu * gamma * d_grain / (I_0 * sqrt(pressure / rho_g) + d_grain * sqrt(gamma * gamma + lambda * lambda));

    scalar tau = sqrt(0.5 * trace(str_dev, str_dev)); // Second invariant of deviation stress
    scalar shr_yld = tau + str_sph * mu; // Shear yielding
    if (shr_yld > 0.0) {
        scalar ratio = tau < 1e-10 ? 1.0 : (0.0 - mu * str_sph) / tau;
        str_dev = str_dev * ratio;
    }
    sigma = str_dev + SymTensorIdentity() * str_sph;
}

/**
* Update stress tensor using basic hypoplastic constitutive model 
* Please refer to wu2017basic for more details
*/
static __inline__ __device__ void cuMatModHypo(
    const scalar deltat,
    const scalar4 omega,
    const tensor2d strain_rate,  
    tensor2d& sigma,
    scalar* par_material)
{
    scalar Ei = par_material[0];
    scalar Vi = par_material[1];
    scalar phi_c = par_material[2];
    scalar cohesion = par_material[3];
    scalar p_ref = par_material[4];

    scalar C1 = -Ei / (3.0 * p_ref * (1 + Vi));
    scalar a = 3.0 / (sqrt(2.0) * sin(phi_c));
    scalar C2 = C1 * ((3.0 * Vi + a * sqrt(1.0 + Vi * Vi)) / (1 - Vi) - a * a / 3.0);
    scalar C3 = a * a * C1;
    scalar C4 = a * C1;

    // Jaumann stress - rotation
    tensor2d jau_str_rate;
    tensor2dful spin = SkewTensor2FullTensor(make_scalar3(omega));
    tensor2dful stress_full = SymTensor2FullTensor(sigma);
    if (cohesion > 0.0) stress_full = stress_full - Tensor2dFulIdentity() * cohesion;
    jau_str_rate = FullTensor2SymTensor(LinearTransformation(spin, stress_full) - LinearTransformation(stress_full, spin));

    // Stress regularization for zero stress state
    tensor2d str = FullTensor2SymTensor(stress_full);
    scalar I1 = trace(str);
    if (I1 > -4.0) {
        I1 = -4.0;
        str = make_tensor2d(-1.0, -1.0, -2.0, 0.0, 0.0, 0.0);
        sigma = str;
        return;
    }

    // Variables related to current stress state
    tensor2d str_dev = str - SymTensorIdentity() * (I1 / 3.0); // deviation stress
    scalar tr_str = trace(str);
    scalar tr_stn_rate = trace(strain_rate);
    scalar tr_str_stn_rate = trace(str, strain_rate);
    scalar norm_stn_rate = sqrtf(trace(strain_rate, strain_rate));

    // Stress update with the hypoplastic model considering the critical state
    tensor2d stress_rate = jau_str_rate + strain_rate * (C1 * tr_str) + str * (C2 * tr_stn_rate) + str * (C3 * tr_str_stn_rate / tr_str) + (str + str_dev) * (C4 * norm_stn_rate);
    str = sigma + stress_rate * deltat;

    // New stress state -- check whether tensile yielding
    scalar str_sph = trace(str) / 3.0;
    if (str_sph > 0.0) {
        sigma = make_tensor2d(-1.0 + cohesion, -1.0 + cohesion, -2.0 + cohesion, 0.0, 0.0, 0.0);
        return;
    };
    str_dev = str - SymTensorIdentity() * str_sph;

    // New tress state -- check whether shear yielding
    scalar k_phi = 3.0 / (sqrt(2.0) *a);
    scalar tau = sqrtf(0.5 * trace(str_dev, str_dev)); // Second invariant of deviation stress
    scalar shr_yld = tau - (-str_sph * k_phi); // Shear yielding
    if (shr_yld > 0.0) {
        scalar ratio = tau < 1e-10 ? 1.0 : (-k_phi * str_sph) / tau;
        str_dev = str_dev * ratio;
    }

    // Final stress state
    sigma = str_dev + SymTensorIdentity() * (str_sph + cohesion);
}

/**
* Update stress tensor using hypoplastic constitutive model considering the critical state
* Please refer to wu2017basic and zhu2024SPH for more details
*/

static __inline__ __device__ void cuMatModHypo(const scalar deltat, const scalar density, const scalar4 omega, const tensor2d strain_rate,  
    tensor2d& sigma, scalar* par_material, scalar& M_crt) {
    // Material parameters
    scalar Ei = par_material[0];
    scalar Vi = par_material[1];
    scalar phi_c = par_material[2];
    scalar cohesion = par_material[3];
    scalar p_ref = par_material[4];
    scalar void_co = par_material[5];
    scalar lambda = par_material[6];
    scalar xi = par_material[7];
    scalar alpha = par_material[8];
    scalar void_min = par_material[9];
    scalar void_max = par_material[10];
    scalar void_0 = par_material[11];
    scalar rho0 = par_material[12];

    scalar C1 = -Ei / (3.0 * p_ref * (1 + Vi));
    scalar a = 3.0 / (sqrt(2.0) * sin(phi_c));
    scalar C2 = C1 * ((3.0 * Vi + a * sqrt(1.0 + Vi * Vi)) / (1 - Vi) - a * a / 3.0);
    scalar C3 = a * a * C1;
    scalar C4 = a * C1;

    // Jaumann stress - rotation
    tensor2d jau_str_rate;
    tensor2dful spin = SkewTensor2FullTensor(make_scalar3(omega));
    tensor2dful stress_full = SymTensor2FullTensor(sigma);
    if (cohesion > 0.0) stress_full = stress_full - Tensor2dFulIdentity() * cohesion;
    jau_str_rate = FullTensor2SymTensor(LinearTransformation(spin, stress_full) - LinearTransformation(stress_full, spin));

    // Stress regularization for zero stress state
    tensor2d str = FullTensor2SymTensor(stress_full);
    scalar I1 = trace(str);
    if (I1 > -4.0) {
        I1 = -4.0;
        str = make_tensor2d(-1.0, -1.0, -2.0, 0.0, 0.0, 0.0);
        sigma = str;
        return;
    }

    // Variables related to current stress state
    tensor2d str_dev = str - SymTensorIdentity() * (I1 / 3.0); // deviation stress
    scalar tr_str = trace(str);
    scalar tr_stn_rate = trace(strain_rate);
    scalar tr_str_stn_rate = trace(str, strain_rate);
    scalar norm_stn_rate = sqrtf(trace(strain_rate, strain_rate));


    // Variables related to critical state
    scalar p = -I1 / 3.0;
    scalar void_ratio = (1.0 + void_0) * rho0 / density - 1.0;
    void_ratio = max(void_ratio, void_min);
    void_ratio = min(void_ratio, void_max);
    scalar e_crt = void_co * exp(-lambda * pow(p / p_ref, xi));
    scalar fd = pow((void_ratio - void_min) / (e_crt - void_min), alpha);

    //critical state parameter
    M_crt = void_ratio - e_crt;

    // Stress update with the hypoplastic model considering the critical state
    tensor2d stress_rate = jau_str_rate + strain_rate * (C1 * tr_str) + str * (C2 * tr_stn_rate) + str * (C3 * tr_str_stn_rate / tr_str) + (str + str_dev) * (fd * C4 * norm_stn_rate);
    str = sigma + stress_rate * deltat;

    // New stress state -- check whether tensile yielding
    scalar str_sph = trace(str) / 3.0;
    if (str_sph > 0.0) {
        sigma = make_tensor2d(-1.0 + cohesion, -1.0 + cohesion, -2.0 + cohesion, 0.0, 0.0, 0.0);
        return;
    };
    str_dev = str - SymTensorIdentity() * str_sph;

    // New tress state -- check whether shear yielding
    scalar fa = C2 / C1;
    scalar fb = C3 / C1;
    scalar fc = C4 / C1;
    scalar A = fd * fd * fc * fc / 3.0 - (1.0 + fa + fb / 3.0) * (1.0 + fa + fb / 3.0);
    scalar B = 2.0 * fd * fd * fc * fc * ((1.0 + fa + fb / 3.0) * (1.0 + fa + fb / 3.0) + 2.0 * fa + 3.0) - 4.0 * fb * (1 + fa + fb / 3.0);
    scalar C = 4.0 / 3.0 * fd * fd * fc * fc * fb * fb - 4.0 * fb * fb;
    scalar k2 = (-B - sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
    scalar k_phi = 3.0 / sqrt(k2);
  
    scalar tau = sqrtf(0.5 * trace(str_dev, str_dev)); // Second invariant of deviation stress
    scalar shr_yld = tau - (-str_sph * k_phi); // Shear yielding
    if (shr_yld > 0.0) {
        scalar ratio = tau < 1e-10 ? 1.0 : ( - k_phi * str_sph) / tau;
        str_dev = str_dev * ratio;
    }

    // Final stress state
    sigma = str_dev + SymTensorIdentity() * (str_sph + cohesion);
}

#endif