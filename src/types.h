/*********************************************************************************
 * Copyright (c) 2018, Chong Peng. All rights reserved.

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
  This file declares all global parameters. Any additional parameters should be added here.
 */

#pragma once
//#define DOUBLE_PRECISION

# if defined(DOUBLE_PRECISION)
using scalar = double;
# else
using scalar = float;
# endif

#define PI (3.14159265358979323846f)
#define PI_D (3.14159265358979323846)
#define COMPUTE_DISPLACEMENT 0
#define MATERIAL_NUMBER (100.0f)
#define MODEL_NUMBER_MAX (10)
#define MATERIAL_SIZE_MAX (15)
#define RIGID_MOTION_MAX (10)

#include <ctime>
#include <cuda_runtime.h>
#include <vector>
#include <climits>

// Enumeration variables
enum status {Fix, NonFix};
enum dimension { OneD = 1, TwoD = 2, ThreeD = 3 };
enum shape { Cuboid, Circle };
enum model_mat { RIGID, DP_EPP, MuI_DP_EPP, HypoBasic, HypoCS }; //material model
// DP_EPP      elasto-perfectly plasticity with Drucker-Prager yield surface c.f. peng2019Loquat
// MuI_DP_EPP  elasto-perfectly plasticity with Mu(I) relation enhanced Drucker-Prager yield surface c.f. zhu2022lagrangian
// HypoBasic   Basic Hypo-plasticity proposed by Wei Wu, c.f. wu2017basic
// HypoCS      Enhanced Hypo-plasticity with critical state. c.f. wu2017basic & zhu2023SPH

// Self-defined variables used in Loquat
struct scalar2 { scalar x, y; };
struct scalar3 { scalar x, y, z; };
struct scalar4 { scalar x, y, z, w; };
struct tensor2d { scalar xx, yy, zz, xy, xz, yz;};
struct tensor2dful { scalar xx, xy, xz, yx, yy, yz, zx, zy, zz;};

using material = scalar[MATERIAL_SIZE_MAX];

struct RigidMotion {		
	bool is_trans;
	bool is_rotat;
	scalar3 trans_vel;
	scalar3 trans_acc;

	shape pattern;
	scalar3 pos_rp;      //Position of reference point
	scalar3 rotat_vel;
	scalar3 rotat_acc;
};

/// Imposed motion structure
using table = scalar[10];
using table3 = scalar3[10];
struct MotionTable {
	//Translation
	bool is_translation = false;
	int table_size_trans;
	table3 trans_vel;
	table trans_time;
	table3 trans_acc;

	// Rotation
	bool is_rotation = false;
	unsigned int pattern;
	int table_size_rotat;
	scalar3 pos_rp;      //Position of reference point 
	scalar3 rotat_vel;
	table rotat_time;
	table3 rotat_acc;
};

// Parameters for the SPH simulation
struct Parameters {
	// Configuration
	dimension dim;          // Dimension of problem
	bool periodbound;
	scalar4 planeleft;      // The former three variables denote the unit normal of plane left
	scalar4 planeright;     // The former three variables denote the unit normal of plane right

	int n_motion;           // Number of movtion bodies
	RigidMotion mv[RIGID_MOTION_MAX]; 

	int np;					// Total number of particle
	int npb;				// Number of boundary particle
	int npm;				// Number of material particle
	int np_delete_m;		// Number of deleted material particle
	

	scalar3 domain_size;	// Size of calculation domain
	scalar3 domain_min;		// Starting point of the calculation domain

	int3 cell_num;			// Number of cells in 3 directions
	int ncell;				// Total number of cell
	int ncell_material;		// Starting cell number for material particles
	int ncell_delete_m;		// The cell index for deleted material particles

	scalar h;				// Smoothing length
	scalar i_h;				// Inverse of smoothing length
	scalar dr;				// Particle size
	scalar hdr;				// Ratio between h and dr, hdr = h / dr
	scalar vol_def;         // Defaul particle volumn
	scalar bound_thickness; // Width of the boundary layer = ceil(2.f * hdr) * dr

	scalar3 gravity;		// Body force


	// Material parameters
	scalar rho0_m;				// Reference density of solid
	scalar damping;              // damping
	int n_model;                    // Number of models
	material model_par[MODEL_NUMBER_MAX];		// Model parameter

	// Numerical stabilization techniques 
	scalar kernel_normalization_par;				// Kernel renormalization parameter
	scalar kernel_gradient_normalization_par;	// Kernal gradient renormalization parameter

	int stress_shepard_filter;					// Interval of steps to apply shepard filter to stress
	int dynamic_time_step;						// Switch for variable time step
	int integration_method;						// Method for time integration, 1- Predictor-Corrector is implemented

	int viscosity_type;							// Type of viscosity treatment, 1- artificial viscosity
	scalar artificial_viscosity_alpha_m;			// Parameter for the artificial viscosity for the solid phase
	scalar cs_m;					// Speed of sound of solid
	scalar eta;					// 0.01 * h * h, to prevent singularity	
	scalar cfl;					// CFL number

	scalar artificial_pressure_coefficient_material;		// Parameter of the artificial pressure for material phase
	scalar kernel_zero;							// W(0)
	scalar kernel_dr;							// W(dr)	

	// Simulation output	
	scalar time_max;								// Total simulation time
	scalar time;									// Current time
	scalar dt;									// Time step
	scalar output_frequency;						// Time interval for saving results
	int output_number;							// Number of output files
	int output_format;                          // 1 - dat; 2 - vtu;						
};

struct Timer {
	double total;
	double neighbor_search;
	double adami_boundary;
	double interpolation;
	double interaction;
	double integration;
	double saving;

	std::clock_t clock_initial;
	std::clock_t clock_neighbor_search;
	std::clock_t clock_adami_boundary;
	std::clock_t clock_interpolation;
	std::clock_t clock_interaction;
	std::clock_t clock_integration;
	std::clock_t clock_saving;
	std::clock_t clock_interval;
};
