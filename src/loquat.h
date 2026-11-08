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

#pragma once
#ifndef __LOQUAT_H__
#define __LOQUAT_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <boost/variant.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "types.h"
#include "functions.h"
#include "tinyxml2.h"
#include "loquat_io.h"
#include "loquat_grid.h"


class LoquatSph
{
	private:
	
		std::string project_name;
		int selected_device;
		
		Parameters* par_host;
		Parameters* par_device;		
		Timer* timer;
		LoquatGrid* grid;
		LoquatIo* io;

		// Rigid motion
		MotionTable* kinematic_table;
		RigidMotion* moving_body;

		// Primary arrays
		int4* codes;        // Store block ID, status, material type, material ID. 
		int4* cellpos;		// Coordinates for cell
		scalar4* pospres;		// Position + pression
		scalar4* pospres_ori;		// Position + pression for accumulating displacement
		scalar4* velrhop;		// Velocity + density
		scalar4* idmassidx;		// Object number + particle mass + Original particle index
		tensor2d* str;			// Shear stress tensor
		tensor2d* stn;			// Total strain tensor
		scalar4* normal_m;       // Normal direction and friction info for material
		tensor2dful* contact_mag;	// Contact friction x y z + Normal x y z + Vel of nearest j of i / Exp vel of j x y z
		scalar4* contact_var;  // Integrate gker + Distance ij + Shift with contact or not + Contact structure ID

		// Arrays for second order integration
		scalar4* pospres_ini;
		scalar4* velrhop_ini;
		tensor2d* str_ini;
		tensor2d* stn_ini;
		
		// Arrays for computation, no need for sorting
		scalar4* accad;			// Acceleration + density rate
		scalar4* spin;			// Components of spin tensor, for Jaumann stress rate
		tensor2d* acestn;		// Strain rate
		
		// Cache arrays for sorting and temporary
		int4* int4_cache;
		scalar* scalar_cache;
		scalar4* scalar4_cache;
		tensor2d* tensor2d_cache;

		// Variables related to time
		double time; // Current time
		int step;	 // Current step
		double output_time;    // Time interval since last output
		int step_interval;     // Steps since last output

		std::string TimeToString(double time);
		void GPUInfoExibition(int argc, char* argv[]);
		void InitializeParameters();
		void AllocateGpuMemory();
		void CopyDataFromHostToDevice();
		void CopyDataFromDeviceToHost();
		void TimerInitiation();
		void SortArraysPrimary();
		void SortArraysAll();	
		void StepCalculation();	
		void AdamiBoundary();
		void RigidBodyUpdate();
		void FindCornerParticle();
		void IntegrateParticleMaterial();
		void ParticleInteraction();
		void Predictor();
		void Corrector();
		void PredictorSection();
		void CorrectorSection();
		void StressRegularization();

		void StressSavingRegularization();
		void Saving(std::ofstream& time_history_ofs);
		void SummaryToCalculation();

	public:
		LoquatSph(std::string name);
		~LoquatSph();
		void RunSimulation(int argc, char* argv[]);
};


#endif
