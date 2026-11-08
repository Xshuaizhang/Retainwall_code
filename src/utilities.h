/*********************************************************************************
 * Copyright (c) 2019, Chong Peng. All rights reserved.

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
#ifndef __UTILITIES__
#define __UTILITIES__

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "types.h"
#include "functions.h"

void GetSurface(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	const int* cell_beg,
	const int* cell_end);

void GetValuesAlongLine(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	tensor2d* stn,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end);

void GetSurfaceAtPoint(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	const int* cell_beg,
	const int* cell_end,
	std::ofstream& time_history_ofs);

void InterpolateResults(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end,
	std::ofstream& time_history_ofs);

void GetValuesAtPoint(
	std::string project_name,
	Parameters* par_host,
	Parameters* par_device,
	scalar4* pospres,
	scalar4* velrhop,
	scalar4* idmassidx,
	tensor2dful* contact_mag,
	scalar4* contact_var,
	tensor2d* str,
	int4* cellpos,
	const int* cell_beg,
	const int* cell_end,
	std::ofstream& time_history_ofs);

void TestKernelFunction(Parameters* par_host);

#endif