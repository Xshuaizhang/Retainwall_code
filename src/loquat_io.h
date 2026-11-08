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
#ifndef __LOQUAT_IO_H__
#define __LOQUAT_IO_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <boost/variant.hpp>
#include <boost/filesystem.hpp>
#include "types.h"
#include "functions.h"
#include "tinyxml2.h"

class LoquatIo
{
	private:
		std::string project_name;

		Parameters* par_host;

		// Primary arrays
		int4* codes;
		scalar4* pospres;
		scalar4* velrhop;
		scalar4* accad;
		scalar4* idmassidx;
		tensor2d* str;
		tensor2d* stn;
		scalar4* normal_m;

		int4* int4_cache;
		scalar4* scalar4_cache;
		tensor2d* tensor2d_cache;

		std::vector<scalar> vec_max_disp;

		boost::variant<std::string, int, scalar, model_mat, dimension, bool> CheckParameter
		(
			std::string parameter_name,	std::map<std::string, 
			boost::variant<std::string, int, scalar, model_mat, dimension, bool> > parameters_from_xml
		);

		void SortInt4(int np, std::vector<int> sort_idx, int4* data, int4* cache);
		void SortScalar4(int np, std::vector<int> sort_idx, scalar4* data, scalar4* cache);
		void SortTensor2d(int np, std::vector<int> sort_idx, tensor2d* data, tensor2d* cache);

	public:
		RigidMotion* MotionInfoSimple;
		MotionTable* MotionInfoALL;
		LoquatIo(std::string name);

		~LoquatIo();

		Parameters* GetParameters() { return par_host; }
		int4* GetCodes() { return codes; }
		scalar4* GetPosPres() {return pospres; }
		scalar4* GetVelRhop() {return velrhop; }
		scalar4* GetAccad() {return accad; }
		scalar4* GetIdMassIdx() {return idmassidx; }
		tensor2d* GetStr() {return str; }
		tensor2d* GetStn() {return stn; }
		scalar4* GetNormalMaterial() {return normal_m; }
		std::vector<scalar> GetVecMaxDisp() {return vec_max_disp; }
		void LoadParameters();
		void LoadParticles();
		void SaveXml();
		void SaveParticles();
		void GetDisplacement();
		void ResetInitialPosition();
};


#endif
