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

#include "loquat_io.h"

LoquatIo::LoquatIo(std::string name)
{
    project_name = name;

	codes = NULL;
    pospres = NULL;
    velrhop = NULL;
	accad = NULL;
	idmassidx = NULL;  
    str = NULL;
    stn = NULL;
	normal_m = NULL;
	int4_cache = NULL;
    scalar4_cache = NULL;
    tensor2d_cache = NULL;
	par_host = NULL;

	MotionInfoSimple = NULL;
	MotionInfoALL = NULL;
}


LoquatIo::~LoquatIo()
{
	if (codes != NULL) cudaFreeHost(codes);
    if(pospres != NULL) cudaFreeHost(pospres);
    if(velrhop != NULL) cudaFreeHost(velrhop);
	if(accad != NULL) cudaFreeHost(accad);
	if(idmassidx != NULL) cudaFreeHost(idmassidx);
    if(str != NULL) cudaFreeHost(str);
    if(stn != NULL) cudaFreeHost(stn);  
    if(normal_m != NULL) cudaFreeHost(normal_m);
	if(int4_cache != NULL) cudaFreeHost(int4_cache);
    if(scalar4_cache != NULL) cudaFreeHost(scalar4_cache);
    if(tensor2d_cache != NULL) cudaFreeHost(tensor2d_cache);
}


// Check if a parameter exists, if it exists, return its value
boost::variant<std::string, int, scalar, model_mat, dimension, bool> LoquatIo::CheckParameter(
	std::string parameter_name,
	std::map<std::string, boost::variant<std::string, int, scalar, model_mat, dimension, bool> > parameters_from_xml)
{
	if(parameters_from_xml.count(parameter_name) == 0)
	{
		std::cout << "Error! " << parameter_name << " does not exist in the xml file!" << std::endl;
		exit(EXIT_FAILURE);
	}
	return parameters_from_xml.at(parameter_name);
}


void LoquatIo::LoadParameters()
{
	std::cout << std::endl;
	std::cout << "Loading parameters ..." << std::endl;
	
	std::string xml_name(project_name);
	xml_name.append(".xml");
	
	tinyxml2::XMLDocument xml;
	xml.LoadFile(xml_name.c_str());
	
	if(xml.Error()){
		std::cout << "Error! Cannot load the parameter file: " << xml_name << std::endl;
		std::cout << "Check the XML file, the names much match!" << std::endl;
		exit(EXIT_FAILURE);
	}
	else{
		std::cout << "The parameter file \"" << xml_name <<"\" is loaded sucessfully" << std::endl;
		std::cout << "Parsing the parameters" << std::endl;
		std::cout << std::endl;
	}
	
	tinyxml2::XMLElement* xml_parameters = xml.FirstChildElement();
	std::map<std::string, boost::variant<std::string, int, scalar, model_mat, dimension, bool> > parameters_from_xml;
	for(tinyxml2::XMLElement* iterator = xml_parameters->FirstChildElement(); iterator != NULL; iterator = iterator->NextSiblingElement()){
		// The name of the parameter
		std::string parameter_name(iterator->Name());
		
		// The actual value of the parameter
		boost::variant<std::string, int, scalar, model_mat, dimension, bool> parameter_value;
		if(iterator->Attribute("type", "int")){
			parameter_value = atoi(iterator->GetText());
		}
		else if(iterator->Attribute("type", "scalar")){
			parameter_value = (scalar)atof(iterator->GetText());
		}
		else if (iterator->Attribute("type", "model_mat")) {
			parameter_value = (model_mat)atoi(iterator->GetText());
		}
		else if (iterator->Attribute("type", "dimension")) {
			parameter_value = (dimension)atoi(iterator->GetText());
		}
		else if (iterator->Attribute("type", "bool")) {
			parameter_value = (bool)atof(iterator->GetText());
		}
		else{
			std::cout << "Wrong data type for " << parameter_name << "!!!" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		std::cout << parameter_name << " = " << parameter_value << std::endl;
		parameters_from_xml.insert(std::pair<std::string, boost::variant<std::string, int, scalar, model_mat, dimension, bool> >(parameter_name, parameter_value));
	}

	par_host = new Parameters;

	// Configuration
	par_host->dim = boost::get<dimension>(CheckParameter("dimension-of-problem", parameters_from_xml));
	par_host->n_motion = boost::get<int>(CheckParameter("number-of-motion", parameters_from_xml));
	par_host->periodbound = boost::get<bool>(CheckParameter("period-bound", parameters_from_xml));
	if (par_host->periodbound) {
		par_host->planeleft.x = boost::get<scalar>(CheckParameter("planeleft-A", parameters_from_xml));
		par_host->planeleft.y = boost::get<scalar>(CheckParameter("planeleft-B", parameters_from_xml));
		par_host->planeleft.z = boost::get<scalar>(CheckParameter("planeleft-C", parameters_from_xml));
		par_host->planeleft.w = boost::get<scalar>(CheckParameter("planeleft-D", parameters_from_xml));
		par_host->planeright.x = boost::get<scalar>(CheckParameter("planeright-A", parameters_from_xml));
		par_host->planeright.y = boost::get<scalar>(CheckParameter("planeright-B", parameters_from_xml));
		par_host->planeright.z = boost::get<scalar>(CheckParameter("planeright-C", parameters_from_xml));
		par_host->planeright.w = boost::get<scalar>(CheckParameter("planeright-D", parameters_from_xml));
	}
	par_host->np = boost::get<int>(CheckParameter("number-of-particles", parameters_from_xml));
	par_host->npb = boost::get<int>(CheckParameter("number-of-boundary-particles", parameters_from_xml));
	par_host->npm = boost::get<int>(CheckParameter("number-of-material-particles", parameters_from_xml));
	if(par_host->np != par_host->npb + par_host->npm){
		std::cout << "Error! Wrong number of particles ..." << std::endl;
		exit(EXIT_FAILURE);
	}

	par_host->domain_size.x = boost::get<scalar>(CheckParameter("domain-size-x", parameters_from_xml));
	par_host->domain_size.y = boost::get<scalar>(CheckParameter("domain-size-y", parameters_from_xml));
	par_host->domain_size.z = boost::get<scalar>(CheckParameter("domain-size-z", parameters_from_xml));
	par_host->domain_min.x = boost::get<scalar>(CheckParameter("domain-min-x", parameters_from_xml));
	par_host->domain_min.y = boost::get<scalar>(CheckParameter("domain-min-y", parameters_from_xml));
	par_host->domain_min.z = boost::get<scalar>(CheckParameter("domain-min-z", parameters_from_xml));

	par_host->dr = boost::get<scalar>(CheckParameter("particle-size", parameters_from_xml));
	par_host->hdr = boost::get<scalar>(CheckParameter("h-over-dr", parameters_from_xml));

	par_host->gravity.x = boost::get<scalar>(CheckParameter("acceleration-x", parameters_from_xml));
	par_host->gravity.y = boost::get<scalar>(CheckParameter("acceleration-y", parameters_from_xml));
	par_host->gravity.z = boost::get<scalar>(CheckParameter("acceleration-z", parameters_from_xml));

	// Material parameters
	par_host->rho0_m = boost::get<scalar>(CheckParameter("reference-density-material", parameters_from_xml));	
	par_host->damping = boost::get<scalar>(CheckParameter("damping", parameters_from_xml));	
	par_host->n_model = boost::get<int>(CheckParameter("number-of-model", parameters_from_xml));
	for (int i = 0; i < par_host->n_model; i++) {
		for (int j = 0; j < MATERIAL_SIZE_MAX; j++) {
			std::stringstream model_id;
			model_id << "model-" << i << "-par-" << j;
			par_host->model_par[i][j] = boost::get<scalar>(CheckParameter(model_id.str(), parameters_from_xml));
		}	
	}

	// Numerical stabilization techniques
	par_host->stress_shepard_filter = boost::get<int>(CheckParameter("stress-shepard-filter", parameters_from_xml));
	par_host->dynamic_time_step = boost::get<int>(CheckParameter("dynamic-time-step", parameters_from_xml));
	par_host->integration_method = boost::get<int>(CheckParameter("integration-method", parameters_from_xml));
	par_host->viscosity_type = boost::get<int>(CheckParameter("viscosity-type", parameters_from_xml));
	if(par_host->viscosity_type == 1){
		par_host->artificial_viscosity_alpha_m = boost::get<scalar>(CheckParameter("artificial-viscosity-alpha-material", parameters_from_xml));
	}
	par_host->cs_m = boost::get<scalar>(CheckParameter("speed-of-sound-material", parameters_from_xml));
	par_host->cfl = boost::get<scalar>(CheckParameter("cfl-number", parameters_from_xml));

	par_host->artificial_pressure_coefficient_material = boost::get<scalar>(CheckParameter("artificial-pressure-coefficient-material", parameters_from_xml));
	// Simulation output
	par_host->time_max = boost::get<scalar>(CheckParameter("time-max", parameters_from_xml));
	par_host->time = static_cast<double>(boost::get<scalar>(CheckParameter("time", parameters_from_xml)));
	par_host->dt = boost::get<scalar>(CheckParameter("time-step", parameters_from_xml));
	par_host->output_frequency = boost::get<scalar>(CheckParameter("output-frequency", parameters_from_xml));
	par_host->output_number = boost::get<int>(CheckParameter("output-number", parameters_from_xml));
	par_host->output_format = boost::get<int>(CheckParameter("output-format", parameters_from_xml));

	std::cout << " " << std::endl;
	std::cout << "Finish reading parameters" << std::endl;
	std::cout << " " << std::endl;
}


void LoquatIo::LoadParticles()
{
	std::cout << "Loading particle data ..." << std::endl;

	cudaHostAlloc((void **)&codes, par_host->np * sizeof(int4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&pospres, par_host->np * sizeof(scalar4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&velrhop, par_host->np * sizeof(scalar4), cudaHostAllocDefault);
	cudaHostAlloc((void **)&accad, par_host->np * sizeof(scalar4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&idmassidx, par_host->np * sizeof(scalar4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&str, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaHostAllocDefault);
    cudaHostAlloc((void **)&stn, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaHostAllocDefault);
	cudaHostAlloc((void **)&normal_m, par_host->npb * sizeof(scalar4), cudaHostAllocDefault);

	cudaHostAlloc((void **)&int4_cache, par_host->np * sizeof(int4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&scalar4_cache, par_host->np * sizeof(scalar4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&tensor2d_cache, (par_host->npb + par_host->npm) * sizeof(tensor2d), cudaHostAllocDefault);

	std::string dat_name(project_name);
    dat_name.append(".dat");

	std::ifstream input_stream;
	input_stream.open(dat_name.c_str());
	if(input_stream)
	{
		for(int i = 0; i < par_host->npb; i++) // Boundary
		{
			// x y z p vx vy vz rho id mass ep phi str stn
			input_stream >> codes[i].x >> codes[i].y >> codes[i].z >> codes[i].w
						 >> pospres[i].x >> pospres[i].y >> pospres[i].z >> pospres[i].w 
						 >> velrhop[i].x >> velrhop[i].y >> velrhop[i].z >> velrhop[i].w 
						 >> idmassidx[i].x >> idmassidx[i].y >> idmassidx[i].w 
						 >> str[i].xx >> str[i].yy >> str[i].zz >> str[i].xy >> str[i].xz >> str[i].yz 
						 >> stn[i].xx >> stn[i].yy >> stn[i].zz >> stn[i].xy >> stn[i].xz >> stn[i].yz
						 >> normal_m[i].x >> normal_m[i].y >> normal_m[i].z >> normal_m[i].w;

			idmassidx[i].z = static_cast<scalar>(i);
		}

		for(int i = par_host->npb; i < par_host->npb + par_host->npm; i++) // Material
		{
			// x y z p vx vy vz rho id mass ep phi str stn
			input_stream >> codes[i].x >> codes[i].y >> codes[i].z >> codes[i].w
						 >> pospres[i].x >> pospres[i].y >> pospres[i].z >> pospres[i].w 
						 >> velrhop[i].x >> velrhop[i].y >> velrhop[i].z >> velrhop[i].w 
						 >> idmassidx[i].x >> idmassidx[i].y >> idmassidx[i].w 
						 >> str[i].xx >> str[i].yy >> str[i].zz >> str[i].xy >> str[i].xz >> str[i].yz 
						 >> stn[i].xx >> stn[i].yy >> stn[i].zz >> stn[i].xy >> stn[i].xz >> stn[i].yz;

			idmassidx[i].z = static_cast<scalar>(i);
		}

		printf("Finish reading boundary and soil particles ...\n");

		input_stream.close();
	}
	else
	{
		std::cout << "Error! Cannot open " << dat_name << " !" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (par_host->n_motion > 0) {
		cudaHostAlloc((void**)&MotionInfoALL, par_host->n_motion * sizeof(MotionTable), cudaHostAllocDefault);
		cudaHostAlloc((void**)&MotionInfoSimple, par_host->n_motion * sizeof(RigidMotion), cudaHostAllocDefault);

		std::string dat_name(project_name);
		dat_name.append(".motion");

		std::ifstream input_stream;
		input_stream.open(dat_name.c_str());
		if (input_stream) {
			for (int i = 0; i < par_host->n_motion; i++) {
				input_stream >> MotionInfoALL[i].is_translation >> MotionInfoALL[i].is_rotation;
				if (MotionInfoALL[i].is_translation) {
					input_stream >> MotionInfoALL[i].table_size_trans;
					for (int j = 0; j < MotionInfoALL[i].table_size_trans; j++) input_stream >> MotionInfoALL[i].trans_time[j] >> MotionInfoALL[i].trans_vel[j].x >> MotionInfoALL[i].trans_vel[j].y >> MotionInfoALL[i].trans_vel[j].z 
						>> MotionInfoALL[i].trans_acc[j].x >> MotionInfoALL[i].trans_acc[j].y >> MotionInfoALL[i].trans_acc[j].z;
				}
				if (MotionInfoALL[i].is_rotation) {
					input_stream >> MotionInfoALL[i].pattern
								 >> MotionInfoALL[i].pos_rp.x >> MotionInfoALL[i].pos_rp.y >> MotionInfoALL[i].pos_rp.z
						         >> MotionInfoALL[i].rotat_vel.x >> MotionInfoALL[i].rotat_vel.y >> MotionInfoALL[i].rotat_vel.z >> MotionInfoALL[i].table_size_rotat;
					for (int j = 0; j < MotionInfoALL[i].table_size_rotat; j++) input_stream >> MotionInfoALL[i].rotat_time[j] >> MotionInfoALL[i].rotat_acc[j].x >> MotionInfoALL[i].rotat_acc[j].y >> MotionInfoALL[i].rotat_acc[j].z;
				}
				par_host->mv[i].is_trans = MotionInfoALL[i].is_translation;				
				par_host->mv[i].is_rotat = MotionInfoALL[i].is_rotation;

				if (par_host->mv[i].is_trans) {
					par_host->mv[i].trans_vel = MotionInfoALL[i].trans_vel[0];
				}
				else{
					par_host->mv[i].trans_vel = make_scalar3(0.0);
				}
				if (par_host->mv[i].is_rotat) {
					par_host->mv[i].pattern = shape(MotionInfoALL[i].pattern);
					par_host->mv[i].pos_rp = MotionInfoALL[i].pos_rp;
					par_host->mv[i].rotat_vel = MotionInfoALL[i].rotat_vel;
				}
			}
			std::cout << "Finish loading motion infomation ..." << std::endl;
			input_stream.close();
		}
		else{
			std::cout << "Error! Cannot open " << dat_name << " !" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}


void LoquatIo::SaveXml()
{
	std::stringstream ss;
	ss << project_name << "/output_" << par_host->output_number << ".xml";
	std::string filename = ss.str();

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if(ofs)
	{
		ofs << "<?xml version=\"1.0\" encoding=\"utf-8\" ?>" << std::endl;
		ofs << "<parameters>" << std::endl;

		// Configuration
		ofs << "  <dimension-of-problem type=\"dimension\">" << par_host->dim << "</dimension-of-problem>" << std::endl;
		ofs << "  <number-of-motion type=\"int\">" << par_host->n_motion << "</number-of-motion>" << std::endl;
		ofs << "  <period-bound type=\"bool\">" << par_host->periodbound << "</period-bound>" << std::endl;
		if (par_host->periodbound) {
			ofs << "  <planeleft-A type=\"scalar\">" << par_host->planeleft.x << "</planeleft-A>" << std::endl;
			ofs << "  <planeleft-B type=\"scalar\">" << par_host->planeleft.y << "</planeleft-B>" << std::endl;
			ofs << "  <planeleft-C type=\"scalar\">" << par_host->planeleft.z << "</planeleft-C>" << std::endl;
			ofs << "  <planeleft-D type=\"scalar\">" << par_host->planeleft.w << "</planeleft-D>" << std::endl;
			ofs << "  <planeright-A type=\"scalar\">" << par_host->planeright.x << "</planeright-A>" << std::endl;
			ofs << "  <planeright-B type=\"scalar\">" << par_host->planeright.y << "</planeright-B>" << std::endl;
			ofs << "  <planeright-C type=\"scalar\">" << par_host->planeright.z << "</planeright-C>" << std::endl;
			ofs << "  <planeright-D type=\"scalar\">" << par_host->planeright.w << "</planeright-D>" << std::endl;
		}
		ofs << "  <number-of-particles type=\"int\">" << par_host->np << "</number-of-particles>" << std::endl;
		ofs << "  <number-of-boundary-particles type=\"int\">" << par_host->npb << "</number-of-boundary-particles>" << std::endl;
		ofs << "  <number-of-material-particles type=\"int\">" << par_host->npm << "</number-of-material-particles>" << std::endl;
		
		ofs << "  <domain-size-x type=\"scalar\">" << par_host->domain_size.x << "</domain-size-x>" << std::endl;
		ofs << "  <domain-size-y type=\"scalar\">" << par_host->domain_size.y << "</domain-size-y>" << std::endl;
		ofs << "  <domain-size-z type=\"scalar\">" << par_host->domain_size.z << "</domain-size-z>" << std::endl;
		ofs << "  <domain-min-x type=\"scalar\">" << par_host->domain_min.x << "</domain-min-x>" << std::endl;
		ofs << "  <domain-min-y type=\"scalar\">" << par_host->domain_min.y << "</domain-min-y>" << std::endl;
		ofs << "  <domain-min-z type=\"scalar\">" << par_host->domain_min.z << "</domain-min-z>" << std::endl;

		ofs << "  <particle-size type=\"scalar\">" << par_host->dr << "</particle-size>" << std::endl;
		ofs << "  <h-over-dr type=\"scalar\">" << par_host->hdr << "</h-over-dr>" << std::endl;

		ofs << "  <acceleration-x type=\"scalar\">" << par_host->gravity.x << "</acceleration-x>" << std::endl;
		ofs << "  <acceleration-y type=\"scalar\">" << par_host->gravity.y << "</acceleration-y>" << std::endl;
		ofs << "  <acceleration-z type=\"scalar\">" << par_host->gravity.z << "</acceleration-z>" << std::endl;

		// Material parameters
		ofs << "  <reference-density-material type=\"scalar\">" << par_host->rho0_m << "</reference-density-material>" << std::endl;
		ofs << "  <damping type=\"scalar\">" << par_host->damping << "</damping>" << std::endl;
		ofs << "  <number-of-model type=\"int\">" << par_host->n_model << "</number-of-model>" << std::endl;
		for (int i = 0; i < par_host->n_model; i++) {
			for (int j = 0; j < MATERIAL_SIZE_MAX; j++) {
				std::stringstream model_id;
				model_id << "model-" << i << "-par-" << j;
				ofs << "  <" << model_id.str() << " type=\"scalar\">" << par_host->model_par[i][j] << "</" << model_id.str() << ">" << std::endl;
			}
		}

		// Numerical stabilization techniques
		ofs << "  <stress-shepard-filter type=\"int\">" << par_host->stress_shepard_filter << "</stress-shepard-filter>" << std::endl;
		ofs << "  <dynamic-time-step type=\"int\">" << par_host->dynamic_time_step << "</dynamic-time-step>" << std::endl;
		ofs << "  <integration-method type=\"int\">" << par_host->integration_method << "</integration-method>" << std::endl;
		ofs << "  <viscosity-type type=\"int\">" << par_host->viscosity_type << "</viscosity-type>" << std::endl;
		if(par_host->viscosity_type == 1){
			ofs << "  <artificial-viscosity-alpha-material type=\"scalar\">" << par_host->artificial_viscosity_alpha_m << "</artificial-viscosity-alpha-material>" << std::endl;
		}
		ofs << "  <speed-of-sound-material type=\"scalar\">" << par_host->cs_m << "</speed-of-sound-material>" << std::endl;
		ofs << "  <cfl-number type=\"scalar\">" << par_host->cfl << "</cfl-number>" << std::endl;
		ofs << "  <artificial-pressure-coefficient-material type=\"scalar\">" << par_host->artificial_pressure_coefficient_material << "</artificial-pressure-coefficient-material>" << std::endl;

		// Simulation output
		ofs << "  <time-max type=\"scalar\">" << par_host->time_max << "</time-max>" << std::endl;
		ofs << "  <time type=\"scalar\">" << static_cast<scalar>(par_host->time) << "</time>" << std::endl;
		ofs << "  <time-step type=\"scalar\">" << par_host->dt << "</time-step>" << std::endl;
		ofs << "  <output-frequency type=\"scalar\">" << par_host->output_frequency << "</output-frequency>" << std::endl;
		ofs << "  <output-number type=\"int\">" << par_host->output_number << "</output-number>" << std::endl;
		ofs << "  <output-format type=\"int\">" << par_host->output_format << "</output-format>" << std::endl;

		ofs << "</parameters>" << std::endl;

		ofs.close();
	}
	else
	{
		std::cout << "Error! Cannot write file " << filename << " !" << std::endl;
	}
}

void LoquatIo::SortInt4(int size, std::vector<int> sort_idx, int4* data, int4* cache)
{
	memcpy(cache, data, size * sizeof(int4));
	for (int i = 0; i < size; i++)
	{
		data[i] = cache[sort_idx[i]];
	}
}

void LoquatIo::SortScalar4(int size, std::vector<int> sort_idx, scalar4* data, scalar4* cache)
{
    memcpy(cache, data, size * sizeof(scalar4));
    for(int i = 0; i < size; i++)
    {
        data[i] = cache[sort_idx[i]];
    }
}


void LoquatIo::SortTensor2d(int size, std::vector<int> sort_idx, tensor2d* data, tensor2d* cache)
{
    memcpy(cache, data, size * sizeof(tensor2d));
    for(int i = 0; i < size; i++)
    {
        data[i] = cache[sort_idx[i]];
    }
}


void LoquatIo::SaveParticles(){
	if(par_host->output_format == 1) // Save as plain data file
	{
		std::stringstream ss;
		ss << project_name << "/" << par_host->output_number << ".dat" << std::endl;
		std::string filename = ss.str();

		std::ofstream ofs(filename.c_str(), std::ofstream::out);
		if(ofs)
		{
			for(int i = 0; i < par_host->npb + par_host->npm; i++)
			{
				// x y z p vx vy vz rho id mass ep phi
				ofs << codes[i].x << " " << codes[i].y << " " << codes[i].z << " " << codes[i].w << " "
					<< pospres[i].x << " " << pospres[i].y << " " << pospres[i].z << " " << pospres[i].w << " "
					<< velrhop[i].x << " " << velrhop[i].y << " " << velrhop[i].z << " " << velrhop[i].w << " "
					<< idmassidx[i].x << " " << idmassidx[i].y << " " << idmassidx[i].w << " "
					<< str[i].xx << " " << str[i].yy << " " << str[i].zz << " " << str[i].xy << " " << str[i].xz << " " << str[i].yz << " "
					<< stn[i].xx << " " << stn[i].yy << " " << stn[i].zz << " " << stn[i].xy << " " << stn[i].xz << " " << stn[i].yz << " "
					<< normal_m[i].x << " " << normal_m[i].y << " " << normal_m[i].z << " " << normal_m[i].w << std::endl;
			}

			for(int i = par_host->npb + par_host->npm; i < par_host->np; i++)
            {
				// x y z p vx vy vz rho id mass ep phi
				ofs << codes[i].x << " " << codes[i].y << " " << codes[i].z << " " << codes[i].w << " "
					<< pospres[i].x << " " << pospres[i].y << " " << pospres[i].z << " " << pospres[i].w << " "
					<< velrhop[i].x << " " << velrhop[i].y << " " << velrhop[i].z << " " << velrhop[i].w << " "
					<< idmassidx[i].x << " " << idmassidx[i].y << " " << idmassidx[i].w << " "
					<< str[i].xx << " " << str[i].yy << " " << str[i].zz << " " << str[i].xy << " " << str[i].xz << " " << str[i].yz << " "
					<< stn[i].xx << " " << stn[i].yy << " " << stn[i].zz << " " << stn[i].xy << " " << stn[i].xz << " " << stn[i].yz << " " << std::endl;
			}
            ofs.close();
        }
        else
        {
            std::cout << "Error! Cannot write in " << filename << " !" << std::endl;
        }
    }

    else if(par_host->output_format == 2) // Save as vtu file for ParaView
    {
		std::clock_t clock_sorting = std::clock();

	    std::vector<int> sort_key;
    	std::vector<int> sort_idx;
	    sort_key.resize(par_host->np);
    	sort_idx.resize(par_host->np);

	    for(int i = 0; i < par_host->np; i++)
	    {
			int id = static_cast<int>(idmassidx[i].x);
		    sort_key[i] = id;
		    sort_idx[i] = i;
	    }

	    thrust::sort_by_key(&sort_key[0], &sort_key[0] + par_host->np, &sort_idx[0], thrust::less<int>());

		SortInt4(par_host->np, sort_idx, codes, int4_cache);
	    SortScalar4(par_host->np, sort_idx, pospres, scalar4_cache);
    	SortScalar4(par_host->np, sort_idx, velrhop, scalar4_cache);
		SortScalar4(par_host->np, sort_idx, accad, scalar4_cache);
		SortScalar4(par_host->np, sort_idx, idmassidx, scalar4_cache);
		SortScalar4(par_host->npb, sort_idx, normal_m, scalar4_cache);
	    SortTensor2d(par_host->npb + par_host->npm, sort_idx, str, tensor2d_cache);
	    SortTensor2d(par_host->npb + par_host->npm, sort_idx, stn, tensor2d_cache);

	    std::vector<int> ids;
	    std::vector<int> id_positions;
	    int current_id = static_cast<int>(idmassidx[0].x);
	    ids.push_back(current_id);
	    id_positions.push_back(0);
	    for(int i = 0; i < par_host->np; i++)
	    {
            int id_i = static_cast<int>(idmassidx[i].x);
		    if(current_id != id_i)
    		{
			    current_id = id_i;
			    ids.push_back(current_id);
			    id_positions.push_back(i);
		    }
	    }

		std::vector<int> number_of_particles;
		std::cout << "|";
	    for(int i = 0; i < ids.size(); i++)
	    {
		    if(i < ids.size() - 1)
		    {
			    number_of_particles.push_back(id_positions[i + 1] - id_positions[i]);
		    }
		    else if(i == ids.size() - 1)
		    {
			    number_of_particles.push_back(par_host->np - id_positions[i]);
			}
			std::cout << ids[i] << ": " << number_of_particles[i] << "|";
		}
		std::cout << std::endl;

		double time_sorting = (std::clock() - clock_sorting) / static_cast<double>(CLOCKS_PER_SEC);

		std::clock_t clock_writing = std::clock();
	    for(int i = 0; i < ids.size(); i++)
    	{
		    std::stringstream ss;
		    ss << project_name << "/" << ids[i] << "_" << par_host->output_number << ".vtu";
		    std::string filename = ss.str();

		    std::ofstream ofs(filename.c_str(), std::ofstream::out);
		    if(ofs)
		    {
			    ofs << "<?xml version=\"1.0\"?>" << std::endl;
			    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << std::endl;
			    ofs << "  <UnstructuredGrid GhostLevel=\"0\">" << std::endl;
			    ofs << "    <Piece NumberOfPoints=\"" << number_of_particles[i] << "\" NumberOfCells=\"" << number_of_particles[i] << "\" Name=\"" << ids[i] << "\">" << std::endl;
    			ofs << "      <PointData>" << std::endl;

			    // OriginalID
				ofs << "        <DataArray type=\"Float64\" Name=\"Material\">" << std::endl;
				for (int j = 0; j < number_of_particles[i]; j++)
				{
					int index = id_positions[i] + j;
					ofs << codes[index].w << " ";
				}
				ofs << "</DataArray>" << std::endl;

			    ofs << "        <DataArray type=\"Float64\" Name=\"Ori_ID\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << idmassidx[index].w << " ";
    			}
    			ofs << "</DataArray>" << std::endl;

    			// Velocity
			    ofs << "        <DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << velrhop[index].x << " " << velrhop[index].y << " " << velrhop[index].z << " ";
			    }
    			ofs << "</DataArray>" << std::endl;

				// Acceleration
			    ofs << "        <DataArray type=\"Float64\" Name=\"Acceleration\" NumberOfComponents=\"3\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << accad[index].x << " " << accad[index].y << " " << accad[index].z << " ";
			    }
    			ofs << "</DataArray>" << std::endl;

    			// Density
			    ofs << "        <DataArray type=\"Float64\" Name=\"Density\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << velrhop[index].w << " ";
    			}
				ofs << "</DataArray>" << std::endl;
				
				// Mass
			    ofs << "        <DataArray type=\"Float64\" Name=\"Mass\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << idmassidx[index].y << " ";
    			}
			    ofs << "</DataArray>" << std::endl;

				if(ids[i] < MATERIAL_NUMBER){// Boundary
					// Normal
					ofs << "        <DataArray type=\"Float64\" Name=\"Normal\" NumberOfComponents=\"3\">" << std::endl;
					for(int j = 0; j < number_of_particles[i]; j++)
					{
						int index = id_positions[i] + j;
						ofs << normal_m[index].x << " " << normal_m[index].y << " " << normal_m[index].z << " ";
					}
					ofs << "</DataArray>" << std::endl;
				}

				// Stress
				ofs << "        <DataArray type=\"Float64\" Name=\"Stress\" NumberOfComponents=\"6\">" << std::endl;
				for(int j = 0; j < number_of_particles[i]; j++){
					int index = id_positions[i] + j;
					ofs << str[index].xx << " " << str[index].yy << " " << str[index].zz << " " << str[index].xy << " " << str[index].yz << " " << str[index].xz << " ";
				}
				ofs << "</DataArray>" << std::endl;

				// Strain
				ofs << "        <DataArray type=\"Float64\" Name=\"Strain\" NumberOfComponents=\"6\">" << std::endl;
				for(int j = 0; j < number_of_particles[i]; j++){
					int index = id_positions[i] + j;
					ofs << stn[index].xx << " " << stn[index].yy << " " << stn[index].zz << " " << stn[index].xy << " " << stn[index].yz << " " << stn[index].xz << " ";
				}
				ofs << "</DataArray>" << std::endl;


                ofs << "      </PointData>" << std::endl;

                ofs << "      <CellData>" << std::endl;
                ofs << "      </CellData>" << std::endl;

                // Position
                ofs << "      <Points>" << std::endl;
                ofs << "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
					int index = id_positions[i] + j;
					ofs << pospres[index].x << " " << pospres[index].y << " " << pospres[index].z << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "      </Points>" << std::endl;

                // Cell connectivity
                ofs << "      <Cells>" << std::endl;
                ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    ofs << j << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\"> " << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    ofs << j+1 << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "        <DataArray type=\"Int64\" Name=\"types\" format=\"ascii\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    ofs << 1 << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "      </Cells>" << std::endl;

                ofs << "    </Piece>" << std::endl;
                ofs << "  </UnstructuredGrid>" << std::endl;
                ofs << "</VTKFile>" << std::endl;

                ofs.close();
		    }
		    else
		    {
			    std::cout << "Error! Cannot write file " << filename << " !" << std::endl;
    		}
		}
		
		double time_writing = (std::clock() - clock_writing) / static_cast<double>(CLOCKS_PER_SEC);
		printf("Save time: sorting = %0.4f, writing = %0.4f\n", time_sorting, time_writing);
    }
}



