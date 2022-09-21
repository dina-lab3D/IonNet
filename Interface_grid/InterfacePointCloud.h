/**
 * \file InterfacePointCloud.h
 * \brief
 *
 * \authors Dina Schneidman
 * Copyright 2007-2013 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMP_INTERFACEPOINTCLOUD_H
#define IMP_INTERFACEPOINTCLOUD_H

#include "Vector3.h"
#include "RigidTrans3.h"
#include "ChemMolecule.h"

#include <vector>
#include <iostream>
#include <fstream>
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"

class InterfacePointCloud {
public:
    InterfacePointCloud(const ChemMolecule &mol);

    void outputH5grid(const std::string path, const std::string dataset_name);

private:
    std::vector<std::vector<float> > data_; // N x 21 (x,y,z, mol2type, charge, asa, radius)
};

#endif /* IMP_INTERFACEPOINTCLOUD_H */
