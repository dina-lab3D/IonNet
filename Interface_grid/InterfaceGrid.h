#ifndef INTERFACE_GRID_H
#define INTERFACE_GRID_H

#include "Vector3.h"
#include "RigidTrans3.h"
#include "ChemMolecule.h"

#include <vector>
#include <iostream>
#include <fstream>

#include <limits>
//For std::min/max
#include <algorithm>

// for 'map_list_of()'
#include <boost/assign/list_of.hpp>
#include <map>

#include <HashInterface.h>

//#include "cnpy.h"
#include <string.h>

#include <sys/stat.h>
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"

constexpr auto RES_TYPE_SIZE_START_LINE = __LINE__;
enum class ResidueType {
    ALA, //alanine
    ARG, //arginine
    ASN, //asparagine
    ASP, //aspartic acid
    CYS, //cysteine
    GLN, //glutamine
    GLU, //glutamic
    GLY, //glycine
    HIS, //histidine
    ILE, //isoleucine
    LEU, //leucine
    LYS, //lysine
    MET, //methionine
    PHE, //phenylalanine
    PRO, //proline
    SER, //serine
    THR, //threonine
    TRP, //tryptophan
    TYR, //tyrosine
    VAL, //valine
    backBone,
    sideChain
};
constexpr auto RES_TYPE_SIZE = __LINE__ - RES_TYPE_SIZE_START_LINE - 3;


class InterfaceGrid {
public:
    // GROUP: Constructors
    //InterfaceGrid(const ChemMolecule& mol, float voxelSize);

    InterfaceGrid(const ChemMolecule &mol, const Vector3 &origin, int nx, int ny, int nz, float voxelSize);

    // GROUP: Inspectors
    // map top corner
    Vector3 getTop() const { return top_; }

    // voxel size
    float getVoxelSize() const { return voxelSize_; }

    // map origin
    Vector3 getOrigin() const { return origin_; }

    // number of voxels in the map
    unsigned long getNumberOfVoxels() const { return n_; }

    // get value for coordinate
    std::vector<float> getValue(const Vector3 &point) const;

    // get value for index
    std::vector<float> getValue(long index) const {
        return data_[index];
    }

    // get for x,y,z index //TODO - can be private?
    std::vector<float> getValue(int x, int y, int z) const {
        long index = z * nx_ * ny_ + y * nx_ + x;
        return data_[index];
    }

    // gets a point corresponding to an index
    Vector3 getPointForIndex(long index) const { return coords_[index]; }

    // get index for coordinate, -1 is returned if outside the map
    long getIndexForPoint(const Vector3 &point) const {
        int x = (int) std::floor((point[0] - origin_[0]) / voxelSize_ + 0.5);
        int y = (int) std::floor((point[1] - origin_[1]) / voxelSize_ + 0.5);
        int z = (int) std::floor((point[2] - origin_[2]) / voxelSize_ + 0.5);
        if (x >= 0 && x < nx_ && y >= 0 && y < ny_ && z >= 0 && z < nz_)
            return z * nx_ * ny_ + y * nx_ + x;
        return -1;
    }

    // check if the point falls in the grid
    bool isPointInGrid(const Vector3 &point) const {
        int x = (int) std::floor((point[0] - origin_[0]) / voxelSize_ + 0.5);
        int y = (int) std::floor((point[1] - origin_[1]) / voxelSize_ + 0.5);
        int z = (int) std::floor((point[2] - origin_[2]) / voxelSize_ + 0.5);
        return (x >= 0 && x < nx_ && y >= 0 && y < ny_ && z >= 0 && z < nz_);
    }

    // GROUP: grid operations

    // mark the voxels belonging to the atom with the type
    void placeAtom(const ChemAtom &atom);

    void outputInterfaceAtoms(const ChemMolecule &mol1, const ChemMolecule &mol2, const float &thr, ChemMolecule &mol);

    void outputGridPdb(std::ostream &outstream);

    // void outputGridNpy(std::ostream& outstream);

    void outputGridNpy(const std::string path);

    std::vector<float> to1dim();

    std::vector<std::vector<float> > getData(){return data_;};

    void outputH5grid(const std::string path, const std::string dataset_name);

private:
    void computeCoordinates();

    long getIndexForIntPoint(int x, int y, int z) const {
        return z * nx_ * ny_ + y * nx_ + x;
    }

    void normalize();

    int string_to_residue_enum(const char *residue_3_letter) {
        std::string upper_cased = residue_3_letter;
        std::transform(upper_cased.begin(), upper_cased.end(), upper_cased.begin(), ::toupper);
        return int(string_enum_residue_map[upper_cased]);
    }

    bool isDataPositive(long index);

private:
    float voxelSize_;

    // Interface mol1AtomInterface_, mol2AtomInterface_;

    Vector3 origin_; // xMin, yMin, zMin
    int nx_, ny_, nz_;  // grid size (#voxels) in x,y,z dimensions
    long n_;
    Vector3 top_; // xMax, yMax, zMax


    // our data is vector<float> for ~20 residue types

    std::vector<std::vector<float> > data_;  // the order is ZYX (Z-slowest)
    //std::vector<float> data_;
    bool isDataAllocated_;

    bool normalized_ = true;

    // XYZ coordinates for each of the voxels centers (they are
    // precomputed and each one is of size nvox, where nvox is the
    // size of the map)
    std::vector<Vector3> coords_;
    // true if the locations have already been computed
    bool coordsComputed_;


    std::map<std::string, ResidueType> string_enum_residue_map = boost::assign::map_list_of
            ("ALA", ResidueType::ALA)("ARG", ResidueType::ARG)("ASN", ResidueType::ASN)
            ("ASP", ResidueType::ASP)("CYS", ResidueType::CYS)("GLN", ResidueType::GLN)
            ("GLU", ResidueType::GLU)("GLY", ResidueType::GLY)("HIS", ResidueType::HIS)
            ("ILE", ResidueType::ILE)("LEU", ResidueType::LEU)("LYS", ResidueType::LYS)
            ("MET", ResidueType::MET)("PHE", ResidueType::PHE)("PRO", ResidueType::PRO)
            ("SER", ResidueType::SER)("THR", ResidueType::THR)("TRP", ResidueType::TRP)
            ("TYR", ResidueType::TYR)("VAL", ResidueType::VAL);
};

#endif
