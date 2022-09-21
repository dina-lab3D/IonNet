#include "InterfaceGrid.h"


InterfaceGrid::InterfaceGrid(const ChemMolecule &mol, const Vector3 &origin,
                             int nx, int ny, int nz, float voxelSize) :
        voxelSize_(voxelSize), origin_(origin), nx_(nx), ny_(ny), nz_(nz), coordsComputed_(false)
{

    n_ = nx_ * ny_ * nz_;
    data_.clear();
    std::vector<float> v(n_, 0.0);

    int FEATURE_SIZE = 16; // 15 mol2 types, charge and asa
						   // + radius
    data_.resize(FEATURE_SIZE, v);

    top_[0] = origin_[0] + nx_ * voxelSize_;
    top_[1] = origin_[1] + ny_ * voxelSize_;
    top_[2] = origin_[2] + nz_ * voxelSize_;

    for (unsigned int i = 0; i < mol.size(); i++) {
        placeAtom(mol[i]);
    }
    isDataAllocated_ = true;
}

void InterfaceGrid::placeAtom(const ChemAtom &atom) {

    Vector3 point = atom.position();
    float radius = atom.getRadius();
    MOL2_TYPE mol2type = atom.getMol2Type();
    float charge = atom.getCharge();
    float asa = atom.getASA();

    int x = (int) std::floor((point[0] - origin_[0]) / voxelSize_ + 0.5);
    int y = (int) std::floor((point[1] - origin_[1]) / voxelSize_ + 0.5);
    int z = (int) std::floor((point[2] - origin_[2]) / voxelSize_ + 0.5);

    if (x >= 0 && x < nx_ && y >= 0 && y < ny_ && z >= 0 && z < nz_) {

        int int_radius = (int) std::floor(radius / voxelSize_ + 0.5); // TODO: check if +1 is needed
        int int_radius2 = int_radius * int_radius;

        int i_bound, j_bound, k_bound;
        i_bound = int_radius;
        // iterate circle indices
        for (int i = -i_bound; i <= i_bound; i++) {
            j_bound = (int) sqrt(static_cast<double>(int_radius2 - i * i));
            for (int j = -j_bound; j <= j_bound; j++) {
                k_bound = (int) sqrt(static_cast<double>(int_radius2 - i * i - j * j));
                for (int k = -k_bound; k <= k_bound; k++) {
                    // int int_dist2 = i * i + j * j + k * k;

                    long l = getIndexForIntPoint(x + i, y + j, z + k);
////                    if(mol2type==3 || mol2type == 12 || mol2type == 14){
////                        std::cout << "Mol2 type not relevent for this task: " << mol2type << std::endl;
//                    }
                     if (l >= 0 && l < n_) {
                        data_[mol2type][l] = 1.0;
                        data_[3][l] = charge;
                        data_[15][l] = asa;
                    }
                }
            }
        }
    } else {
        std::cerr << "Can't fit the interface in the grid" << std::endl;
        throw "Can't fit the interface in the grid";
    }
}


void InterfaceGrid::computeCoordinates() {
    if (coordsComputed_) return;
    coords_.reserve(getNumberOfVoxels());
    int ix = 0, iy = 0, iz = 0;
    for (unsigned long i = 0; i < getNumberOfVoxels(); i++) {
        Vector3 p(ix, iy, iz);
        p *= voxelSize_;
        p += origin_;
        coords_.push_back(p);

        ix++;
        if (ix == nx_) {
            ix = 0;
            ++iy;
            if (iy == ny_) {
                iy = 0;
                ++iz;
            }
        }
    }
    coordsComputed_ = true;
}

void InterfaceGrid::normalize() {
    if (normalized_) return;
}

void InterfaceGrid::outputGridPdb(std::ostream &outstream) {
    computeCoordinates();
    std::cout << coords_.size() << std::endl;
    for (long l = 0; l < (long) coords_.size(); l++) {
        if (isDataPositive(l)) {
            Atom a(coords_[l], 'A', (l + 1) / 10, (l + 1) / 100, " CA ", 'A'); //Todo: 'Ala'
            outstream << a << std::endl;
        }
    }
}

void InterfaceGrid::outputH5grid(const std::string path, const std::string dataset_name) {
    using namespace HighFive;
    //example taken from https://github.com/BlueBrain/HighFive/blob/master/src/examples/create_dataset_double.cpp

    try {
        // Create a new file using the default property lists.
        File file(path, File::ReadWrite | File::Create | File::Truncate);

        // Define the size of our dataset: 2x6
        std::vector<size_t> dims(2);
        dims[0] = this->data_.size();
        dims[1] = this->data_[0].size();


        // Create the dataset
        DataSet dataset =
                file.createDataSet<float>(dataset_name, DataSpace(dims));

        // write it
        dataset.write(this->data_);

    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

}

std::vector<float> InterfaceGrid::to1dim() {
    size_t N1 = data_.size();
    size_t M1 = data_[0].size();

    std::vector<float> raw1(N1 * M1);
    for (size_t row = 0; row < N1; row++) {
        for (size_t col = 0; col < M1; col++) {
            raw1[row * M1 + col] = data_[row][col];
        }
    }

    return raw1;
}

void InterfaceGrid::outputGridNpy(const std::string file_name) {


    size_t N1 = data_.size();
    size_t M1 = data_[0].size();
    std::vector<float> raw1 = to1dim();
    cnpy::npy_save(file_name, &raw1[0], {N1, M1}, "w");
}

bool InterfaceGrid::isDataPositive(long index) {
    bool positive = false;

    for (unsigned int i = 0; i < data_.size(); i++) {
        if (data_[i][index] > 0) {
            positive = true;
            break;
        }
    }
    return positive;
}
