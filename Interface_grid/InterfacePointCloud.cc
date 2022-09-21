#include "InterfacePointCloud.h"


InterfacePointCloud::InterfacePointCloud(const ChemMolecule &mol) {
    // TODO make constant
    int FEATURE_SIZE = 18; // 15 mol2 types-3, charge and asa
    // and radius and 3 position
    data_.insert(data_.begin(), mol.size(), std::vector<float>(FEATURE_SIZE, 0.0));
    for (unsigned int i = 0; i < mol.size(); i++) {
        ChemAtom atom = mol[i];
        Vector3 point = atom.position();
        float radius = atom.getRadius();
        MOL2_TYPE mol2type = atom.getMol2Type();
        float charge = atom.getCharge();
        float asa = atom.getASA();


        if (mol2type == 3 || mol2type == 12 || mol2type == 14) {
            std::cout << "Mol2 type not relevent for this task: " << mol2type << std::endl;
        } else {
            data_[i][0] = point[0];
            data_[i][1] = point[1];
            data_[i][2] = point[2];
            data_[i][mol2type + 3] = 1.0;
            data_[i][6] = charge; //3+3
            data_[i][15] = asa;// 12+3
            data_[i][17] = radius; //14+3
        }
    }

}


void InterfacePointCloud::outputH5grid(const std::string path, const std::string dataset_name) {
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

    } catch (Exception &err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
    }

}
