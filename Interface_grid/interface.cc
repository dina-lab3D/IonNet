#include "MolecularInterface.h"
#include "InterfaceGrid.h"
#include "GraphInterface.h"

#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>
#include <random>
#include <algorithm>
#include <limits>

using namespace boost;
using namespace boost::lambda;
namespace po = boost::program_options;
namespace fs = boost::filesystem;




#define CHEM_LIB_IDX 1
#define MOL_IDX 2
#define CHAIN_IDX 3
#define LIGAND_IDX 4
#define RES_IDX 5
#define VOX_SIZE_IDX 6
#define X_IDX 7
#define Y_IDX 8
#define Z_IDX 9
#define OUT_FOLDER_IDX 10
#define DATASET_IDX 11
#define INTERFACE_OUTPUT 12
#define PDB_OUTPUT 13
#define NPY_OUTPUT 14
#define H5_OUTPUT 15



using std::ifstream;

class MGSelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        std::string type = PDB::atomType(PDBrec);
        return type[0] == 'M' && type[1] == 'G' && type[3] == ' ';
    }
};

class ProteinRNASelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        std::string type = PDB::atomType(PDBrec);
        return (PDB::isATOMrec(PDBrec) && PDB::WaterHydrogenUnSelector()(PDBrec));
    }
};

class RNASelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        std::string type = PDB::atomType(PDBrec);
        return (PDB::isATOMrec(PDBrec) && PDB::NucleicSelector()(PDBrec) && PDB::WaterHydrogenUnSelector()(PDBrec));
    }
};

//// Selector that picks  water atoms
class WaterSelector : public PDB::Selector
{
public:
    bool operator()(const char *PDBrec) const
    {
        return (((PDBrec[17] == 'H' && PDBrec[18] == 'O' && PDBrec[19] == 'H') ||
                 (PDBrec[17] == 'D' && PDBrec[18] == 'O' && PDBrec[19] == 'D')));
    }
};

class PBSelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        std::string type = PDB::atomType(PDBrec);
        return ((type[0] == 'P' && type[1] == 'B') || (type[1] == 'P' && type[2] == 'B'));
    }
};


PDB::Selector* getSelector(std::string selectorStr, std::map<std::string, PDB::Selector*>& selectorMap)
/**
 * Returns the correct selector according to the given string
 * @param selectorStr
 * @return
 */
{
    if(selectorStr == "MG")
    {
        return selectorMap["MG"];
    }
    else if(selectorStr == "H20")
    {
        return selectorMap["H20"];
    }
    else if(selectorStr == "PB")
    {
        return selectorMap["PB"];
    }

    //Default value if all fails.
    return selectorMap["MG"];

}

void moltype_filecheck(ChemLib& clib, const std::string& filename){
    /**
     * This function is given each file one by one just to make sure all RNA selected atoms are written correctly
     * and are given a type by mol2type.
     * addMol2Type throws an invalid argument exception and writes to a file.
     * quality check should catch this exception.
     */
    ChemMolecule allAtoms;
    ifstream molFile1(filename);
    if(!molFile1) {
        std::cerr << "Can't open file " << filename << std::endl;
        return;
    }
    allAtoms.loadMolecule(molFile1, clib, RNASelector());
    molFile1.close();
    try{
        allAtoms.addMol2Type();
    }
    catch (const std::invalid_argument& exception){
        std::cerr << filename << " was corrupt, writing to file\n";
        std::ofstream corrupt_file("/cs/labs/dina/punims/MGClassifierV2/corrupt_file.txt", std::ios_base::app);
        corrupt_file << filename << std::endl;
        corrupt_file.close();
    }
}

std::string getFileName(std::string filePath, bool withExtension = true, char seperator = '/')
{
    // Get last dot position
    std::size_t dotPos = filePath.rfind('.');
    std::size_t sepPos = filePath.rfind(seperator);

    if(sepPos != std::string::npos)
    {
        return filePath.substr(sepPos + 1, filePath.size() - (withExtension || dotPos != std::string::npos ? 1 : dotPos) );
    }
    return "";
}


/***
 * extractClouds is a function that receives a ChemMolecule of MG atoms in an RNA strand and returns spheres
 * surrounding these MG atoms in a radius of epsilon with the am MG atom as the center of each of these spheres.
 * @param mgIons - collection of MG atoms in an RNA strand
 * @param allAtoms - collection of all other atoms in the same RNA stand
 * @param epsilon - Radius to calculate the sphere surrounding the RNA
 * @param cs - cube size for the GeomHash. We take a geometric hash for the surroundings at first, then to get a sphere
 * we trim it down using euclidean distance
 * @return
 */
std::vector<ChemMolecule> extractClouds(const ChemMolecule& mgIons, const ChemMolecule& allAtoms, float epsilon,  float cs = 1.0)
{

    std::vector<ChemMolecule> mgClouds(mgIons.size());
    // d is the dimension of the hash, cs is the voxel size
    GeomHash <Vector3,int> gHash(3, cs);

    for(unsigned int i=0;i<allAtoms.size();i++) {
        // hashing for faster searches later on
        gHash.insert(allAtoms[i].position(),i);
    }
    for(unsigned int i=0; i<mgIons.size(); i++){
        HashResult<int> result;
        // from all atoms in the hash, take radius in the RNA and add it to result
        gHash.query(mgIons[i], epsilon, result);
        HashResult<int>::iterator x_end=result.end(), x = result.begin();
        // must also check distance because we're taking a cube and not an actual radius.
        // this for loop trims the GeomHash from cubes to a sphere.
        for (; x != x_end; x++){
            float d = mgIons[i].dist(allAtoms[*x]);
            if(d < epsilon)
                mgClouds[i].push_back(allAtoms[*x]);
        }
    }
    return mgClouds;


}

/***
 * Given an atom, if it is water (label 0) find closest MG atom that is in pureMgIons.
 * Otherwise return 0
 * @param pureMgIons
 * @param targetAtom
 * @param label
 * @return
 */
float findClosestMG(const ChemMolecule& pureMgIons, const ChemAtom& targetAtom, int label)
{
    if (!label)
    {
        float dist = std::numeric_limits<float>::max();
        for(const auto& atom: pureMgIons)
        {
            float cur_dist = atom.dist(targetAtom);
            if (cur_dist < dist)
            {
                dist = cur_dist;
            }
        }
        return dist;
    }
    else
    {
        return 0;
    }
}

bool is_file_exist(const std::string& fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

/**
 * Creates a new subdirectory for pipeline_to_interface output.
 * @param base_dir
 * @param i
 * @return new directory path
 */
std::string create_new_subdirectory(const std::string& base_dir, int i)
{
    std::string new_path = base_dir + "/" + "raw_" + std::to_string(i);
    if(!boost::filesystem::is_directory(new_path) || !boost::filesystem::exists(new_path)){
        boost::filesystem::create_directory(new_path);
    }
    return new_path;

}


/***
 * pipeline_to_interface is the main work horse of this program. It opens the file,
 * collects the relevant atoms from the RNA and then produces spheres of atoms surrounding MG atoms in that RNA.
 * Once that is done the function saves the data for each sphere in H5 files
 * @param clib - ChemLib from main
 * @param filename - RNA filename
 * @param radius - radius for extractClouds
 * @param voxelSize - how many Armstrong are contained in each voxel (voxel is basically a 3D pixel, we're asking for the length of each voxel in all 3 dimensions)
 * @param n - the total size of the grid cubed
 * @param out_folder - name of the output folder
 */
int pipeline_to_interface(ChemLib& clib, std::string filename, const float radius, const float voxelSize, const int n, std::string out_folder, PDB::Selector& selector, bool graph_representation, int num_of_mg, int ratio=5, bool probe=false){
    ChemMolecule mgIons, allAtoms, pureMgIons;
    //float origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
    ifstream molFile1(filename);
    if(!molFile1) {
        std::cerr << "Can't open file " << filename << std::endl;
        return 0;
    }
    mgIons.loadMolecule(molFile1, clib, selector);
    if (typeid(selector) == typeid(WaterSelector)){
        // in the case of taking waters we don't necessarily want all of them as there can be far too many samples
        // So we'll limit the amount to some multiple of the MG atoms by manipulating the iterator in mgIons.
        // we can shuffle the atoms and then choose a multiple of the MG atoms
        int seed = 1000;
        std::shuffle(mgIons.begin(), mgIons.end(), std::default_random_engine(seed));
        mgIons.resize(std::min(num_of_mg*ratio, (int)mgIons.size()));
    }
    molFile1.clear();
    molFile1.seekg(0, std::ios::beg);
    allAtoms.loadMolecule(molFile1, clib, RNASelector());
    molFile1.clear();
    molFile1.seekg(0, std::ios::beg);
    pureMgIons.loadMolecule(molFile1, clib, MGSelector());
    molFile1.close();
    mgIons.setMol2TypeToUnk(); //Make sure Mgions / Water molecules don't give away what type they are!
    allAtoms.addMol2Type();

    //TODO do we need to compute surface area for mgIons?
//    mgIons.computeASAperAtom();
    allAtoms.computeASAperAtom();


    // call analyze here with MG selector / Zinc selector. Consider cloud returned by analyze as the contacts mol
    std::vector<ChemMolecule> mgClouds = extractClouds(mgIons, allAtoms, radius);
    int counter = 0;
    int subdir_count = 0;
    std::string subdir_name = "";
    for(unsigned int i = 0; i < mgClouds.size(); i++)
    // for each mg radius we now create a grid to create our data vector.
    {
        if (mgClouds[i].size() == 0){
            continue;
        }
        if (counter%1000 == 0){
            subdir_name = create_new_subdirectory(out_folder, subdir_count);
            subdir_count++;
        }

        if (!graph_representation)
        {
            Vector3 center = mgIons[i];
            // move the contacts mol to grid center
            double factor = voxelSize/2.0;
            Vector3 gridCentroid = Vector3(n, n, n);
            gridCentroid *= factor;
            Vector3 translation = gridCentroid - center;
            mgClouds[i].rigidTrans(RigidTrans3(Vector3(0,0,0), translation));
            Vector3 origin(0.0, 0.0, 0.0);
            InterfaceGrid grid(mgClouds[i], origin, n, n, n, voxelSize);
            std::string folder(out_folder);
            const std::string name = "mgArea";
            const std::size_t dotIndex = filename.rfind("/");
            const std::string pdbName = filename.substr(dotIndex);
            const std::string gridFileNameHDF5 = folder + '/' + pdbName + "_" + std::to_string(counter) + "_i.h5";
            const std::string DATASET_NAME("dataset");
            grid.outputH5grid(gridFileNameHDF5, DATASET_NAME);
            counter++;
        }
        else
        {
            if(!probe)
            {
                int label = typeid(selector) == typeid(MGSelector) ? 1: 0;
                float distClosestMG = findClosestMG(pureMgIons, mgIons[i], label);
                mgClouds[i].insert(mgClouds[i].begin(), mgIons[i]); //unlike the 3d convolution representation here we must add the mg atom as part of the graph
                GraphInterface graph(mgClouds[i], radius, label, distClosestMG, filename);
                std::string folder(subdir_name);
                const std::size_t dotIndex = filename.rfind("/");
                const std::string pdbName = filename.substr(dotIndex);
                const std::string fileType = label ? "_MG": "_H2O";
                const std::string gridFileNameHDF5 = folder + pdbName + fileType + "_" + std::to_string(counter) + "_i.h5";
                const std::string RNAatomGridFileNameHDF5 = folder  + '/' + pdbName + "_RNAatoms.h5";
                graph.outputH5grid(gridFileNameHDF5, "edge1", "edge2", "repr", "label", "closest_mg");
                counter++;
            }
            else{
                int label = 0;
                float distClosestMG = findClosestMG(pureMgIons, mgIons[i], label);
                mgClouds[i].insert(mgClouds[i].begin(), mgIons[i]); //unlike the 3d convolution representation here we must add the mg atom as part of the graph
                GraphInterface graph(mgClouds[i], radius, label, distClosestMG, filename);
                std::string folder(subdir_name);
                const std::size_t dotIndex = filename.rfind("/");
                const std::string pdbName = filename.substr(dotIndex);
                const std::string fileType = "_PB";
                const std::string gridFileNameHDF5 = folder + pdbName + fileType + "_" + std::to_string(counter) + "_i.h5";
                const std::string RNAatomGridFileNameHDF5 = folder  + '/' + pdbName + "_RNAatoms.h5";
                graph.outputH5grid(gridFileNameHDF5, "edge1", "edge2", "repr", "label", "closest_mg");
                counter++;
            }
        }
    }
    return mgIons.size();

}

/***
 * Function that does the same as pipeline to interface but instead works specifically for graph representations on probes.
 * most importantly the function saves all probe graphs under one hierarchical file
 * @param clib
 * @param filename
 * @param radius
 * @param voxelSize
 * @param n
 * @param out_folder
 * @param selector
 * @param graph_representation
 * @param num_of_mg
 * @param ratio
 * @param probe
 * @return
 */
int graph_inference_pipeline(ChemLib& clib, std::string filename, const float radius, std::string out_folder, PDB::Selector& selector)
{
    ChemMolecule mgIons, allAtoms, pureMgIons;
    ifstream molFile1(filename);
    if(!molFile1) {
        std::cerr << "Can't open file " << filename << std::endl;
        return 0;
    }
    mgIons.loadMolecule(molFile1, clib, selector);
    molFile1.clear();
    molFile1.seekg(0, std::ios::beg);
    allAtoms.loadMolecule(molFile1, clib, RNASelector());
    molFile1.clear();
    molFile1.seekg(0, std::ios::beg);
    pureMgIons.loadMolecule(molFile1, clib, MGSelector());
    molFile1.close();
    mgIons.setMol2TypeToUnk(); //Make sure Mgions / Water molecules don't give away what type they are!
    allAtoms.addMol2Type();
    allAtoms.computeASAperAtom();


    // call analyze here with MG selector / Zinc selector. Consider cloud returned by analyze as the contacts mol
    std::vector<ChemMolecule> mgClouds = extractClouds(mgIons, allAtoms, radius);
    int counter = 0;
    int subdir_count = 0;
    std::vector<GraphInterface> probe_vec = std::vector<GraphInterface>();
    std::vector<std::string> probe_name = std::vector<std::string>();
    for(unsigned int i = 0; i < mgClouds.size(); i++){
        if (mgClouds[i].size() == 0){
            continue;
        }
        int label = 0;
        float distClosestMG = findClosestMG(pureMgIons, mgIons[i], label);
        mgClouds[i].insert(mgClouds[i].begin(), mgIons[i]); //unlike the 3d convolution representation here we must add the mg atom as part of the graph
        GraphInterface graph(mgClouds[i], radius, label, distClosestMG, filename);
        const std::string group_name = "PB_" + std::to_string(counter);
        probe_vec.push_back(graph);
        probe_name.push_back(group_name);
        counter++;
    }
    const std::string filepath = out_folder + "/" + "raw_probes.h5";
    GraphInterface::outputH5Vector(probe_vec, filepath, "edge1", "edge2", probe_name, "label", "closest_mg");
    return mgIons.size();
}




int main(int argc, char** argv)
{
    bool no_progress = false, no_h5 = false, graph_representation=false, overwrite=false, probe=false, quality_check=false;
    double radius = 8.0, cs = 1.0, vox_size = 1.0;
    int dimension = 3, max_clouds = 10, x = 32, y = 32, z = 32;
    std::string selector = "MG";
    std::string input_dir = ".", input_text_file, output_dir, meta_path, atom_str, chem_lib_path = "/cs/usr/punims/Desktop/punims-dinaLab/MGClassifier/Interface_grid/chem.lib";
    po::options_description desc("Allowed options");
    desc.add_options()
            ("input-dir,i", po::value<std::string>(&input_dir)->default_value(input_dir),
             "a path to the directory that contains pdb entries")
            ("selector, s", po::value<std::string>(&selector)->default_value(selector),
            "Choose selector which chooses which atoms have their surroundings examined. Default is MGselector; Options: MG, H20")
            ("output-dir,o", po::value<std::string>(&output_dir),
             "a path to the directory where the cuts will be stored (=input_dir/cuts)")
            ("meta,m", po::value<std::string>(&meta_path),
             "a path to the file that describes the output in form\n"
             "cloud_number (center_atom): num_of_cloud_atoms (=output_dir/meta.txt)")
            ("atom,a", po::value<std::string>(&atom_str),
             "the name of the center atom")
            ("help,h", "produce help message")
            ("cube-size,c", po::value<double>(&cs)->default_value(cs),"geohash resolution")
            ("dimension,d", po::value<int>(&dimension)->default_value(dimension),"geohash dimension")
            ("radius,r", po::value<double>(&radius)->default_value(radius),"radius of analysis in angstrom")
            ("max-clouds,M", po::value<int>(&max_clouds)->default_value(max_clouds),
             "restricts the maximum number of clouds per pdb entry")
            ("no-progress,P", po::bool_switch(&no_progress)->default_value(no_progress),
             "don't shows progress")
            ("chem-lib,l", po::value<std::string>(&chem_lib_path)->default_value(chem_lib_path),
             "a path to the 'chem.lib' file")
            ("voxel-size", po::value<double>(&vox_size)->default_value(vox_size))
            ("x,x", po::value<int>(&x)->default_value(x), "the number of Angstroms one voxel has in x direction")
            ("y,y", po::value<int>(&y)->default_value(y), "the number of Angstroms one voxel has in y direction")
            ("z,z", po::value<int>(&z)->default_value(z), "the number of Angstroms one voxel has in z direction")
            ("no-h5,H", po::bool_switch(&no_h5)->default_value(no_h5),
             "specifying this won't dump the output into .h5 and instead output into .pdb")
            ("graph_representation", po::bool_switch(&graph_representation)->default_value(graph_representation),
             "if true, the representation saved will be of a graph and not a 3d cube.")
             ("input-text-file,T", po::value<std::string>(&input_text_file)->default_value(input_text_file),
                     "a path to the text file containing pdb files to process")
             ("overwrite,OW", po::bool_switch(&overwrite)->default_value(overwrite),
                 "overwrite files that have been preprocessed")
             ("probe,PB", po::bool_switch(&probe)->default_value(probe),
                 "use probe selector for graph representation")
                 ("quality_check, QC", po::bool_switch(&quality_check)->default_value(quality_check),
                     "use to activate quality check function instead of creating features");


    po::positional_options_description p;
    p.add("input-dir", 1).add("atom", 1).add("dataset-name", 1);
    po::variables_map vm;
    po::command_line_parser parser{argc, argv};
    po::store(parser.options(desc).positional(p).run(), vm);
    po::notify(vm);


    if (vm.count("help")) {
        std::cout << "Usage: options_description [options]" << std::endl;
        std::cout << desc;
        return 0;
    }

    // here is the old main
    const char* cstr_chem_path = chem_lib_path.c_str();
    const std::string  mg_dir_name = "MG_dir";
    const std::string  graph_dir_name = "Graph_dir";
    ChemLib clib((cstr_chem_path));
    std::cerr << "chem lib done" << std::endl;
    namespace fs = filesystem;
    MGSelector mgSelector = MGSelector(); WaterSelector waterSelector = WaterSelector();
    PBSelector pbSelector = PBSelector();
    std::map<std::string, PDB::Selector*> selectorMap = std::map<std::string, PDB::Selector*>();
    selectorMap["MG"] = &mgSelector; selectorMap["H20"] = &waterSelector; selectorMap["PB"] = &pbSelector;
    PDB::Selector* cur_selector = getSelector(selector, selectorMap);

    std::vector<fs::path> dir_vec;
    if (!input_text_file.empty()){
        std::ifstream file(input_text_file);
        std::string str;
        while (std::getline(file, str)) {
            dir_vec.emplace_back(input_dir + "/" +str);
        }
    }
    else{
        dir_vec = std::vector<fs::path>(begin(fs::directory_iterator(input_dir)), end(fs::directory_iterator(input_dir)));
    }

    std::cout << "files that this batch is working on are\n:";
    for(auto& dirEntry : dir_vec){
        std::cout << dirEntry << '\n';
    }

    if (quality_check){
        std::cout << "running quality check\n";
        int i = 0;
        for(auto& dirEntry : dir_vec){
            std::cout << "checking " << dirEntry.native() << std::endl;
            moltype_filecheck(clib, dirEntry.native());

        }
        std::cout << "done running quality check\n";
        return 0;

    }

    if (!graph_representation)
    {
//        for(const auto& dirEntry : fs::directory_iterator(input_dir))
        for(auto& dirEntry : dir_vec)
        {
            // notice x,y,z should all be the same size
            if(dirEntry.extension().string() == ".pdb")
            {
                std::cout << "Working on file: " + dirEntry.string() + "\n\n";
                // the old representation requires two output folders, one for all positive samples and one for all negative samples
                int num_of_mg = pipeline_to_interface(clib,  dirEntry.native(), radius, vox_size, x, output_dir + "/positive", *getSelector("MG", selectorMap), graph_representation, 0);
                pipeline_to_interface(clib,  dirEntry.native(), radius, vox_size, x, output_dir + "/negative", *getSelector("H20", selectorMap), graph_representation, num_of_mg);

                // once complete, so that we can keep track of what files have been processed successfully I'll write down the file names in a text file
                std::string batch_number;
                if (!input_text_file.empty()){
                    size_t first = input_text_file.find_last_of('_')+1;
                    size_t last = input_text_file.find_last_of('.');
                    batch_number = input_text_file.substr(first, last - first);
                }
                else{
                    batch_number = "0";
                }
                std::ofstream completed_file;
                completed_file.open(output_dir + "/processed_files/processed_batch_"+batch_number + ".txt", std::fstream::in | std::fstream::out | std::fstream::app);
                completed_file << dirEntry.native() << '\n';
                completed_file.close();
            }
        }
    }
    else
    {
        // populate vector either with the text file or just from the directory.

        for(auto& dirEntry : dir_vec)
        {
            // notice x,y,z should all be the same size
            // TODO might want each sample in a directory of its own, if so then uncomment these next two lines
            std::string new_output = output_dir + "/" + dirEntry.filename().string();
            if(overwrite){//skip folder creation
                 }
            else if (probe){
                // don't create a folder for inference, keep the one file in the raw directory.
                new_output = output_dir;
            }
            else if(!boost::filesystem::is_directory(new_output) || !boost::filesystem::exists(new_output)){
                boost::filesystem::create_directory(new_output);
            }
            else{
                std::cerr << new_output << " is already a directory\n";
                continue;
            }
            if(dirEntry.extension().string() == ".pdb")
            {
                //check if file already exists in some capacity in the raw folder.
                const std::size_t dotIndex = dirEntry.native().rfind("/");
                const std::string pdbName = dirEntry.native().substr(dotIndex);
                const std::string fileType = typeid(*cur_selector) == typeid(MGSelector) ? "_MG": "_H2O";
                const std::string gridFileNameHDF5 = new_output + pdbName + fileType + "_" + std::to_string(0) + "_i.h5";

                std::cout << "Working on file: " + dirEntry.string() + "\n\n";
                if(is_file_exist(gridFileNameHDF5) && !overwrite){
                    std::cout << "File " << dirEntry.string() << " already processed\n";
                    continue;
                }
                if (!probe)
                {
                    int num_of_mg = pipeline_to_interface(clib,  dirEntry.native(), radius, vox_size, x, new_output, *getSelector("MG", selectorMap), graph_representation, 0);
                    pipeline_to_interface(clib,  dirEntry.native(), radius, vox_size, x, new_output, *getSelector("H20", selectorMap), graph_representation, num_of_mg);
                }
                else{
                    graph_inference_pipeline(clib,  dirEntry.native(), radius, new_output, *getSelector("PB", selectorMap));
                }
            }
        }
    }
}
