//
// Created by User on 1/8/2022.
//

#include "GraphInterface.h"
#include <stdexcept>

void GraphInterface::createAllRepresentations ()
{

}

std::vector<float> GraphInterface::createRepresentation (const ChemAtom &atom, const bool mg_or_water, std::string& filename)
{
    const int REPRESENTATION_SIZE = 16;
    std::vector<float> retVec(REPRESENTATION_SIZE);
    MOL2_TYPE mol2type = atom.getMol2Type();
    if (mg_or_water && (mol2type == UNK_MOL2_TYPE))
    {
        // being in here means it is not the first atom in the graph so the mol2type should always be something that is
        // not unknown, otherwise we have a huge issue with the parsing of the PDB
        std::cerr << "non-mg/water atom got given unknown label! watch out\n";
        std::ofstream corrupt_file("/cs/labs/dina/punims/MGClassifierV2/corrupt_file.txt");
        corrupt_file << filename;
        corrupt_file.close();
        throw std::invalid_argument("file " + filename + "was corrupt" );
    }
    float charge = atom.getCharge();
    float asa = atom.getASA();
    retVec[mol2type] = 1.0;
    //the first index (0) is always the mg or water. So we receive the index, if it is 0 we set the charge and asa
    // to 0 otherwise we set it to charge
    retVec[3]= mg_or_water ? charge : 0;
    retVec[15] = mg_or_water ? asa : 0;

    return retVec;
}


GraphInterface::GraphInterface (const ChemMolecule &mol, float radius, int label, float distClosestMG, std::string& filename): edge1(std::vector<int>()), edge2(std::vector<int>()),
dataRepresentation(std::vector<std::vector<float>>()), distMatrix(std::vector<std::vector<float>>(mol.size(), std::vector<float>(mol.size(), 0))),
label(label), distClosestMG(distClosestMG)
{

  for(int i=0; i<mol.size(); i++)
    {
      dataRepresentation.push_back (createRepresentation (mol[i], i, filename));
      auto coords = mol[i].position();
      coordinates.push_back(std::vector<float>({coords[0], coords[1], coords[2]}));
      for(int j=i; j<mol.size (); j++){
          float dist = (mol[i] - mol[j]).norm ();
          if(dist <= radius)
          {
            edge1.push_back (i);
            edge2.push_back (j);
            distMatrix[i][j] = dist; //note that we don't need the distance if there is no edge because the dataset won't use that distance
            // if we need the distance later for computing translations then we have coordinates.
          }
      }

    }
}

const std::vector<std::vector<float>> &GraphInterface::getDataRepresentation() const
{
    return dataRepresentation;
}

const std::vector<int> &GraphInterface::getEdge1() const
{
    return edge1;
}

const std::vector<int> &GraphInterface::getEdge2() const
{
    return edge2;
}

const std::vector<std::vector<float>> &GraphInterface::getDistMatrix() const
{
    return distMatrix;
}

const std::vector<std::vector<float>> &GraphInterface::getCoordinates() const
{
    return coordinates;
}

int GraphInterface::getLabel() const
{
    return label;
}

float GraphInterface::getDistClosestMg() const
{
    return distClosestMG;
}

GraphInterface::GraphInterface() {}
//Just for reference
//enum MOL2_TYPE {C2 = 0, C3 =1 , Car = 2, charge = 3, N2 = 4, N4 = 5, Nam = 6,
//        Nar = 7, Npl3 = 8, O2 = 9, O3 = 10, Oco2 = 11, P3 = 12, S3 = 13, UNK_MOL2_TYPE = 14 , asa = 15};
