//
// Created by User on 1/8/2022.
//

#ifndef _GRAPHINTERFACE_H_
#define _GRAPHINTERFACE_H_

#include "ChemMolecule.h"
#include <vector>
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"
#include <boost/range/combine.hpp>
#include <algorithm>
/**
 * Creates a graph representation for an area surrounding some atom.
 * ChemMolecule is a map of a few atoms. These atoms already have all the information imbedded in them
 * (mol2type, charge and asa). We simply go over every pair of atoms and ask if they are at a distance of "radius"
 * if so we say there is an edge between them. an edge coming out of atom i is represented by i existing in edge1 and in the same index
 * a node in edge2
 * The embedding for the ith node is a length of 16 vector of floats that exists in dataRepresentation
 */
class GraphInterface {

 public:
    GraphInterface();

    GraphInterface(const ChemMolecule &mol, float radius, int label, float distClosestMG, std::string& filename);

  /***
   * Output an h5 file containing edges (edge1 contains one side of the edge and edge2 contains the node on the other side)
   * While repr contains the vector embedding of the atom.
   * @param path
   * @param dataset_edge1_name
   * @param dataset_edge2_name
   * @param dataset_datarepr_name
   */
  void outputH5grid(const std::string path, const std::string dataset_edge1_name, const std::string dataset_edge2_name, const std::string dataset_datarepr_name,
                    const std::string dataset_label, const std::string closest_mg_label) {
    using namespace HighFive;
    //example taken from https://github.com/BlueBrain/HighFive/blob/master/src/examples/create_dataset_double.cpp
    try {
        for(auto row: dataRepresentation){
            if(std::any_of(row.begin(), row.end(), [](double d) { return std::isnan(d); })){
                throw Exception("nan exists in file in data representation" + path);
            }
        }
        for(auto row: distMatrix){
            if(std::any_of(row.begin(), row.end(), [](double d) { return std::isnan(d); })){
                throw Exception("nan exists in file in distance matrix" + path);
            }
        }
        if(std::any_of(edge1.begin(), edge1.end(), [](double d) { return std::isnan(d); })){
            throw Exception("nan exists in file in data representation" + path);
        }
        if(std::any_of(edge2.begin(), edge2.end(), [](double d) { return std::isnan(d); })){
            throw Exception("nan exists in file in data representation" + path);
        }

        // Create a new file using the default property lists.
        File file(path, File::ReadWrite | File::Create | File::Truncate);


        // Create the three attributes
        DataSet dataset  =
                file.createDataSet(dataset_datarepr_name, DataSpace(1), AtomicType<int>());
        // Make attributes and write
        Attribute repr = dataset.createAttribute<float>(
                dataset_datarepr_name, DataSpace::From(dataRepresentation));
        repr.write(dataRepresentation);


        Attribute edges1 = dataset.createAttribute<int>(
                dataset_edge1_name, DataSpace::From(edge1));
        edges1.write(edge1);

        Attribute edges2 = dataset.createAttribute<int>(
                dataset_edge2_name, DataSpace::From(edge2));
        edges2.write(edge2);

        Attribute closest_mg_dist = dataset.createAttribute<float>(
                closest_mg_label, DataSpace::From(distClosestMG)
                );
        closest_mg_dist.write(distClosestMG);

        Attribute the_label = dataset.createAttribute<int>(
                dataset_label, DataSpace::From(this->label));
        the_label.write(this->label);

        // Attributes can only save very small datasets, the distance matrix is larger so we'll save it in its
        // own dataset
        DataSet distances = file.createDataSet<float>("distances", DataSpace::From(distMatrix));
        distances.write(this->distMatrix);

        DataSet coords = file.createDataSet<float>("coordinates", DataSpace::From(coordinates));
        coords.write(this->coordinates);



      } catch (Exception& err) {
        // catch and print any HDF5 error
//        throw err;
        std::cerr << err.what() << std::endl;
      }

  }
  static void outputH5Vector(const std::vector<GraphInterface> &vec, const std::string& path, const std::string& dataset_edge1_name, const std::string& dataset_edge2_name, const std::vector<std::string>& group_names,
                             const std::string& dataset_label, const std::string& closest_mg_label)
  {
      using namespace HighFive;
      //example taken from https://github.com/BlueBrain/HighFive/blob/master/src/examples/create_dataset_double.cpp
      try {

          // Create a new file using the default property lists.
          File file(path, File::ReadWrite | File::Create | File::Truncate);
          for(auto  tup: boost::combine(vec, group_names)){
              GraphInterface graph;
              std::string group_name;
              boost::tie(graph, group_name) = tup;
              Group group = file.createGroup(group_name);

              // Create the three attributes
              DataSet dataset  =
                      group.createDataSet("attributes", DataSpace(1), AtomicType<int>());
              // Make attributes and write
              Attribute repr = dataset.createAttribute<float>(
                      "repr", DataSpace::From(graph.getDataRepresentation()));
              repr.write(graph.getDataRepresentation());


              Attribute edges1 = dataset.createAttribute<int>(
                      dataset_edge1_name, DataSpace::From(graph.getEdge1()));
              edges1.write(graph.getEdge1());

              Attribute edges2 = dataset.createAttribute<int>(
                      dataset_edge2_name, DataSpace::From(graph.getEdge2()));
              edges2.write(graph.getEdge2());

              Attribute closest_mg_dist = dataset.createAttribute<float>(
                      closest_mg_label, DataSpace::From(graph.getDistClosestMg())
                      );
              closest_mg_dist.write(graph.getDistClosestMg());

              Attribute the_label = dataset.createAttribute<int>(
                      dataset_label, DataSpace::From(graph.getLabel()));
              the_label.write(graph.getLabel());

              // Attributes can only save very small datasets, the distance matrix is larger so we'll save it in its
              // own dataset
              DataSet distances = group.createDataSet<float>("distances", DataSpace::From(graph.getDistMatrix()));
              distances.write(graph.getDistMatrix());

              DataSet coords = group.createDataSet<float>("coordinates", DataSpace::From(graph.getCoordinates()));
              coords.write(graph.getCoordinates());

          }



      } catch (Exception& err) {
          // catch and print any HDF5 error
          //        throw err;
          std::cerr << err.what() << std::endl;
      }
  }

 private:
  std::vector<std::vector<float>> dataRepresentation;
public:
    const std::vector<std::vector<float>> &getDataRepresentation() const;

    const std::vector<int> &getEdge1() const;

    const std::vector<int> &getEdge2() const;

    const std::vector<std::vector<float>> &getDistMatrix() const;

    const std::vector<std::vector<float>> &getCoordinates() const;

    int getLabel() const;

    float getDistClosestMg() const;

private:
    std::vector<int> edge1;
  std::vector<int> edge2;
  std::vector<std::vector<float>> distMatrix;
  std::vector<std::vector<float>> coordinates;
  int label;
  float distClosestMG;

  void createAllRepresentations();
  std::vector<float> createRepresentation(const ChemAtom &atom, const bool mg_or_water, std::string& filename);


};

#endif //_GRAPHINTERFACE_H_
