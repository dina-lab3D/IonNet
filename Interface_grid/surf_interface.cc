#include <fstream.h>
#include <string>
#include <map>
#include "PDB.h"
#include "Molecule.h"
#include "Atom.h"
#include "MoleculeGrid.h"
#include "Surface.h"


int main(int argc, char **argv)
{
  if(argc!=5) {
    cout << "Program Usage: " << endl;
    cout << "surf_interface [lig_pdb_filename] [lig_surf_filename] [rec_surf_filename] [thr]" << endl;
    return 0;
  }
        
  float thr=atof(argv[4]);

  Molecule<Particle> mol;
  ifstream molFile(argv[1]);
  mol.readPDBfile(molFile);
  molFile.close();

  Surface ligSurface;
  ifstream ligSurfFile(argv[2]);
  ligSurface.readShouFile(ligSurfFile);
  ligSurfFile.close();

  Surface recSurface;
  ifstream recSurfFile(argv[3]);
  recSurface.readShouFile(recSurfFile);
  recSurfFile.close();
  
  MoleculeGrid* grid = new MoleculeGrid(ligSurface, 0.5, thr) ;
  grid->computeDistFromSurface(ligSurface);
  grid->markTheInside(mol);

  vector<int> interface(recSurface.size(), 0);
  for(int i=0; i<(int)recSurface.size(); i++) {
    if(grid->getDist(recSurface(i)) < thr)
      interface[i]++;
  }
  
  for(int i=0; i<(int)recSurface.size(); i++)
    if(interface[i]>0)
      cout << recSurface[i];
  cout << endl;
  return 0;
}
