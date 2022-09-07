#include <ChemMolecule.h>
#include <Atom.h>
#include <Common.h>
#include <connolly_surface.h>
#include "PDB.h"

// Program inputs: pdbfile, output absolute path

class RNASelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        std::string type = PDB::atomType(PDBrec);
        return (PDB::isATOMrec(PDBrec) && PDB::NucleicSelector()(PDBrec) && PDB::WaterHydrogenUnSelector()(PDBrec));
    }
};

class MGSelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        std::string type = PDB::atomType(PDBrec);
        return type[0] == 'M' && type[1] == 'G' && type[3] == ' ';
    }
};

int main(int argc, char **argv) {

    // read lib with atomic radii
    ChemLib clib("chem.lib");

    // read molecule
    ChemMolecule mol;
    ChemMolecule mgMol;
    //Common::readChemMolecule(argv[1], mol, clib);
    Common::readChemMolecule(argv[1], mol, RNASelector(), clib);
    Common::readChemMolecule(argv[1], mgMol, MGSelector(), clib);
    // generate surface
    float probe_radius = 1.4;
    Surface surface = get_connolly_surface(mol, 0.5, probe_radius);
   
    // go over the surface and output probe centers
    Molecule<Atom> probes;
    for(unsigned int i=0; i<surface.size(); i++) {
        if(surface[i].surfaceType() == SurfacePoint::Pits ||
           surface[i].surfaceType() == SurfacePoint::Belts) continue;
        int atom_index = surface[i].atomIndex(0);
        Vector3 atom_center = mol(atom_index);
        float atom_radius = mol[atom_index].getRadius();
        Vector3 normal = surface[i].normal();
        Vector3 probe_center = atom_center + normal * (atom_radius + probe_radius);
        Atom atom(probe_center, 'C', (i+1) % 10000, (i+1) % 10000, " PB ", 'X'); // modulo of 10,000 to make sure that the pdb format is correct.
        //atom.setAtomName("PB");
        probes.push_back(atom);
    }


    std::ofstream out(argv[2]);
    out << probes;

    std::ofstream RNAOut(argv[3]);
    RNAOut << mol;

    std::ofstream MGOut(argv[4]);
    MGOut << mgMol;

    
    return 0;
}









