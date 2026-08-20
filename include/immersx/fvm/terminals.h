#include <fstream>
#include <vector>
using namespace std;
class terminal
{
 public:

  /*
    Class for terminals of the type:
    .          qA                      qAl                       qV   
    1D artery1 -> rpA1 -> cA1 -> rVb1                -> rpVen1 -> 1D vein1
    1D artery1 -> rpA1 -> cA1 -> rVb1        - cVen  -> rpVen2 -> 1D vein2
    .               ...                                       ...
    1D arteryNa -> rpANa -> cANa -> rVbNa            -> rpVenNv -> 1D veinNv
    


    where:
    - rpA: proximal arterial resistance
    - cA: arterial/arteriolar compliance
    - rVb: vascular bed resistance
    - cVen: venous compliance
    - rpVen: equivalent of Ra for veins (characteristic impedance
    
    NB:
    - the terminal model allows for multiple feeding arteries and draining veins
    - the terminal time step can be fixed (independent from 1D time step)
    - the external pressure can vary in time (its time derivative must be provided)
    - UNITS: cm/s/g
   */

  // state variables
  vector <double> pAl;   // proximal pressure
  vector <double> pAlold;   // proximal pressure
  vector <double> pC;     // pressure after vascular bed
  vector <double> pCold;     // pressure after vascular bed
  vector <double> pCap;
  
  // derived variables
  vector < vector <double>  > qAl;    // flow in arterioles
  vector < vector <int>  >qAlIdx;    // indicates to which vein the arteriole contributes
  vector < vector < vector <int> > > qAlVenIdx; // indicates which arteriole contributes to which vein
  vector <int> qAlVenN;
  vector <int> qAlN;
  double qC;
  // parameters
  vector <double> rpA;
  vector <double> cA;
  vector <double> vA;
  double vol;
  double volArt;
  double volVen;
  double roundValVol;
  vector <double> vAOld;
  vector <double> cAtot;
  vector < vector <double>  > rVb;
  vector < vector <double>  > rVb0;
  vector < vector <double>  > cApVen; // peripheral compliance of arteries draining a vein
  vector <double> cVen;
  vector <double> vVen;
  vector <double> vVenOld;
  vector <double> rpVen;
  double pExt; // external pressure
  double pIM; // intramiocardial pressure, only valid for coronary terminals
  int intracranial;
  int coronary; // if 1, then the terminal is coronary
  int coronaryChamber; // if coronary==1, defines the cardiac chamber 
                       // to which the terminal belongs
  int idxCsf;
  double qOut;
  // boundary conditions
  int nA;         // number of arteries feeding the terminal 
  int nV;         // number of veins draining the terminal
  std::vector<int> arteryIdx; // indexes for arteries feeding the terminal 
  std::vector<int> veinIdx;   // indexes for veins draining the terminal
  vector <double> qA;     // flow at feeding arteries
  vector <double> qV;     // flow at draining veins
  vector <double> qIn;
  int ID;
  int listID;
  // time
  double time;
  double timeold;
  double dt;
  int plotRes;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;
  int iT;
  // output
  ofstream sample;
  void output();
  // state
  void saveState();
  void readState();
  void getState(vector<double>& state);
  void setState(vector<double>& state);
  int stateSize;
  string outDir;
  string stateDir;
  // methods
  void solveTimeStep();


}; 
