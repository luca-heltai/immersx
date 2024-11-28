#include <fstream>
#include <vector>
using namespace std;
class terminalCoronary
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
   - the external pressure can vary in time (its time derivative must be
   provided)
   - UNITS: cm/s/g
   */

  // variables comming from heart model
  double pHeart[4], vHeart[4], vHeart_0[4];
  double pericardiumPressure;

  // constants
  double alphaSIP, pCEPEpi, pCEPMid, pCEPEndo, alpha;

  // state variables
  vector<double> pAl;    // proximal pressure
  vector<double> pAlold; // proximal pressure
  vector<double> pC;     // pressure after vascular bed
  vector<double> pCold;  // pressure after vascular bed

  vector<double> vNewV;
  vector<double> vNewA;
  // derived variables
  vector<vector<double>> qAl; // flow in arterioles
  vector<vector<int>>
    qAlIdx; // indicates to which vein the arteriole contributes

  vector<double> volA;    // volume irrigated by each artery
  vector<int>    chamber; // cardiac chamber that the artery irrigates

  double mioDensity; // density of heart muscle

  //  vector < vector <double> >  volArt;  // volume irrigated by each
  //  "arteriole" must sum up to volA
  vector<vector<double>> volArt; //
  // epicardium variable
  vector<double> pCEPepi;
  vector<double> pSIPepi;
  vector<double> pIMepi;
  vector<double> pIMepiOld;

  vector<vector<double>> vNewepi1;
  vector<vector<double>> vNewmid1;
  vector<vector<double>> vNewendo1;
  vector<vector<double>> vNewepi2;
  vector<vector<double>> vNewmid2;
  vector<vector<double>> vNewendo2;

  vector<vector<double>> ptm1epi;    //
  vector<vector<double>> ptm1epiold; //
  vector<vector<double>> dptm1epidt; //
  vector<vector<double>> ptm2epi;    //
  vector<vector<double>> ptm2epiold; //
  vector<vector<double>> dptm2epidt; //
  vector<vector<double>> p1epi;      //
  vector<vector<double>> q1epi;      //
  vector<vector<double>> c1epi;      //
  vector<vector<double>> r1epi;      //
  vector<vector<double>> r1epi_0;    //
  vector<vector<double>> v1epi;      //
  vector<vector<double>> v1epi_0;    //
  vector<vector<double>> qmepi;      //
  vector<vector<double>> rmepi;      //
  vector<vector<double>> rmepi_0;    //
  vector<vector<double>> p2epi;      //
  vector<vector<double>> c2epi;      //
  vector<vector<double>> r2epi;      //
  vector<vector<double>> r2epi_0;    //
  vector<vector<double>> v2epi;      //
  vector<vector<double>> v2epi_0;    //
  vector<vector<double>> q2epi;      //
  // midwall variables
  vector<double>         pCEPmid;
  vector<double>         pSIPmid;
  vector<double>         pIMmid;
  vector<double>         pIMmidOld;
  vector<vector<double>> ptm1mid;    //
  vector<vector<double>> ptm1midold; //
  vector<vector<double>> dptm1middt; //
  vector<vector<double>> ptm2mid;    //
  vector<vector<double>> ptm2midold; //
  vector<vector<double>> dptm2middt; //
  vector<vector<double>> p1mid;      //
  vector<vector<double>> q1mid;      //
  vector<vector<double>> c1mid;      //
  vector<vector<double>> r1mid;      //
  vector<vector<double>> r1mid_0;    //
  vector<vector<double>> v1mid;      //
  vector<vector<double>> v1mid_0;    //
  vector<vector<double>> qmmid;      //
  vector<vector<double>> rmmid;      //
  vector<vector<double>> rmmid_0;    //
  vector<vector<double>> p2mid;      //
  vector<vector<double>> c2mid;      //
  vector<vector<double>> r2mid;      //
  vector<vector<double>> r2mid_0;    //
  vector<vector<double>> v2mid;      //
  vector<vector<double>> v2mid_0;    //
  vector<vector<double>> q2mid;      //
  // endocardium variables
  vector<double>         pCEPendo;
  vector<double>         pSIPendo;
  vector<double>         pIMendo;
  vector<double>         pIMendoOld;
  vector<vector<double>> ptm1endo;    //
  vector<vector<double>> ptm1endoold; //
  vector<vector<double>> dptm1endodt; //
  vector<vector<double>> ptm2endo;    //
  vector<vector<double>> ptm2endoold; //
  vector<vector<double>> dptm2endodt; //
  vector<vector<double>> p1endo;      //
  vector<vector<double>> q1endo;      //
  vector<vector<double>> c1endo;      //
  vector<vector<double>> r1endo;      //
  vector<vector<double>> r1endo_0;    //
  vector<vector<double>> v1endo;      //
  vector<vector<double>> v1endo_0;    //
  vector<vector<double>> qmendo;      //
  vector<vector<double>> rmendo;      //
  vector<vector<double>> rmendo_0;    //
  vector<vector<double>> p2endo;      //
  vector<vector<double>> c2endo;      //
  vector<vector<double>> r2endo;      //
  vector<vector<double>> r2endo_0;    //
  vector<vector<double>> v2endo;      //
  vector<vector<double>> v2endo_0;    //
  vector<vector<double>> q2endo;      //

  vector<vector<vector<int>>>
              qAlVenIdx; // indicates which arteriole contributes to which vein
  vector<int> qAlVenN;
  vector<int> qAlN;
  double      qC;
  // parameters
  vector<double>         rpA;
  vector<double>         cA;
  vector<double>         vA;
  double                 vol;
  double                 volArtTot;
  double                 volVenTot;
  double                 volCapTot;
  vector<double>         vAOld;
  vector<double>         cAtot;
  vector<vector<double>> rVb;
  vector<vector<double>> rVb0;
  vector<vector<double>>
                 cApVen; // peripheral compliance of arteries draining a vein
  vector<double> cVen;
  vector<double> vVen;
  vector<double> vVenOld;
  vector<double> rpVen;
  double         pExt; // external pressure

  int intracranial;
  int coronary;        // if 1, then the terminal is coronary
  int coronaryChamber; // if coronary==1, defines the cardiac chamber
                       // to which the terminal belongs
  int idxCsf;
  // boundary conditions
  int            nA; // number of arteries feeding the terminal
  int            nV; // number of veins draining the terminal
  vector<double> qA; // flow at feeding arteries
  vector<double> qV; // flow at draining veins
  int            ID;
  // time
  double time;
  double tIni;
  double timeold;
  double dt;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;

  int iT;
  // output
  ofstream sample;

  std::vector<int> arteryIdx; // indexes for arteries feeding the terminal
  std::vector<int> veinIdx;   // indexes for veins draining the terminal

  // methods
  void
  solveTimeStep();
  void
  setFlows();
  void
  output();
  void
  initOutput();
  void
  setIntraMuralPressure();
  // state
  void
  saveState();
  void
  readState();
  void
  getState(vector<double> &state);
  void
         setState(vector<double> &state);
  int    stateSize;
  string outDir;
  string stateDir;
};
