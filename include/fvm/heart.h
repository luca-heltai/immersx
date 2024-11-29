#include <math.h>

#include <fstream>
#include <vector>

#include "valve.h"

using namespace std;
class heartMynard
{
public:
  /*
    Class for lumped model of the heart
    proposed by Mynard & Smolich (2015) and references therein

    Chamber order is:
    0 -> right atrium
    1 -> right ventricle
    2 -> left atrium
    3 -> left ventricle

    NB:
    - UNITS: cm/s/g
   */

  // time
  double time;
  double dt;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;
  double timeStop;
  double tIni;

  int iT;
  // number of chambers
  int nChambers;
  // cardiac cycle duration
  double T0;
  int    verbose;
  // "state" variables
  vector<double> g1; // contraction function
  vector<double> g2; // relaxation function
  vector<double> h;
  vector<double> hInt;
  vector<double> k;                  // scaling factor
  vector<double> efw;                // free-wall elastance [dyn/cm2/cm3]
  vector<double> eSep;               // septal elastance [dyn/cm2/cm3]
  vector<double> qIn;                // chamber inflow
  vector<double> qOut;               // chamber outflow
  vector<double> q;                  // chamber flow (dvdt)
  vector<double> eNat;               // native elastance [dyn/cm2/cm3]
  vector<double> v;                  // volume   [cm3]
  vector<double> rs;                 // source resistance
  double         vpc;                // pericardial volume [cm3]
  double         pIntraThoracic;     // intrathoracic pressure [dyn/cm2]
  double         ppc;                // pericardial pressure [dyn/cm2]
  vector<double> p;                  // pressure [dyn/cm2]
  vector<double> pOld;               // pressure [dyn/cm2]
  double         qCaval;             // flow from cava veins (SVC+IVC)
  double         pAorticRoot;        // pressure at aortic root
  double         pPulmonaryArterial; // pressure at pulmonary arteries
  double         qPulmonaryVenous;   // flow coming from pulmonary veins
  // parameters
  vector<double> tOnset;            //
  vector<double> tau1;              // contraction time offset [s]
  vector<double> tau2;              // relaxation time offset [s]
  vector<double> tOnsetref;         //
  vector<double> tOnsetref_initial; //
  vector<double> tau1ref;           // contraction time offset [s]
  vector<double> tau2ref;           // relaxation time offset [s]
  vector<double> tau1ref_initial;   // contraction time offset [s]
  vector<double> tau2ref_initial;   // relaxation time offset [s]
  vector<double> m1;                // contraction rate constant
  vector<double> m2;                // relaxation rate constant
  vector<double> m1_initial;        // contraction rate constant
  vector<double> m2_initial;        // relaxation rate constant
  vector<double> efwMin;            // min. free-wall elastance [dyn/cm2/cm3]
  vector<double> efwMin_initial;    // max. free-wall elastance [dyn/cm2/cm3]
  vector<double> efwMax;            // max. free-wall elastance [dyn/cm2/cm3]
  vector<double> efwMax_initial;    // max. free-wall elastance [dyn/cm2/cm3]
  vector<double> kL;                // septal elastance constant
  vector<double> kR;                // septal elastance constant

  vector<double> muav;         // atrioventricular plane piston constant
  vector<double> muav_initial; // atrioventricular plane piston constant
  vector<double> v0;           // volume at p=0   [cm3]
  vector<double> ks;           // source resistance constant   [cs/cm3]
  double         kpc;          // (0.5 mmHg)
  double         v0pc;         // (641 cm3)
  double         phipc;        // (40)
  double         vpcf;         // (30 cm3) pericardial fluid volume [cm3]
  double         vmio;         // (192 cm3)
  vector<int>    iCL;          // controlateral chamber index
  vector<int>    ventricleIdx;
  double         rho;
  // valves

  valveMynard tricuspidValve;
  valveMynard pulmonaryValve;
  valveMynard mitralValve;
  valveMynard aorticValve;

  int            nValves; // number of valves
  vector<double> valveMstSpec;
  vector<double> valveMrgSpec;
  vector<double> valveKvoSpec;
  vector<double> valveKvcSpec;
  vector<double> valveDPopenSpec;
  vector<double> valveDPcloseSpec;
  vector<double> valveLeffSpec;
  vector<double> valveAannSpec;
  vector<int>    valveTypeSpec;
  vector<int>    valveidxLeftSpec;
  vector<int>    valveidxRightSpec;


  // blood source
  double volSource;      // [ml] blood volume to be added to left ventricel
  double volSourceAdded; //
  double timeSource;     // time along with to add volSource
  double qSource;
  // output
  ofstream sample;

  string name;
  // methods
  void
  init(string ifile, string outDir, string testcase);
  void
  readHeartParameters(string _file);
  void
  printHeartParameters();
  void
  end();
  void
  setTimeConstants();
  void
  getg1(double t);
  void
  getg2(double t);
  void
  geth(double t);
  void
  getk();
  void
  getefw();
  void
  geteSep();
  void
  getq();
  void
  geteNat();
  void
  getv();
  void
  getrs();
  void
  getvpc();
  void
  getppc();
  void
  getPressure();
  void
  updateState();
  // state
  void
  saveState();
  void
  readState();
  void
  getState(vector<double> &state);
  void
  setState(vector<double> &state);
  void
         output();
  int    stateSize;
  string outDir;
  string stateDir;
};

class leftVentricleModel
{
public:
  /*
    Class for lumped model of the left ventricle
    using the time-varying elastance function proposed in
    Segers & Stergiopulos, 1996: Determinants of stroke
    volume and systolic and diastolic aortic pressure

    NB:
    - UNITS: cm/s/g
   */

  // time
  double time;
  double dt;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;
  double tIni;

  int iT;
  int elastanceType;
  // number of chambers
  int nChambers;
  // cardiac cycle duration
  double T0;
  int    verbose;
  // parameters
  double tp;
  double eMin;
  double eMax;
  double alpha;
  double alpha1;
  double alpha2;
  double n1;
  double n2;
  double pAtrium; // pressure at "virtual atrium"
  double v0;      // ventricle dead volume

  double tp_initial;
  double T0_initial;
  double eMin_initial;
  double eMax_initial;
  double alpha_initial;
  double alpha1_initial;
  double alpha2_initial;
  double n1_initial;
  double n2_initial;
  double pAtrium_initial; // pressure at "virtual atrium"
  double v0_initial;      // ventricle dead volume

  double rho;


  // "state" variables
  double e;
  double v;
  double p;    // pressure [dyn/cm2]
  double pOld; // pressure [dyn/cm2]
  double pAorta;
  // valves

  valveMynard mitralValve;
  valveMynard aorticValve;

  int            nValves; // number of valves
  vector<double> valveMstSpec;
  vector<double> valveMrgSpec;
  vector<double> valveKvoSpec;
  vector<double> valveKvcSpec;
  vector<double> valveDPopenSpec;
  vector<double> valveDPcloseSpec;
  vector<double> valveLeffSpec;
  vector<double> valveAannSpec;
  vector<int>    valveTypeSpec;
  vector<int>    valveidxLeftSpec;
  vector<int>    valveidxRightSpec;


  // output
  ofstream sample;

  string name;
  // methods
  void
  init(string ifile, string outDir, string testcase);
  void
  readLeftVentricleParameters(string _file);
  void
  printLeftVentricleParameters();
  void
  gete(double t);
  void
  setAlpha();
  double
  geteAdim(double t);
  void
  getPressure();
  void
  updateState();
  // state
  void
  saveState();
  void
  readState();
  void
  getState(vector<double> &state);
  void
  setState(vector<double> &state);
  void
         output();
  int    stateSize;
  string outDir;
  string stateDir;
};
