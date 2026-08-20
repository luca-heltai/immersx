#include <fstream>
#include <vector>
#include <cmath>

/*
#undef log 
#undef exp
#define log(x) __builtin_log(x)
#define exp(x) __builtin_exp(x)
*/

using namespace std;
class pulmonarySun
{
 public:
  /*
    Class for lumped model of the pulmonary circulation
    proposed by Sun et al (1997)

    Compartment order is:
    0 -> arterial
    1 -> capillary
    2 -> venous

    NB:
    - capillary wedge pressure circuit ommited
    - UNITS: cm/s/g
   */
  int iT; // iterations 
  // "state" variables
  double p[3];      // pressure [dyn/cm2]
  double v[3];      // volume [cm3]
  double dv[3];      // dvdt [cm3/s]
  double q[3];      // flow [cm3/s]
  double dq[3];      // dqdt [cm3/s^2]
  double pIntraThoracic;
  int verbose;
  // boundary conditions
  double qPulmonaryValve; // flow from pulmonary valve
  double pLA;       // pressure at LA
  // parameters
  double E0[3];     // ref. elastance [(dyn/cm2)/(cm3)]
  double L[3];     // inertance  [(dyn/cm2)/(cm3/s^2)]
  double R[3];     // resistance [(dyn/cm2)/(cm3/s)]
  double VE[3];     // visco-elastance [(dyn/cm2)/(cm3/s)]
  double PHI[3];   // volume constant [cm3]

  // time
  double tIni;
  double time;
  double dt;

  // output
  ofstream samplePul;
  string namePul;
  double tSampleIni,tSampleEnd,dtSample;

  // methods
  void init(string ifile, string outDir, string testcase);
  void readPulmonaryParameters(string _file);
  void printPulmonaryParameters();
  void end();
  void getPressure();
  void getVolumeFromPressure();
  void updateState();

  // state
  void saveState();
  void readState();
  void getState(vector<double>& state);
  void setState(vector<double>& state);
  int stateSize;
  string outDir;
  string stateDir;
}; 
