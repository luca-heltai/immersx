#include <fstream>
#include <vector>
using namespace std;
class valveMynard
{
public:
  /*
    Valve model presented in Mynard & Smolich (2015) and therein cited
    references NB:
    - UNITS: cm/s/g
   */

  // 'state' variables
  string name;
  double zeta, dzeta; // valve opening state [-]
  double q, dq;       // flow across the valve [cm3/s]
  double aeff;        // effective area [cm2]
  double b;           // resistance [dyn/cm2 /cm3 * s]
  double l;           // inertance
  double dp;          // pressure difference [dyn/cm2]
  double pL;
  double pR;
  // parameters
  int valveType;              // 0 if cardiac valve, 1 if between 2 1D segments
  int idxLeft, idxRight;      // left vessel index, right vessel index
  int iCellLeft, iCellRight;  // left vessel cell index, right vessel cell index
  double signLeft, signRight; // account for orientation of 1D vessels
  double aeffmin;             // min. effective area [cm2]
  double aeffmax;             // max. effective area [cm2]
  double aann;                // annulus area [cm2]
  double kvo;                 // valve opening rate coefficient
  double kvc;                 // valve closing rate coefficient
  double kvo_initial;
  double kvc_initial;
  double leff;    // effective length
  double rho;     // fluid density
  double mrg;     // 0 if healthy valve, 1 if inexistent valve
  double mst;     // 1 if healthy valve, 0 if atretic valve (fully closed)
  double dpopen;  // minimum pressure drop necessary for valve to open
  double dpclose; // minimum pressure drop necessary for valve to close

  // output
  ofstream sample;
  int      valveID;

  // time
  double time;
  double dt;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;

  int iT;
  // output
  // methods
  void
  getaeff(double qLoc);
  void
  getb();
  void
  getl();
  double
  getStateDiff(double zetaLoc);
  double
  getFlowDiff(double qLoc, double zetaLoc);
  void
  getDiffImp(double *x, double *dxdt);
  void
  getFuncImp(double *x, double *f);
  void
  getFuncIncImp(double *x, double *f, double *dx);
  void
  evolveState();
  void
  evolveFlow();
  void
  update();
  void
  output();
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
