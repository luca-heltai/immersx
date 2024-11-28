#include <fstream>
#include <vector>
using namespace std;
class stenosisYoungTsai1973
{
 public:

  /*
    Stenosis model presented in Young & Tsai (1973)
    and implemented as described in Liang et al (2009):
    Multi-scale modeling of the human cardiovascular system 
    with applications to aortic valvular and arterial stenoses
    NB:
    - UNITS: cm/s/g
   */

  // 'state' variables
  string name;
  double q; // flow across the stenosis [cm3/s]
  double dq;
  // parameters
  double a0; // reference area [cm2]
  double as; // area of stenosis [cm2]
  double stenodeg; // as/a0
  double d0; // diameter of reference area [cm]
  double ds; // diameter of stenosis area [cm]
  double mu; // blood viscosity [Poise]
  double kv; // experimetal constant accounting for viscous term
  double kt; // experimental constant accounting for turbulence
  double ku_initial; // experimental constant accounting for inertial term
  double kt_initial; // experimental constant accounting for turbulence
  double kv_initial; // experimetal constant accounting for viscous term
  double kv_mult;
  double ku; // experimental constant accounting for inertial term
  double dp; // pressure difference across stenosis [dyn/cm2]
  double ls; // stenosis length [cm]
  int idxLeft,idxRight; // left vessel index, right vessel index
  int iCellLeft,iCellRight; // left vessel cell index, right vessel cell index
  double signLeft,signRight; // account for orientation of 1D vessels
  double rho; // fluid density
  // output
  ofstream sample;
  int stenosisID;
  // time
  double time;
  double dt;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;

  int iT;
  // output
  // methods
  double getFlowDiff(double qLoc);
  double getFlowDiffDer(double qLoc);
  double getDP(double qLoc, double dqLoc);
  void evolveFlow();
  void evolveFlowImplicit();
  double funcImplicit(double qNew, double qOld);
  double funcImplicitDer(double qNew);
  void update();
  void output();
  // state
  void saveState();
  void readState();
  void getState(vector<double>& state);
  void setState(vector<double>& state);
  string outDir;
  string stateDir;
}; 
