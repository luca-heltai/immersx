#include <fstream>
#include <vector>
using namespace std;
class starling
{
public:
  /*
    Starling model presented in Mynard & Smolich (2015) and therein cited
    references NB:
    - UNITS: cm/s/g
   */

  // 'state' variables
  int    ID;
  double zeta;
  double dzeta;
  double dpp, dpr;
  double dp;
  double dq;
  double q; // flow across starling resitor [cm3/s]
  //  double r;
  double pDown;
  double kop, kcp, kor, kcr;
  double pUpstream;   // upstream pressure [dyn/cm2]
  double pDownstream; // downstream pressure [dyn/cm2]
  double pExt;        // external pressure [dyn/cm2]
  // parameters
  int idxLeft, idxRight;     // left vessel index, right vessel index
  int iCellLeft, iCellRight; // left vessel cell index, right vessel cell index
  int csfFlag;
  double signLeft, signRight; // account for orientation of 1D vessels
  double rOpen;               // resistance for open state
  double rClosed;             // resistance for closed state
  double pZeroFlow;
  double ptm0;
  double leff;
  double aeff;
  double aeffmin;
  double aeffmax;
  double rho;
  double l;
  double b;
  // output
  int      starlingID;
  ofstream sample;
  string   name;
  // time
  double time;
  double dt;
  double dtSample;
  double tSampleIni;
  double tSampleEnd;
  int    iT;
  // output
  // methods
  double
  getStateDiff(double zetaLoc);
  double
  getFlowDiff(double qLoc);
  void
  updateAux(double zetaLoc);
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
