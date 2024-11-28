//	Author: 		Lucas O. Mueller
// 	Description:	This is the main file for generating vessel objects
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
// This is the vessel prototype
class vessel
{
public:
  vessel(){};
  ~vessel(){};

  double initial_node[4]; // first value is index of initial node and last 3 are
                          // the coordinates of the node
  double final_node[4]; // first value is index of final node and last 3 are the
                        // coordinates of the node

  int    isTerminal;
  int    terminalSide;
  double terminalSign;
  // ader scheme parameters
  int nDOF;
  int nDOFs;
  int otherPartitionID;
  /* int otherVessSide; */
  double  compliance;
  int     tlType;
  int     wasValve;
  int     wasStarling;
  int     wasTerminal;
  int     wasSteno;
  double  dtMinValve;
  double  dtMinSteno;
  double  dtMinStarling;
  double  dtMinTerminal;
  int     discretization;
  int     whichSteno;
  int     MPIVesselIndex;
  int     globalVessID;
  int     ID;
  int     iT;
  int     iTprev, iTprevPeriod;
  int     updsPrev, upds;
  int     NCELLS; // number of cells
  int     iSample;
  int     iSampleSpace;
  int     iSampleL;
  int     iSampleR;
  double  wSampleL;
  double  wSampleR;
  double  xSample;
  int     iJuncL, iJuncLorder;
  int     iJuncR, iJuncRorder;
  double  CFL;
  int     cflAdaptive;
  double  dx;     // mesh spacing [m]
  int     vT;     // vessel type (1: artery; 2: vein)
  double  xL;     // left coordinate of physical domain, usually xL=0
  double  xR;     // right coordinate of physical domain, usually xR=L
  double  thetaL; // angle for left bifurcation (Mynard & Valen-Sendstad)
  double  thetaR; // angle for right bifurcation  (Mynard & Valen-Sendstad)
  double  L;      // vessel length [m]
  double  rL;     // left radius [m]
  double  dR;     // rR-rL
  double  E;      // Young modulus
  double  h;      // vessel wall thickness
  double *xC;     // cell barycenter [m]
  double *a;      // cell area [m^2]
  double *au;     // cell flow rate [m^3/s]

  // ####################################################
  // Gravity related variables (General quantities like g or g_intensity are
  // declared in model1d.h)
  // ####################################################
  vector<double> g; // projection of gravity vector on vessels' axis. It has the
                    // size of the computational cells.
  vector<vector<double>>
    vessPositSupine; // vector containing the position of the vessels in the 3d
                     // space wrt the fixed reference system (supine position).
                     // Size: number of computational cells x 3.
  vector<vector<double>>
    vessVersorPositSupine; // versors containing the position of the vessels in
                           // the 3d space wrt the fixed reference system
                           // (supine position). Size: number of computational
                           // cells x 3.
  vector<vector<double>>
    vessVersorPositInit; // versors containing the initial position of the
                         // vessels in the 3d space as selected in input.
  double angle_x;        // angle (in rad) of rotation pitch
  double angle_y;        // angle (in rad) of rotation yaw
  double angle_z;        // angle (in rad) of rotation roll



  // auxiliar variables for semidiscrete discretization
  vector<double> highOrder[2];
  vector<double> aLSemiDiscrete;  // cell area [m^2]
  vector<double> auLSemiDiscrete; // cell flow rate [m^3/s]
  vector<double> aRSemiDiscrete;  // cell area [m^2]
  vector<double> auRSemiDiscrete; // cell flow rate [m^3/s]

  vector<double> a1SemiDiscrete;  // cell area [m^2]
  vector<double> au1SemiDiscrete; // cell flow rate [m^3/s]

  vector<double> a0SemiDiscrete;  // cell area [m^2]
  vector<double> au0SemiDiscrete; // cell flow rate [m^3/s]

  vector<double> a0CSemiDiscrete;
  vector<double> a0LSemiDiscrete;
  vector<double> a0RSemiDiscrete;


  vector<double> h0CSemiDiscrete;
  vector<double> h0LSemiDiscrete;
  vector<double> h0RSemiDiscrete;

  vector<double> eeCSemiDiscrete;
  vector<double> eeLSemiDiscrete;
  vector<double> eeRSemiDiscrete;

  vector<double> ecCSemiDiscrete;
  vector<double> ecLSemiDiscrete;
  vector<double> ecRSemiDiscrete;

  vector<double> peCSemiDiscrete;
  vector<double> peLSemiDiscrete;
  vector<double> peRSemiDiscrete;

  vector<double> a0diffCSemiDiscrete;
  vector<double> h0diffCSemiDiscrete;
  vector<double> eediffCSemiDiscrete;
  vector<double> ecdiffCSemiDiscrete;
  vector<double> pediffCSemiDiscrete;


  double **si; // space-time average of source term
  double   mean_au;
  double   mean_p;
  double   mean_a;
  double  *psi; // cell auxiliar variable for flow rate spatial gradient (see
                // Montecinos et al., 2014) [m^2/s]
  double *psib; // adimensional psi

  /* double *a0; // cross-sectional area at p=p0 [m^2] */
  /* double *aRef; // cross-sectional area at reference state [m^2] */
  /* double *pe; // external pressure [Pa] */
  /* double *h0; // vessel wall thickness [m] */
  /* double *ee; // Young modulus for elastin [Pa] */
  /* double *ec; // Young modulus for collagen [Pa] */
  /* double *ep0; */
  /* double *epr; */

  double *a0av;   // cross-sectional area at p=p0 [m^2]
  double *aRefav; // cross-sectional area at reference state [m^2]
  double *peav;   // external pressure [Pa]
  double *h0av;   // vessel wall thickness [m]
  double *eeav;   // Young modulus for elastin [Pa]
  double *ecav;   // Young modulus for collagen [Pa]
  double *ep0av;
  double *eprav;


  double **a0var;   // cross-sectional area at p=p0 [m^2]
  double **aRefvar; // cross-sectional area at p=p0 [m^2]
  double **pevar;   // external pressure [Pa]
  double **h0var;   // vessel wall thickness [m]
  double **eevar;   // Young modulus for elastin [Pa]
  double **ecvar;   // Young modulus for collagen [Pa]
  double **ep0var;
  double **eprvar;

  double *mu;         // fluid dynamic viscosity [Pa s]
  double *h0_initial; // (initial) vessel wall thickness [m] (for Kalman)
  double *ee_initial; // Young modulus for elastin [Pa]
  double *ec_initial; // Young modulus for collagen [Pa]
  double *dxi;        // mesh spacing [m]
  int    *pathID;     // 1 if segment; 0 if defined by integral curves
  double *ep0_initial;
  double *epr_initial;
  double  p0;
  double  alphaM;
  double  Gamma; // Viscoelastic coefficient (constant along vessel) Pa s m (See
                 // Alastruey et al., 2011)
  double Gamma_initial; // Viscoelastic coefficient (constant along vessel) Pa s
                        // m (See Alastruey et al., 2011)
  double GammaSplitting;
  double Gammas;
  double T;
  double Tb;

  double AreaIni; // area at mid point vessel, before solveTimeStep
  // Variables for adimensionalization
  double a0s;
  double c0s;
  double c02s;
  double k0s;
  double h0s;
  double Gamma0s;
  double muEq;
  // Adim variables
  double   *xCb;
  double    dxb;
  double   *ab;
  double   *aub;
  double   *a0b;
  double   *h0b;
  double   *eeb;
  double   *ecb;
  double    ees;
  double   *Kb;
  double   *peb;
  double   *rib;
  double   *celb;
  double    dtb;
  double    dt;
  double    time;
  double    timeSemiDiscrete;
  double   *gb;
  double   *mub;
  double   *al;
  double   *ql;
  double   *psil;
  double   *alsdg;
  double   *qlsdg;
  double   *psilsdg;
  double ***wFluc;
  double ***qhatLoc;
  double ***shatLoc;
  double ***alphahatLoc;
  double ***qhatLocNC;
  double ***qhatdiffLoc;
  double ***qhatdiffLocNC;
  double ***qhatdiffdiffLoc;
  double  **F0DGwLoc;
  double  **B;
  double  **K1qhat;
  double  **Fqnew;
  double  **JS;
  double  **JJ;
  double  **iJJ;
  double ***JJt;
  double ***iJJt;
  double  **deltab;
  double  **Fnew;
  double   *qLjunc;
  double   *qRjunc;
  double    qM[8];
  double    qP[8];
  double   *qLjuncLTS;
  double   *qRjuncLTS;
  double    fluctL[3];
  double    fluctR[3];
  double    tJuncL;
  double    tJuncR;
  double   *rADANvisco;
  double   *irADANvisco;
  double   *lADANvisco;
  double   *rADAN;
  double   *irADAN;
  double   *lADAN;

  double               **QDELTA;
  double               **MSdgDQ;
  double               **SdgDQ;
  double               **MSdgS;
  double               **FL;
  double               **FR;
  int                    iTmeanOld;
  int                    intracranial;
  int                    idxCsf;
  double                 vol, volOld;
  double                 timeOld;
  double                 p0b;
  double                 qL[3];     // value of left bcs
  double                 qR[3];     // value of right bcs
  int                    BCS[2];    // bcs type
  int                    bcType[2]; // bcs type
  vector<vector<double>> outflowBC;
  int                    type; // vessel type (1: artery; 0: vein)
  int vasTer;                  // vascular territory to which the vessel belons
  int vasTerType;              // 0 generic, 1 coronary
  int vasTerInd;     // 0 to nTerminals index of vascular territory to which the
                     // vessel belongs
  int     vasTerIdx; // vascular territory to which the vessel belongs
  int     vasTerCvnIdx;
  double *Q;
  // #######################################à
  //  for distributed source
  double Qdistributed;
  double Rdistributed;
  double Rdistributed_initial;
  double Rdistributed1;
  double Rdistributed2;
  double Cdistributed;
  double Pdistributed;
  int    doDistributedSource;
  int    calibrateDistributedResistances;
  double qOutVesselDistributed;
  // #######################################à
  double Rb; // peripherial resistance
  double pVb;
  double pV;
  double pC;
  double pCCurrent;
  double dpdtRCR;
  double pProx;
  double pRA;
  double rProx;
  double qProx;
  double qC;
  double qBC;

  double   Rbb;    // adimensional Rb
  double   Gammab; // Gamma/(L*sqrt(k0s*rho*a0s))
  int      recInfoReceived;
  int      updated;
  int      updatedCsf;
  double   pred;
  int      updateL;
  int      updateR;
  int      done;
  int      initial;
  int      mult, multCeil, multFloor;
  double   maxEig;
  int      artven[2];
  int      artvenN;
  int      clusterIdx;
  ofstream sampleMid;
  ofstream sampleSpace;
  ofstream sampleSpaceA;
  ofstream sampleSpaceQ;
  ofstream sampleSpaceP;
  ofstream sampleSpaceAref;
  ofstream sampleSpacePartition;
  ofstream sampleSpaceCeler;
  ofstream sampleSpaceEE;
  ofstream sampleSpaceEC;
  ofstream sampleSpaceKM;
  ofstream sampleSpaceWT;
  ofstream sampleConvergence;
  double   weights[20];
  int      counterWeights;
  // stenosis
  int    iSteno;    // 0: no steno; 1: steno (vessel)
  int   *iStenoLoc; // 0: no steno; 1: steno (cell)
  int    iPreSteno; // index of cell left to the stenosis
  double lV;        // stenosis length [m]


  // save results in vector
  vector<vector<double>> solutionTime;
  vector<vector<double>> solutionSpace;
  vector<double>         solutionSpaceX;
  vector<double>         solutionSpaceP;
  vector<double>         solutionSpaceD;

  vector<double> StateA;
  vector<double> StateP;
  vector<double> StateQ;
  vector<double> StateAold;
  vector<double> StatePold;
  vector<double> StateQold;
  double         L1A;
  double         L2A;
  double         LinfA;
  double         L1Q;
  double         L2Q;
  double         LinfQ;
  double         L1P;
  double         L2P;
  double         LinfP;

  // save the previous state
  double     *aub_n;
  double     *ab_n;
  vector<int> leftNeighs;
  vector<int> rightNeighs;
  double      juncA[2], juncQ[2];
  int        *sID;
  double     *XYZ;
  double     *c0;

  // visualization: we need the physical position of the vessel
  double x_3d_0[3]; // start
  double x_3d_1[3]; // end

  double cE;
  double epsilon;
  double epsilonb;
  double aOld[2];
  double psiL;
  double psiR;

  // LEFT AND RIGHT BOUNDARY VALUES  FOR ADER-DET-DG
  double polL[3];
  double polR[3];

  double  nV;
  double  eigMax;
  double  R0;
  double  Rinflow;
  double  pCinflow;
  double  R1;
  double  R1_initial;
  double  R2;
  double  R2_initial;
  double  RTOT;
  double  R3;
  double  R4;
  double  C1;
  double  C2;
  double  C3;
  double  C4;
  double  L1;
  double  L2;
  double  L3;
  double  L4;
  double  CC[4], RR[5], LL[4];
  double  C; // peripherial compliance
  double  C_initial;
  int     region;
  double  aLb;
  double  aRb;
  double  dAb;
  double  kLb;
  double  dKb;
  double  kRb;
  double  qRCR[4], pRCR[4], qRCRold[4], pRCRold[4];
  double *qLxt, *qRxt;
  double  qAvL[3], qAvR[3], qAvLL[3], qAvRR[3];
  double  qAvLx[3], qAvRx[3], qAvLxx[3], qAvRxx[3];

  // cbf
  double         qMean;
  double         qBase;
  double         G_aut;
  double         pBase;
  double         rAlBase;
  double         vABase;
  double         cAlBase;
  double         meanvA;
  double         Pmean;
  double         deltaC;
  double         xco2;
  double         xaut;
  double         Aco2;
  double         kC;
  vector<double> StateMeans;
  vector<double> StateMeansP;
  vector<double> StateMeansV;

  ofstream sampleCBF;


  // lymphatics

  // ###########################
  // lymphatic vessels (See Contarino & Toro 2018)
  // ######################
  int doLymphatics;

  // parameters
  double peLymph;
  double tauKvar;
  double tau1Kvar;
  double contractKvar;
  double relaxKvar;
  double mLymph;
  double nLymph;
  double kLymphaticMin;
  double kLymphaticMax;
  double leffLymphatics;
  double kvoLymphatics;
  double kvcLymphatics;
  double RILymph;
  double proximalPressure;
  double distalPressure;

  double a1Lymph;
  double a2Lymph;
  double a3Lymph;
  double b1Lymph;
  double b2Lymph;
  double lambdaCaLymph;
  double nCaLymph;
  double krelLymph;
  double kNOLymph;
  double nNOLymph;
  double tauNOLymph;
  double c1Lymph;
  double c2Lymph;
  double fminLymph;
  double texcitedLymph;
  double fCaLymph;

  // state variables
  double vLymph;
  double wLymph;
  double ILymph;
  double sLymph;

  double qLymph[4];
  double dqLymphdt[4];

  // auxiliar variables
  double fILymph;
  double fNOLymph;
  double fsLymph;
  double lambdaThetaLymph;
  double tauLymph;
  double kCa1Lymph;
  double kCa2Lymph;
  double ITildeLymph;

  void
  getLymphaticsAuxVariables(double *q);
  void
  getLymphaticsTimeDerivative(double *q, double *dqdt);

  // ###########################
  // ###########################
  // ###########################
  // ###########################

  // methods
  virtual double
  geta0(const int &iCell, const int &iDofs);
  virtual double
  getaRef(const int &iCell, const int &iDofs);
  virtual double
  geth0(const int &iCell, const int &iDofs);
  virtual double
  getee(const int &iCell, const int &iDofs);
  virtual double
  getec(const int &iCell, const int &iDofs);
  virtual double
  getep0(const int &iCell, const int &iDofs);
  virtual double
  getepr(const int &iCell, const int &iDofs);
  virtual double
  getpe(const int &iCell, const int &iDofs);

  virtual double
  geta0av(const int &iCell);
  virtual double
  getaRefav(const int &iCell);
  virtual double
  geth0av(const int &iCell);
  virtual double
  geteeav(const int &iCell);
  virtual double
  getecav(const int &iCell);
  virtual double
  getep0av(const int &iCell);
  virtual double
  geteprav(const int &iCell);
  virtual double
  getpeav(const int &iCell);

  virtual void
  seta0const(const int &iCell, const double &val);
  virtual void
  setaRefconst(const int &iCell, const double &val);
  virtual void
  seth0const(const int &iCell, const double &val);
  virtual void
  seteeconst(const int &iCell, const double &val);
  virtual void
  setecconst(const int &iCell, const double &val);
  virtual void
  setep0const(const int &iCell, const double &val);
  virtual void
  seteprconst(const int &iCell, const double &val);
  virtual void
  setpeconst(const int &iCell, const double &val);

  virtual void
  multa0const(const int &iCell, const double &val);
  virtual void
  multaRefconst(const int &iCell, const double &val);
  virtual void
  multh0const(const int &iCell, const double &val);
  virtual void
  multeeconst(const int &iCell, const double &val);
  virtual void
  multecconst(const int &iCell, const double &val);
  virtual void
  multep0const(const int &iCell, const double &val);
  virtual void
  multeprconst(const int &iCell, const double &val);
  virtual void
  multpeconst(const int &iCell, const double &val);

  virtual void
  setParamAtEdge(const int &iM, const int &iMvar, double *q);
};
