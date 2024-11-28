// include necessary packages
// #include <functional>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "omp.h"
#ifdef USE_MPI
#  include <metis.h>

#  include "mpi.h"
#endif
// #include "rp.h"
#include "gridgen.h"
#include "heart.h"
#include "inflow.h"
#include "linalg.h"
#include "observer.h"
#include "pulmonary.h"
#include "starling.h"
#include "stenosis.h"
#include "terminals.h"
#include "terminalsCoronary.h"
#include "vessel.h"

// #undef log
// #undef exp
// #define log(x) __builtin_log(x)
// #define exp(x) __builtin_exp(x)

#define BCTYPE_outflow_curve -66
#define BCTYPE_inflow_pressure -65
#define BCTYPE_inflow_pressure_curve -64



#ifndef __MODEL1D_H
#  define __MODEL1D_H 1

#  define sqrtPI 1.772453850905520



#  define GRI \
    0 // 0 for generalized Riemann invariants and 1 for characteristic variables

#  define empConvTest 0     // 1 for empirical convergence
#  define empConvBoundary 0 // 0 for periodic domain
                            // 1 for sharp interface

#  define useMontecinosBalsara \
    0 // 1 if MontecinosBalsara2020 GRP solver is to be used instead of DET

#  define timeDependentK 0 // 1 for time dependent K in Trento tube law

using namespace std;

extern "C"
{
  // Montecinos-Balsara (2020) GRP solver simplified CK procedure
  extern void
  predictorfd_(double *dt,
               double *ader_tolerance,
               int    *max_iter_ader_dg,
               double *xc,
               double *dx,
               double  w[],
               double  Qhat[],
               double  S_int[],
               double *ArgTubeLaw_m,
               double *ArgTubeLaw_n,
               double *ArgTubeLaw_K,
               double *ArgTubeLaw_A0,
               double *Argpde_rho,
               double *Argpde_nu,
               int    *Accuracy,
               int    *nVar,
               double *cfl,
               int    *nGP,
               int    *NUM_ZONECEN_BASE,
               int    *iE);
}



class Model1d
{
public:
  Model1d()
  {
    flag        = 0;
    partitionID = 0;
    nproc       = 1;
  };
  ~Model1d(){};



  // ADAN
  // int tlType; // 0: ADAN, 1 linear (Alastruey); 2 Pedley
  int    tlTypeArtery, tlTypeVein;
  int    fixKinPedelyTubelaw;
  double fixKinPedelyTubelawValue;
  double tlCoeff; // 1. ADAN and Pedley, 4/3 linear (Alastruey)
  double Lmin;    // minimum vessel lenght for high order scheme, if
                  // vess[i].L<Lmin, then first order is used
  int flag;
  int iTsolveTimeStep;
  // end of simulation
  int endOfSimulation;
  int nUpdate;
  int multCells;
  // momentum balance coefficients
  // xiM dq/dt + 2 d(gammaM q^2/A)/dx + kappaM 8 pi mu / rho q/A = - A/rho dp/dx
  // or
  // dq/dt + d(2 gammaM / xiM q^2/A)/dx + kappaM/xiM 8 pi mu / rho q/A = -
  // A/rho/xiM dp/dx which can be written as dq/dt + d(alphaM q^2/A)/dx +  8 pi
  // muTilde / rhoTilde q/A = - A/rhoTilde dp/dx with alphaM = 2 gammaM / xiM
  // kappaM (given)
  // rhoTilde = rho xiM
  // muTilde = mu kappaM
  double xiM;
  double gammaM;
  int    mODEriemannInvariant; // number of integration steps for computing
                               // integral of Riemann invariant
  double kappaM;
  double alphaM;
  double xiM_initial;
  double gammaM_initial;
  double kappaM_initial;
  double alphaM_initial;

  // change ADAN parameters
  int    hypertense;
  double multCt;
  double multThick;
  double multRad;
  double multResTot;
  int    iniCCO; // vessel index of first segment of CCO network
  double multRadCCO;
  int    multType;
  // modify parameters by group at model initialization
  int         numGroupMult;        // number of groups of regions
  vector<int> numRegionsMult;      // number of regions for each group
  vector<int> typeRegionsMult;     // type of parameter to be multiplied
  vector<int> typeVessRegionsMult; // vessel type of parameter to be multiplied
                                   // (vein (0) or artery (1))
  vector<double>
    coeffRegionsMult; // coeff by which parameter is going to be multiplied
  vector<vector<int>> indexesGroupMult; // region indexes for each group
  // modify parameters by group to match average flow at specific vessel
  int         numGroupMultAverageFlow;         // number of groups of regions
  vector<int> vessIndexRegionsMultAverageFlow; // index of vessel where average
                                               // flow will be monitored
  vector<double>
    vessFlowRegionsMultAverageFlow; // index of vessel where average flow will
                                    // be monitored
  vector<int> numRegionsMultAverageFlow;  // number of regions for each group
  vector<int> typeRegionsMultAverageFlow; // type of parameter to be multiplied
  vector<int>
    typeVessRegionsMultAverageFlow; // vessel type of parameter to be multiplied
                                    // (vein (0) or artery (1))
  vector<double> coeffRegionsMultAverageFlow; // coeff by which parameter is
                                              // going to be multiplied
  vector<vector<int>>
    indexesGroupMultAverageFlow; // region indexes for each group


  // input parameters
  string inpuFilename;
  string testcase;
  string outDir;
  string stateDir;
  string runDir;
  string meshDir;
  string fullStateFile;
  string testRP;
  string statename;
  void
  openOutputFiles();

  // for 3D-1D mech coupling
  string externalPressureFile;
  int    readExternalPressureFromFile;
  int readExternalPressureFromFileType; // 0 for single value per vessel; 1 for
                                        // single value for computational cell
  int    writeAreaAtMidPoint;
  string areaAtMidPointFile;



  string midStateFile;
  // network

  double mmHgPa;
  double mmHgDyncm2;
  string inputFile;
  string vesselFile;
  string modelName;
  string nodeFile;
  string vesselFileASE;
  string vesselFileASErad;
  string vesselFileASEcoord;
  int
    ASEcoord; // flag for the use of ASE coordinates: if ASE files are used it
              // can be decided whether or not to use also ASE coord. Expecially
              // needed when gravity!=0 and the model is NOT in supine position
  int    discretization;
  double discretizationTolerance;
  int    NV;          // Number of vessels
  int    NVglob;      // Number of vessels
  int    NVC;         // Number of columns of vessels' file
  int    NN;          // Number of nodes
  int    NNC;         // Number of columns of nodes' file
  int    NEJ;         // Max number of segments per junction
  int    NVCASE;      // Number of columns of vessels' file
  int    NCELLS;      // Number of cells for all vessels
  int    NCELLSfixed; // if > 0 NCELLS fixed to NCELLSfixed
  int    NCELLSmin;   // minimum Number of cells for all vessels
  double DXmax;       // max dx in cm
  int    multNCELLS;
  double DXmin;
  int    OpenClosed; // 1 closed loop; 0 open loop
  int    rampDistalVenousPressure;
  int
    terminalVeins; // if 0 take values for venous terminal resistance from
                   // table, if 1 take that value plus characteristic impedance
  int    readE;
  string flowType;
  Inflow inflow;
  int    heartType; // 0 for euler, 1 for RK4
  int periType; // 0 for not considering pericardium pressure, 1 for considering
                // pericardium pressure
  double periVol; // See Sun et al 97 pericardium pressure
  double periPhi; // See Sun et al 97 pericardium pressure
  // variables regarding the numerical scheme
  //  int nVar;
  int    junctionType; // 0 conventional, 1 Mynard
  int    nMax;
  int    timeStepping;
  int    schemeTy; // 0 for FV, 1 for DG
  int    fullyDiscrete;
  double T0;
  int    iterationsPerCycle;
  double tolLUMPED;
  double tolAFP;
  double tolADER;
  double tolRP;
  int    riemannProblem; //
  double CFL;
  double CFLreduced;
  int    reduceCFL;
  double timeReduceCFL;
  int    reduceCFLDone;
  double CFLnow;
  double tEnd;
  double tEndOri;
  double tSampleSpace[12];
  int    plotRes;
  int    matchStretchedBloodVolume;
  double StretchedBloodVolume;
  double StretchedBloodVolumeCorrection;
  double timeStretchedBloodVolume;
  int    setStretchedBloodVolume;
  // ####################################################
  //  monitor selected vessels and some model compartments
  int         monitorSomeVessels;
  int         monitorSomeVesselsNum;
  vector<int> monitorSomeVesselsIdxs;
  int         monitorHeart;
  int         monitorPulmonary;
  int         monitorCSF;
  int         monitorCoronaries;
  int         monitorTerminals;
  int         monitorVentricle;
  // ####################################################

  int    plotJunc;
  int    verbose;
  int    iSampleSpace;
  int    sampleSpace;
  double dxSampleSpace;
  int    sampleSpaceASE;
  double qOutFlow;

  double tIniSampleSpace;
  double tEndSampleSpace;
  double tIniSampleMid;
  double tEndSampleMid;
  double multSampleMid;
  double tIniSampleMidCSF;
  double tEndSampleMidCSF;
  double multSampleMidOri;
  int    nSampleSpace;
  int    iStop;
  int    iSample;
  string Simple;
  string paramA;
  string paramB;
  string paramC;
  string paramD;
  string paramE;
  string paramF;
  string paramG;
  int    cerebralVeinsType; // 1 for IJNMBE GG lumped model, 2 for Starling
                            // Resistor network
  double epsilon;           // Relaxation term for viscoelastic
  int    adaptiveEpsilon;
  double deltaEpsilon;
  int    kmrefSelection;
  int    junctionSelection;
  int    junctionOrder;
  double ViscoElastic;
  int    ViscoType;


  int sourceType;
  // Define various variables
  int VeinTubeLaw; // Tube law for veins
  // 0: use classic m=10, n=-1.5, K = E/12/(1-nu^2) * (h/R)^3 with varying h(R)
  // 1: use celerities with c0(R)
  int    ValveType; // Valve type
  double valveLength;
  double valveMst;
  double valveMrg;
  double valveKvo;
  double valveKvc;
  double valveDPopen;
  double valveDPclose;
  double valveSteno;
  int    nValveSpec; // number of valves with specific parameters
  vector<int>
    valveIdxSpec; // array with indexes of valves with specific parameters
  vector<double> valveMstSpec;
  vector<double> valveMrgSpec;
  vector<double> valveKvoSpec;
  vector<double> valveKvcSpec;
  vector<double> valveDPopenSpec;
  vector<double> valveDPcloseSpec;
  int            inflowType;
  double         fixInflow;
  double
  qAorticBifurc(double t);

  double qSample[3]; // state vector for sampling
  // ##############################
  //  MPI variables
  // ##############################
  //  Who am I? How many are we?
  void
  metisPartGraph(int p);
  void
      metisPartGraphClosedLoop(int p);
  int partitionID, nproc, partitioning;
  // Number of MPI Shared Vessels
  int MPISharedVesselsNum;
  int sleepCounter;
#  ifdef USE_MPI
  // Vectors of size MPISharedVesselsNum
  int *mBgnCount, *mEndCount,
    *messReady; // message counters, message ready flags
  // buffers: vectors of pointers
  double     **SendBuff, **RecvBuff;
  MPI_Request *SendReq;
  MPI_Request *RecvReq;
  // MPI Shared vessels: Local ID, remote process
  int       *MPISharedVesselsIndexes, *otherPartitionID;
  MPI_Status SRStatus;
#  endif
  int setExternalPressure;
  // inflow
  double timeSystole, timeDiastole, ampSystole, ampDiastole;

  int    changeInflow;
  int    changeInflowCycles;
  double ampSystoleTarget;

  // #############################
  // Tube laws
  // #############################
  // common parameters (arteries and veins)

  int changeA0Artery; // if 1 then A0 from input data is not A(P_0),but
                      // A(pRefArtery), then P_0 is normally 0
  double pRefArtery;
  double percRefArtery; // percentage of pRefArtery yielded by elastin
  double eeref;
  double ecref;
  double kmref;
  double eerefV;
  double ecrefV;
  double kmrefV;

  double kmAref;
  double kmBref;
  double kmCref;

  double eeref_initial;
  double ecref_initial;
  double kmref_initial;



  // artery parameters
  double rABArtery;
  double rBCArtery;
  double wAeArtery, wAcArtery, wAmArtery;
  double wBeArtery, wBcArtery, wBmArtery;
  double wCeArtery, wCcArtery, wCmArtery;
  double wAeArtery_initial, wAcArtery_initial, wAmArtery_initial;
  double wBeArtery_initial, wBcArtery_initial, wBmArtery_initial;
  double wCeArtery_initial, wCcArtery_initial, wCmArtery_initial;

  double wepluswcA, wepluswcB, wepluswcC;

  double aloArtery, bloArtery, cloArtery, dloArtery;
  double ep0Artery, eprArtery;
  double ep0ArteryA, ep0ArteryB, ep0ArteryC;
  double eprArteryA, eprArteryB, eprArteryC;
  double ep0Artery_initial, eprArtery_initial;
  double ep0ArteryA_initial, ep0ArteryB_initial, ep0ArteryC_initial;
  double eprArteryA_initial, eprArteryB_initial, eprArteryC_initial;

  // vein parameters



  double rABVein;
  double rBCVein;
  double wAeVein, wAcVein, wAmVein;
  double wBeVein, wBcVein, wBmVein;
  double wCeVein, wCcVein, wCmVein;
  double aloVein, bloVein, cloVein, dloVein;
  double ep0Vein, eprVein;

  // vein parameters

  int changeA0Vein; // if 1 then A0 from input data is not A(P_0),but
                    // A(pRefArtery), then P_0 is normally 0
  double pRefVein;
  double percRefVein;
  double pRefSinus;
  double percRefSinus;

  double rABSinus;
  double rBCSinus;
  double wAeSinus, wAcSinus, wAmSinus;
  double wBeSinus, wBcSinus, wBmSinus;
  double wCeSinus, wCcSinus, wCmSinus;
  double aloSinus, bloSinus, cloSinus, dloSinus;
  double ep0Sinus, eprSinus;

  void
  initADANTubeLaw(int i, int j);
  void
  initLinearTubeLaw(int i, int j);

  // Pedley relation for tube law: p = K*((a/a0)**m-(a/a0)**n) + p0 + pe
  double
  computeKappabycelerity(int i, int j);
  void
  initPedleyTubeLaw(int i, int j);

  double cMin;
  double cMax;
  double cMin_initial;
  double cMax_initial;
  double cSinuses;
  double rMinVein;
  double rMinArtery;
  double rMaxVein;
  double GammaVein;
  double mArtery;
  double nArtery;
  double mVein;
  double nVein;
  double mSinus;
  double nSinus;

  double k1Artery, k2Artery, k3Artery;
  double k1Vein, k2Vein, k3Vein;

  double   YoungModulus;
  double   P0;
  double   P0vein;
  double   Rinflow;
  double   pRAfixed;
  double   pFixedInlet;
  string   pRAfile;
  int      pRAsteps;
  double **pRAcurve; //[pRAsteps][2];	// time-pressure curve
  double
  getPressure(double tModel, int iVess);
  // prescribed outflow boundary condition
  int outflowBCtype; // 0 for prescribed flow curve (as used in
  // 3Dvs1D paper for coronary vessels; 1 for fixed value)
  string      outflowBCfile;
  int         outflowBCn;
  int         outflowBCnt;
  vector<int> outflowBCidx;
  void
  outflowBCload(int iVess, int iVessLoc);
  double
                         outflowBCget(const double       &t,
                                      const double       *tdata,
                                      const double       *data,
                                      const unsigned int &n);
  vector<vector<double>> outflowBCdata;

  double PiniA;

  double PiniV;
  int    initialConditionIntracranialTerminals;
  double PiniArterioles;     // [dyn/cm2]
  double PiniCapillaries;    // [dyn/cm2]
  double PiniComp1;          // [dyn/cm2]
  double PiniComp2_Endo;     // [dyn/cm2]
  double PiniComp2_Mid;      // [dyn/cm2]
  double PiniComp2_Epi;      // [dyn/cm2]
  double PiniCapillariesICP; // [dyn/cm2]
  double PiniArteriolesCoronary;
  double rho;
  double rhoOri;

  double pDistalRCR;
  double pProximalRCR;

  double BloodVol;
  double ArterialBloodVol;
  double VenousBloodVol;
  double VenousBloodVolOutIn;
  double ArtBloodVolOutIn;
  double CapillarBloodVol;
  double VenulesBloodVol;
  double ArteriolesBloodVol;
  int    pries;
  double MU;
  double velProf; // 4 for parabolic, 11 for
  double velProf_initial;
  double MUOri;
  double MUPERFORATOR;
  double MUPERFORATOROri;

  // ####################################################
  // Gravity
  // ####################################################
  int ModelGravity; // int used as a flag to indicate wheter the model with
                    // gravity (1) will be used or not (0)
  void
  initGravity();
  void
  updateGravity(vessel *vessLoc);
  double
    gravity; // gravity field 9.81 [m/s^2] -> 981. [cm/s^2] to be read in input
  double         g_intensity; // gravity field intensity to be read in input
  vector<double> g_versor;    // versor containing the position of the gravity
                           // field to be read in input
  // rotation angles taken in input (rigid movement of the body): in stands for
  // initial position and end stands for final position
  double
    rotation_angle_pitch_in; // angle (in grad) of rotation that allows the body
                             // to pass from supine to standing positions
                             // [0=supine face up, 90=standing head up,
                             // 180=supine face down, 270=standing upside down]
  double
    rotation_angle_roll_in; // angle (in grad) of rotation that allows the body
                            // to move left and right [0=supine face up,
                            // 90=supine on the left shoulder, 180=supine face
                            // down, 270=supine on the right shoulder]
  double
    rotation_angle_yaw_in; // angle (in grad) of rotation that allows the body
                           // to move side to side (rotation on the back)
  double rotation_angle_pitch_end;
  double rotation_angle_roll_end;
  double rotation_angle_yaw_end;
  int    TransitionOn; // integer indicating whether the transition from one
                    // position to a new one is active (1) or not (0)
  double TransitionTime;     // Time duration of the transition process
  double TransitionTimeInit; // Time at which the transition will start
  double TransitionTimePost; // Minimum time after the transition is done


  // ###########################
  // lymphatic vessels (See Contarino & Toro 2018)
  // ######################
  int doLymphatics;

  // parameters
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

  double peLymph;

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
  double RILymph;
  // state variables
  double vLymph;
  double wLymph;
  double ILymph;
  double sLymph;

  // auxiliar variables
  double fILymph;
  double fNOLymph;
  double fsLymph;
  double lambdaThetaLymph;
  double tauLymph;
  double kCa1Lymph;
  double kCa2Lymph;
  double ITildeLymph;


  // for boundary conditions
  int    timeBCLymph;
  double t1Lymph;
  double t2Lymph;
  double p1Lymph;
  double p2OutLymph;
  double p2InLymph;


  // ###########################
  // ######################

  void
  setKVessel(double Knew, int iVess);

  double redDtValve;

  // #######################################à
  //  for distributed source
  int    doDistributedSource;
  double tIniDistributedSource;
  int    calibrateDistributedResistances;
  double tIniCalibrateDistributedResistances;
  double RCconstantDistributed;
  double dHatDistributed;
  double tauBaseline;
  double tauDelta;
  double rEstimateUp, rEstimateDown;
  double gammaHatDistributed;
  double numberCyclesCalibrateDistributedResistances;
  double
  getTauTarget(double d, double dHat, double tauBaseline, double tauDelta);
  double
  getRelaxationGamma(double d, double dHat);
  void
           computeDistributedResistances();
  ofstream resistanceDistributedFile;
  int      keepMAPconstant;
  int      keepMAPconstantLocal;
  double   tIniKeepMAPconstant;
  int      keepClusterPressureConstant;
  double   tIniKeepClusterPressureConstant;

  double              tReferenceKeepClusterPressureConstant;
  vector<vector<int>> clusterIndexes;
  vector<int>         clusterReferencePressureIndexes;
  vector<int>         clusterReferenceFlowIndexes;
  vector<double>      clusterRatioTerminalResistances;
  vector<double>      clusterReferencePressure;
  vector<double>      clusterReferenceFlow;
  vector<double>      clusterCurrentFlow;
  int                 numberOfClusters;
  // #######################################à
  //  #########################
  //  Respiration
  //  #########################
  void
  initRespiration();
  void
  setIntraThoracicPressure();
  void
         setIntraAbdominalPressure();
  double pIntraThoracic;
  double pIntraAbdominal;
  int    doValsalva;
  int    respirationOn;
  int NidxRegThorax; // number of elements in list of detailed regions belonging
                     // to thorax
  int NidxRegAbdomen;        // number of elements in list of detailed regions
                             // belonging to abdomen
  vector<int> idxRegThorax;  // list of detailed regions belonging to thorax
  vector<int> idxRegAbdomen; // list of detailed regions belonging to abdomen

  vector<int> idxVessThorax;  // list of vessels in thorax
  vector<int> idxVessAbdomen; // list of vessels in abdomen

  vector<int> idxTerThorax;  // list of terminals in thorax
  vector<int> idxTerAbdomen; // list of terminals in abdomen
  // #########################
  // CLEANED UP TO HERE

  int nJunc, iT;

  int    **juncs;         //[NN][12];			// list of junctions
  int     *juncsNodes;    //[NN];			// list of segments in junction
  int    **vesselBcs;     //[NV][6];		// list containing info on BCS
  int     *juncN;         //[NN];
  double **vessels;       //[NV][NVC];	// vessel file is read into this file
  double **vesselsASE;    //[NV];	// number of ASE nodes for this vessel and
                          //location (deprecated)
  double  *vesselsASEn;   //[NV];	// number of ASE nodes for this vessel
  double **vesselsASEx;   //[NV][NVC];	// ASE nodes location
  double **vesselsASErad; //[NV][NVC];	// ASE nodes radii
  vector<vector<double>> vesselsASEcoord; //[NV][vesselsASEn*3+1]; // number of
                                          //nodes + ASE nodes 3d coordinate
  string         vesselFileNamesASE;
  vector<string> vesselsNamesASE;
  double       **nodes; //[NN][NNC];		// node file is read into this file
  juncData      *junctionsData; //[nJunc];	// list of junctions
  // Network
  vessel *vess; //[NV];	// list of vessels

  //  juncDataHOjunctions *junctionsData;//[nJunc];	// list of junctions

  // Boundary conditions
  double qAorticOpen;
  double pRightAtriumOpen;
  double dt, time, tIni, *dtLoc; //[NV],time;
  int
    readStateMode; // 0: read from same mesh, 1: reads from another simulation
                   // (Lumped models as for 0, while 1D vessels are initialized
                   // with last pressure and flow at midpoint of vessel)
  int    changeElasVisco;
  double dtFixed;
  string dtType;
  int    schemeOrder;
  int    schemeFluct;
  double tIniHighOrder;
  // Files for results
  // Number of cells
  int cells;
  // numerical scheme
  int      reconType; // 0 for WENO, 1 for AENO
  int      reconChar; // 0 for well-balanced rec and 1 for char variables
  int      nREC, nDOF, nDOFs, nGP, nGPint, nGPLTS;
  double  *xGP;
  double  *wGP;
  double  *xGPint;
  double  *wGPint;
  double  *xGPLTS;
  double  *wGPLTS;
  double  *Kxi;
  double  *Ktau;
  double  *Kgrad;
  double **KgradLTS;
  double  *K1;
  double **K1LTS;
  double  *M;
  double **MLTS;
  double  *iM;
  double  *F1;
  double  *F0DG;
  double **SdgLTS;
  double **MSdgLTS;
  double **iMdgLTS;
  double **F0DGLTS;
  double  *FRmDG;
  double  *FLpDG;
  double  *FintDG;
  double  *FRt;
  double  *FLt;
  double  *invJJ;
  double  *MSpace;
  double  *iMSpace;
  double  *KxiSpace;
  double  *KgradSpace;

  // Matrixes for ADER-DG
  double *MSdg; // <phi_k,theta_l>
  double *iMdg; // [phi_k,phi_l]^-1
  double *Sdg;  // <diff(phi_k,xi),theta_l>

  // methods
  double
  max(double a, double b)
  {
    if (a > b)
      return a;
    else
      return b;
  };

  // estimating Young for Rigid Tube
  int estimateYoungRigid; // if 1 will try to match max area deformation
                          // estimateYoungRigidTolArea
  double estimateYoungRigidTolArea;
  double currentMaxAreaDef;
  int    vessRigidIdx;
  double areaRigidMax;
  double areaRigidRef;
  double areaASEMax;
  double minTimeAll;
  double thetaRigid;
  int    rigidFound;

  // general
  int nVeins;
  int nArteries;
  void
  computeCompliance();
  void
         computeComplianceSingleVessel(int i);
  double venousSysComp;
  double venousSysCompGlobal;
  double arterialSysComp;
  double venousSysComp1D;
  double arterialSysComp1D;
  double venousSysCompDyn;
  double arterialSysCompDyn;
  double venousSysComp1DDyn;
  double arterialSysComp1DDyn;

  void
  getYoungRigid();
  void
  getMaxAreaDef();

  // timing lts
  double tolLTS;
  double roundVal;
  void
  setTimeStep();
  double
  correctTimeLTS(double time, double tRef);
  void
  setTimeStepAdaptive(int i);
  double
         computeMaxEigenValue(vessel *vessLoc);
  int    idxMinLTS;
  int    idxMaxLTS;
  int    adaptiveCFLLTS;
  double dtMinLTS;
  double dtMaxLTS;
  double dtMaxLTSround;
  double dtLTSbase;
  double dtMaxLTSLIMIT;
  // methods and variables related to starling resistors
  int       nStarling;
  starling *starlingRes, *starlingResTemp;
  void
  initStarling();
  void
  endStarling();
  void
  readStarlingParameters(string ifile);

  void
         printStarlingParameters();
  string starlingParams;
  void
  solveStarling(int i);
  void
  solveStarlingRP(int i);
  // methods and variables related to venous valves
  int          nValves;
  string       venousValvesParams;
  valveMynard *valves, *valvesTemp;
  void
  readValvesParameters(string ifile);
  void
  printValvesParameters();
  void
  initValves();
  void
  solveValves(int i);
  void
  solveValvesRP(int i);
  // methods and variables related to stenosis
  int                    nStenosis;
  string                 stenosisParams;
  stenosisYoungTsai1973 *stenosis, *stenosisTemp;
  void
  readStenosisParameters(string ifile);
  void
  printStenosisParameters();
  void
  initStenosis();
  void
  solveStenosis(int i);
  void
  solveStenosisRP(int i);
  void
  splittingODE(int     iSteno,
               double  tauLoc,
               int     iL,
               int     iR,
               int     idxL,
               int     idxR,
               int     iDofsL,
               int     iDofsR,
               double  q,
               double *qL,
               double *qR,
               double *xL,
               double *xR);
  void
  splittingODEFunc(int     iSteno,
                   double  tauLoc,
                   int     iL,
                   int     iR,
                   int     idxL,
                   int     idxR,
                   int     iDofsL,
                   int     iDofsR,
                   double  q,
                   double  dq,
                   double *qL,
                   double *qR,
                   double *x,
                   double  dp,
                   double  qStar,
                   double *f);
  void
  splittingODEJacobian(int     iSteno,
                       double  tauLoc,
                       int     iL,
                       int     iR,
                       int     idxL,
                       int     idxR,
                       int     iDofsL,
                       int     iDofsR,
                       double  q,
                       double  dq,
                       double *qL,
                       double *qR,
                       double *x,
                       double  dp,
                       double  qStar,
                       double *f,
                       double *jac);

  void
  splittingODEFuncDelta(int     iSteno,
                        double  tauLoc,
                        int     iL,
                        int     iR,
                        int     idxL,
                        int     idxR,
                        int     iDofsL,
                        int     iDofsR,
                        double  q,
                        double  dq,
                        double *qL,
                        double *qR,
                        double *x,
                        double  dp,
                        double  qStar,
                        double *f,
                        double *dx);

  // methods and variables related to starling resistor
  string       starlingResistorParams;
  valveMynard *starlingResistors;
  void
  initStarlingResistors();
  void
  solveStarlingResistorsLTS();
  // methods and variables related to the heart
  string      heartParams;
  heartMynard heart;
  int         idxAorticRoot;
  int         idxRA[3];
  int         idxRAside[3];
  int         hasHeart, hasPul, hasCsf; // needed for MPI part
  // methods and variables related to pulmonary circulation
  string       pulmonaryParams;
  pulmonarySun pulmonary;
  // methods and variables related to terminals
  int                    idxAVF;
  double                 r0minAVF, r0maxAVF, dtAVF;
  double                 muIniAVF, muEndAVF;
  double                 MUAVF;
  vector<double>         valveDtMin;
  vector<double>         stenoDtMin;
  vector<double>         starlingDtMin;
  vector<double>         terminalsDtMin;
  vector<int>            termID;
  vector<int>            termN;
  vector<vector<double>> termCven;
  vector<vector<int>>    termVidx;
  // associated to the coronary vessels
  ofstream resistanceCoronaryFile;
  int      rTerminalMultNoCerebral, rTerminalMultNoCerebralCycle;
  double   rTerminalMultNoCerebralVal;
  double   rTerminalMulIntracranial;
  double   cTerminalMulIntracranial;
  double   rTerminalMulMiocardium;
  double   rTerminalMulMiocardium_RCA;
  double   rTerminalMulMiocardium_LAD;
  double   rTerminalMulMiocardium_LCX;
  double   rTerminalMulGeneric;
  void
  initTerminals();
  void
  initTerminalsCoronary();
  void
  computeCoronaryResistances();
  void
         setResistancesCoronary();
  double tIniComputeCoronaryResistances, tEndComputeCoronaryResistances;
  double coronaryFlowRef, alphaRelaxConst;
  void
  endTerminals();
  void
                    endTerminalsCoronary();
  vector<int>       terminalList;
  vector<int>       terminalCoronaryList;
  int               nTerminals;
  int               nTerminalsCoronary;
  terminal         *terminals;         // list of terminals
  terminalCoronary *terminalsCoronary; // list of terminals
  double            dtTerminals;
  void
  printCoronaryParameters();
  void
                     readCoronaryParameters(string ifile);
  string             coronaryParams;
  string             leftVentricleParams;
  int                hasVentricle;
  leftVentricleModel leftVentricle;
  void
  coupleTerminal(double *x,
                 double  R,
                 double  pC,
                 double  a0,
                 double  h0,
                 double  ee,
                 double  ec,
                 double  pe,
                 double  ep0,
                 double  epr,
                 double  p0,
                 double  alphaM,
                 double  Gamma,
                 double  T,
                 int     iV,
                 int     iT);

  void
  coupleTerminalFunc(double  aL,
                     double  auL,
                     double  psiL,
                     double  R,
                     double  pC,
                     double  a0,
                     double  h0,
                     double  ee,
                     double  ec,
                     double  pe,
                     double  ep0,
                     double  epr,
                     double  p0,
                     double  alphaM,
                     double  Gamma,
                     double  T,
                     double *x,
                     double *f,
                     int     tlTypeVess);
  void
  coupleTerminalFuncJac(double  aL,
                        double  auL,
                        double  psiL,
                        double  R,
                        double  pC,
                        double  a0,
                        double  h0,
                        double  ee,
                        double  ec,
                        double  pe,
                        double  ep0,
                        double  epr,
                        double  p0,
                        double  alphaM,
                        double  Gamma,
                        double  T,
                        double *x,
                        double *f,
                        double *dx,
                        int     tlTypeVess);

  void
  coupleTerminalInflow(double *x,
                       double  R,
                       double  pC,
                       double  a0,
                       double  h0,
                       double  ee,
                       double  ec,
                       double  pe,
                       double  ep0,
                       double  epr,
                       double  p0,
                       double  alphaM,
                       double  Gamma,
                       double  T,
                       int     iV,
                       int     iT);

  void
  coupleTerminalInflowFunc(double  aL,
                           double  auL,
                           double  psiL,
                           double  R,
                           double  pC,
                           double  a0,
                           double  h0,
                           double  ee,
                           double  ec,
                           double  pe,
                           double  ep0,
                           double  epr,
                           double  p0,
                           double  alphaM,
                           double  Gamma,
                           double  T,
                           double *x,
                           double *f,
                           int     tlTypeVess);
  void
  coupleTerminalInflowFuncJac(double  aL,
                              double  auL,
                              double  psiL,
                              double  R,
                              double  pC,
                              double  a0,
                              double  h0,
                              double  ee,
                              double  ec,
                              double  pe,
                              double  ep0,
                              double  epr,
                              double  p0,
                              double  alphaM,
                              double  Gamma,
                              double  T,
                              double *x,
                              double *f,
                              double *dx,
                              int     tlTypeVess);

  // Stochastic collocation related functions
  void
  generateInstances(vector<string>         paramType,
                    vector<vector<double>> paramValue);

  // Kalman filter-related functions
  int kalmanMode;
  void
  getFullState(vector<double> &state);
  void
  setFullState(const vector<double> &state);
  void
  setParameters(double         theta,
                vector<int>    iVess,
                vector<string> iClass,
                vector<int>    iType);
  void
  setParametersVessel(double theta, int iVess, int iType);
  void
  setParametersStenosis(double theta, int iVess, int iType);
  void
  setParametersHeart(double theta, int iVess, int iType);
  void
  setParametersPedleyTubeLaw(double theta, int iType);
  void
  setParametersLVModel(double theta, int iVess, int iType);
  void
  setParametersSystemic(double theta, int iVess, int iType);
  void
  setParametersInflow(double theta, int iType);
  void
  setParametersGlobalWeights(double theta, int iType);
  void
  setParametersWeights(double theta, int iType);
  void
  setParametersDirector(double theta, int iType);
  void
                 setParametersWindkessel(double theta, int iVess, int iType);
  vector<double> aMeasure, uMeasure, auMeasure, pMeasure;
  double
  observe(string measureClass, int measureType, int measureIndex);
  void
           saveResults(const vector<int>            &measureType,
                       const vector<vector<int>>    &measureIndex,
                       const vector<vector<double>> &measureLocation);
  Observer H;
  void
  initializeMeans();
  void
  updateMeans(int i);
  void
  solveTimeStep();

  void
  solveTimeStep(
    double                        t_interval,
    const vector<int>            &measureType     = vector<int>(),
    const vector<vector<int>>    &measureIndex    = vector<vector<int>>(),
    const vector<vector<double>> &measureLocation = vector<vector<double>>(),
    const double                 &tKalmanStart    = 0.);
  void
  computeVaryingParameterSources(double  a,
                                 double  au,
                                 double  a0,
                                 double  h0,
                                 double  ee,
                                 double  ec,
                                 double  ep0,
                                 double  epr,
                                 double  alphaM,
                                 double  Gamma,
                                 double  Trel,
                                 double  psi,
                                 double  rho,
                                 int     tlTypeVess,
                                 double *momentumTerms);
  // Functions
  void
  init(string ifile, Model1d *m1d = NULL);
  void
  initConvergence(string ifile, int NCELLSCONV, int iter);
  double
  interpolate(const double        xLoc,
              const double       *x,
              const double       *f,
              const unsigned int &n);
  double
  integrate_polyline(const double        a,
                     const double        b,
                     const double       *x,
                     const double       *f,
                     const unsigned int &n);
  void
  llsq(int n, double x[], double y[], double &a, double &b);
  void
      solveTimeStepConvergence();
  int var;
  void
  endConvergence();
  void
  rec_FV_LTSConvergence(vessel *vessLoc);
  void
  rec_FV_LTS_convnew(vessel *vessLoc);
  void
  readParametersClosedLoopConvergence(string _file, int iter);
  void
  pred_FV_LTSConvergence(vessel *vessLoc);
  void
  update_FV_LTSConvergence(vessel *vessLoc);

  // ###############################################
  // ###############################################
  //  SemiDiscrete scheme variables and methods
  void
  solveTimeStepSemiDiscrete();
  void
  reconstructionSemiDiscrete(vessel *vessLoc);
  double
  dCeleritySquareIntda0(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        int    tlTypeVess);
  double
  dCeleritySquareIntdh0(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        int    tlTypeVess);
  double
  dCeleritySquareIntdee(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        int    tlTypeVess);
  double
  dCeleritySquareIntdec(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        int    tlTypeVess);
  double
  dCeleritySquareIntdpe(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        int    tlTypeVess);

  void
  conservativeFluxFunctionElasticModel(double  a,
                                       double  au,
                                       double  a0,
                                       double  h0,
                                       double  ee,
                                       double  ec,
                                       double  ep0,
                                       double  epr,
                                       double  alphaM,
                                       double  Gamma,
                                       double  Trel,
                                       double  psi,
                                       double  rho,
                                       int     tlTypeVess,
                                       double *Flux);
  void
  jacobianInverseElasticModel(double   a,
                              double   au,
                              double   a0,
                              double   h0,
                              double   ee,
                              double   ec,
                              double   ep0,
                              double   epr,
                              double   alphaM,
                              double   Gamma,
                              double   Trel,
                              double   psi,
                              double   rho,
                              int      tlTypeVess,
                              double **iJ);
  void
  computeHLLflux(vessel *vessLoc, int iInterface, double *Dm, double *Dp);

  void
  setPressureSemiDiscreteLTS(int i, double pFixed);
  void
  setPressureOutflowSemiDiscreteLTS(int i, double pFixed);
  void
  solveJunctionsSemiDiscrete(int i);
  void
  setInflowSemiDiscreteLTS(int i);
  void
  setRCRLTSSemiDiscrete(int i);
  void
  fixedPointBackwardsSemiDiscrete(vessel *vessLoc, int iE, double *P1);
  void
  fixedPointForwardsSemiDiscrete(vessel *vessLoc, int iE, double *P1);
  void
  backwardsSemiDiscrete(vessel *vessLoc, int iE, double *P);
  void
  forwardsSemiDiscrete(vessel *vessLoc, int iE, double *P);
  void
  computeSlopesSemiDiscrete(double  a,
                            double  au,
                            double  a0,
                            double  h0,
                            double  ee,
                            double  ec,
                            double  ep0,
                            double  epr,
                            double  alphaM,
                            double  Gamma,
                            double  T,
                            double  psi,
                            double  rho,
                            int     tlType,
                            double  mu,
                            double  g,
                            double  a0Diff,
                            double  h0Diff,
                            double  eeDiff,
                            double  ecDiff,
                            double  peDiff,
                            double *P);
  // ###############################################

  double aD;
  double auC;
  void
  initVesselsconvergence(double L,
                         double rL,
                         double rR,
                         double NCELLS,
                         int    test,
                         double NVARS,
                         double T0,
                         double aC,
                         double aD,
                         double auC,
                         double a0C,
                         double a0D,
                         double kC,
                         double kD,
                         double peC,
                         double peD,
                         double a0ref,
                         double k0ref,
                         double t0,
                         double h0ref,
                         double ee0ref);
  void
  allocateVessels();
  void
  initVessels();
  void
  modifyParametersByRegion();
  void
  modifyParametersByVessel();
  void
  modifyParametersByRegionTerminal();
  void
  initVesselsConvergence(int NCELLSCONV);
  void
  initJunctions();
  void
  initBoundaryConditions();
  void
  setInitialPressure1D();
  void
  setBoundaryConditions(int i);
  void
  end();
  virtual void
  saveFullState(int i);
  void
  readFullState(int i);
  void
  readFullStateAverage(int i);
  void
  readFullState();
  void
  saveFullStateLumpedModels();
  void
  readFullStateLumpedModels();
  void
  readStateCsfUrsinoADAVN();
  void
         saveStateCsfUrsinoADAVN();
  int    sampleFullState;
  double sampleStateMult;
  void
  setBoundaryConditions();
  void
  solveTerminals(int i);
  void
  solveTerminalsCoronary(int i);
  void
  getTerminalsCoronaryDT();
  void
  setTransmuralPressureTerminalsCoronary(int i);
  void
  solveHeart();
  void
  getHeartDT();
  void
  setInflowLTS(int i);
  void
  setOutflowLTS(int i);
  void
  setReflexiveInflowLTS(int i);
  void
  setReflexiveOutflowLTS(int i);
  void
  setRCRLTS(int i);
  void
  setRCRLTSHO(int i);
  void
  setRCRLTS_Left(int i);
  void
  setPressureLTS(int i, double pFixed);

  void
  setPressureOutflowLTS(int i, double pFixed);
  void
  setTransparentLTS(int i);
  void
  setTransparentInflowLTS(int i);
  void
  setPeriodicLTS(int i, int side);
  void
  getBoundaryFluctLeftLTS(int i);
  void
  getBoundaryFluctRightLTS(int i);
  void
  getBoundaryFluctRightTerminalLTS(int i, int termIdx);
  void
  getBoundaryFluctTerminalLTS(int i, int termIdx, int iSide);
  void
  getBoundaryFluctRightTerminalCoronaryLTS(int i, int termIdx);
  void
  getBoundaryFluctTerminalCoronaryLTS(int i, int termIdx, int iSide);
  void
  getLeftRightFluxesLTS(int i);
  void
  getBoundaryFluctLeftFlowLTS(int i, double inflow, double tL, double tR);
  void
  getBoundaryFluctRightFlowLTS(int i, double outflow, double tL, double tR);
  void
  getBoundaryFluctFlowLTS(int    i,
                          double flow,
                          double tL,
                          double tR,
                          double sign,
                          int    side);
  void
  getBoundaryFluctLeftPressureLTS(int i, double pressure, double tL, double tR);
  void
  getBoundaryFluctRightPressureLTS(int    i,
                                   double pressure,
                                   double tL,
                                   double tR);
  void
  rec_FV_LTS(vessel *vessLoc);
  void
  pred_FV_LTS(vessel *vessLoc);
  void
  update_FV_LTS(vessel *vessLoc);
  void
  split_FV_LTS_FO(vessel *vessLoc);
  void
  update_FV_LTS_FO(vessel *vessLoc);
  void
  update_DG_LTS(vessel *vessLoc);
  void
  initADER();

  void
  getJunctionsDT(int i);
  void
  solveJunctions(int i);
  void
  plotJunctions(int i);
  void
  computeDt();
  void
  setDt();

  // iofct
  void
  readParameters(string f); ///@brief read parameters from file (GetPot)
  void
  readTerm(string _file);
  void
  readValveDtMin(string _file);
  void
  readStenoDtMin(string _file);
  void
  readStarlingDtMin(string _file);
  void
  printParameters();
  void
      plotTableADAVNpaper();
  int doplotTableADAVNpaper;
  void
  copyParameters(Model1d *m); ///@bried copy parameters from another class

  // junctions

  // dot
  double
  dotMINMOD(double a, double b);
  double
  dotENO(double a, double b);
  void
  dotGP(double *xiGP, double *wGP, int nGP);
  double
  dotLAGPOL(int n, int m, double *xi, double x);
  double
  dotLAGPOLDiff(int n, int m, double *xii, double x);
  void
  dotSTNFbase(double xi, double tau, int n, double *xGP, double *theta);
  void
  dotSNFbase(double xi, int nREC, double *xGP, double *psi);
  void
  dotINTEGRALS();
  void
  dotMATRIXES();
  void
  dotMATRIXES_ADDITIONAL(int     nREC,
                         double *KxiSpace,
                         double *MSpace,
                         double *KgradSpace);
  void
  dotDGMATRIXES();
  void
  dotWENOLTS(double *UU, double **W, int nCells, int nVar, int nSten);

  void
  dotAENOLTS(double  *UU,
             double **W,
             int      nCells,
             int      nVar,
             int      nSten,
             double   dx);

  // onedmod

  void
  setAreaForPressureIC(int i, double p, double au);
  double
  qBcsLiang2011(double t);
  double
  qBcsInlet(double t);
  double
  qSwanseaSingle(double t);
  double
  qPeak(double t);
  double
  qSwanseaCCA(double t);
  double
  qSwanseaTHO(double t);
  double
  qSwanseaBIF(double t);
  double
  qAlastruey2011(double t);
  double
  qADAN(double t);
  double
  qADANMynard(double t);
  double
  qADANARMTHREE(double tModel);
  double
  qSin2(double t);
  double
  qADANinterosseous(double tModel);
  double
  qADANarm1330(double tModel);
  double
  pVADANcoronaries(double tModel);


  // output
  void
  evalSol(int i, double xSample, int iSample, double *qLoc);
  void
  evalSolSpatialDerivative(int i, double xSample, int iSample, double *qLoc);
  void
      sampleMid(int i, double *qSample, double xSample = -1);
  int convergence;
  void
  outputError(int i);
  void
  convergenceTest(int i, int iTLoc);
  void
  plotMid(int i);
  void
  plotMid_AREA(int i); // unused (?)
  void
  plotMidConvergence(int i);
  void
  plotSpace(int i);
  void
  plotSpaceASE(int i);
  void
  output(int i);
  double
  computeArterialBloodVol();
  double
  computeVenousBloodVol();
  double
  computeCapillarBloodVol();
  double
  computeVenulesBloodVol();
  double
  computeArteriolesBloodVol();

  // empirical convergence
  double aCempConv;
  double aDempConv;
  double auCempConv;
  double auDempConv;
  int    empConvergenceType; // 0 : sinusoidal function
  void
  exactSolViscoSinus(double  L,
                     double  T0,
                     double  aC,
                     double  aD,
                     double  auC,
                     double  t,
                     double  x,
                     double *sol);
  double
  exactSolViscoSourceSinus(double Gamma,
                           double L,
                           double T0,
                           double a0C,
                           double aC,
                           double aD,
                           double auC,
                           double ec,
                           double ee,
                           double ep0,
                           double epr,
                           double h0,
                           double rho,
                           double t,
                           double x);
  double
  exactSolViscoSourceSinus2(double Gamma,
                            double L,
                            double T0,
                            double a0C,
                            double aC,
                            double aD,
                            double auC,
                            double ec,
                            double ee,
                            double ep0,
                            double epr,
                            double h0,
                            double rho,
                            double t,
                            double x);
  void
  setExactSolutionConvergence(int i, int side);

  // microvasculature ADAVN
  int            nTerMio;
  int            nVessMio;
  int            idxCoronary, idxCoronaryLeft, idxCoronaryRight;
  vector<int>    idxTerMio; // indexes of coronary vasTer
  vector<int>    idxVessMio;
  vector<int>    idxVessMioLoc;
  vector<double> r2VessMio;
  vector<double> volVessMio;
  vector<double> resTotVessMio;
  vector<int>    chamberVessMio; // chambers for coronary vasTer
  vector<int>    chamberTerMio;  // chambers for coronary vasTer
  vector<int>
         groupVessMio;   // groups for terminal vessels (0 RCA, 1  LAD, 2 LCX)
  double coronaryMin[5]; // factor by which coronary resistance R2
                         // is reduced when it is minimum
  double coronaryMax[5]; // factor by which coronary resistance R2
                         // is increased when it is minimum

  int    idxCCO;
  int    coronaryType;
  double c1ConstMio, c2ConstMio;
  double v1ConstMio, v2ConstMio;
  double r1ConstMio, r2ConstMio;
  double epiConstMio, midConstMio, endoConstMio;
  double caConstMio, cVenConstMio;
  double gammaVolendoMio, gammaVolmidMio, gammaVolepiMio;
  double pCEPEpiConstMio, pCEPMidConstMio, pCEPEndoConstMio;
  double alphaConstMio, alphaSIPConstMio;

  // CBF autoregulation Ursino-Giannessi

  int         cbfType;
  double     *idxPoly;
  int         nTerIntracranialArteries;
  vector<int> idxTerIntracranialArteries;
  double      alphaCBF;
  void
  getIntracranialTerminalVessels();
  void
         readRegulationParameters(string _file);
  string cbfParams;
  void
  printRegulationParameters();
  void
  solveCBFautoregulation(int i);
  void
  solveCBFautoregulationUrsinoLodi(int i);
  void
  initCBFregulationUrsino();
  void
  computeCBFCompliancesUrsino();
  void
  computeCBFResistancesUrsino();
  void
         outputCBF();
  double tau_aut;
  double G_aut;
  double cbfBase;
  double tau_co2;
  double G_co2;
  double k_co2;
  double b_co2;
  double P_co2;
  double sat1;
  double sat2;
  double pco2base;
  double pco2;
  double tIniCbf;
  int    cbfTimeStep;
  string cbfSetFile;

  // CSF ADAVN
  int csfPartitionType; // 0 uses MPI_allreduce, 1 ensures exact volume matching
                        // between 1 and N partitions
  int    csfPartition;
  double pIntracranial;
  double pIntracranialIni;
  double pIntracranialEnd;
  double csfVariables[2];
  int    csfModel;
  double tIniCsf;
  int    idxDuraMater;
  int    idxEncephalon;
  int    idxBrain;
  int    idxCerebellum;
  int    idxPons;
  int    allCsfDone;
  double
  roundDouble(double val, double decimals);
  void
         insertion_sort(std::vector<int>    &arr,
                        std::vector<double> &weight,
                        int                  length);
  double volCerebralBlood, volCerebralBloodOld;
  double volBlood, vol1DArt, vol1DVen, volTerminal, volTerminalArt,
    volTerminalVen, volTerminalCoronary, volTerminalCoronaryArt,
    volTerminalCoronaryCap, volTerminalCoronaryVen, volHeart, volPulmonary;
  ofstream sampleVol;
  double   volIntracranial1D;
  double   volIntracranialTerminal;
  double   volIntracranialArteries1D;
  double   volIntracranialTerminalArt;
  double   volIntracranialVeins1D;
  double   volIntracranialTerminalVen;
  void
  computeCerebralBloodVolume();
  void
  computeVesselVolume();
  void
  computeTotalBloodVolume();
  void
  initCsfUrsino();
  void
  solveCsfUrsino();
  void
  solveCsfUrsinoADAVN();

  int         nIntracranial;
  double      timing;
  int         aortaPartition;
  double      rRatioTerminalResistancesLocal;
  double      rRatioTerminalResistances;
  double      rRatioTerminalResistancesMult;
  int         nIntracranialLoc;
  int         nIntracranialArteriesLoc;
  int         nIntracranialVeinsLoc;
  int         nIntracranialArteries;
  int         nIntracranialVeins;
  int         nIntracranialGlob;
  int         nIntracranialArteriesGlob;
  int         nIntracranialVeinsGlob;
  int         nIntracranialTerminals;
  int         nIntracranialTerminalsLoc;
  double      timeCsf;
  int         updateGlobalCsf;
  double      roundValVol;
  int         plottedLast;
  double      dtCsf;
  vector<int> idxCerebralVessels;
  vector<int> idxCerebralArteries;
  vector<int> idxCerebralVeins;
  vector<int> idxCerebralTerminals;
  int        *idxCerebralVesselsGlobal;
  int        *idxCerebralArteriesGlobal;
  int        *idxCerebralVeinsGlobal;
  double     *VolCerebralVesselsGlobal;
  double     *VolCerebralArteriesGlobal;
  double     *VolCerebralVeinsGlobal;
  int        *idxCerebralTerminalsGlobal;
  double     *VolCerebralTerminalsGlobal;
  double     *VolCerebralTerminalsArtGlobal;
  double     *VolCerebralTerminalsVenGlobal;
#  ifdef USE_MPI
  MPI_Status  *commCerebralTerminals;
  MPI_Status  *commCerebralTerminalsArt;
  MPI_Status  *commCerebralTerminalsVen;
  MPI_Status  *commCerebralVessels;
  MPI_Status  *commCerebralArteries;
  MPI_Status  *commCerebralVeins;
  MPI_Request *RecvReqCsf;
  MPI_Request *SendReqCsf;
  int         *messReadyCsf;
  int         *mEndCountCsf;
  int         *mBgnCountCsf;
  int         *updateCsfVec;
#  endif
  int         updateCsf;
  int       **idxCerebralVesselsLoc;
  int       **idxCerebralVesselsLocOtherPartition;
  int       **idxCerebralArteriesLoc;
  int       **idxCerebralArteriesLocOtherPartition;
  int       **idxCerebralVeinsLoc;
  int       **idxCerebralVeinsLocOtherPartition;
  double    **VolCerebralVesselsLoc;
  double    **VolCerebralArteriesLoc;
  double    **VolCerebralVeinsLoc;
  int       **idxCerebralVesselsLocOrder;
  int       **idxCerebralArteriesLocOrder;
  int       **idxCerebralVeinsLocOrder;
  int       **idxCerebralTerminalsLoc;
  double    **VolCerebralTerminalsLoc;
  double    **VolCerebralTerminalsArtLoc;
  double    **VolCerebralTerminalsVenLoc;
  int       **idxCerebralTerminalsLocOrder;
  int        *nCerebralVesselsLoc;
  int        *nCerebralArteriesLoc;
  int        *nCerebralVeinsLoc;
  int        *nCerebralTerminalsLoc;
  double     *VolCerebralVessels;
  double     *VolCerebralArteries;
  double     *VolCerebralVeins;
  double     *VolCerebralTerminals;
  double     *VolCerebralTerminalsArt;
  double     *VolCerebralTerminalsVen;
  vector<int> nCerebralVessels;
  vector<int> nCerebralTerminals;
  int        *vesselsFlagCsf;
  int        *terminalsFlagCsf;
  double      elastanceCsf, complianceCsf, rInCsf, rOutCsf;
  ofstream    sampleCsf;
  // ########################################
  virtual double
  sourceADAN(const double &a,
             const double &au,
             const double &mu,
             double        g_axial,
             const double &rho);
  virtual double
  tauADAN(const double &a, const double &au, const double &mu);
  virtual double
  flowFromtauADAN(const double &a, const double &tau, const double &mu);
  virtual double
  tauADANdiffq(const double &a, const double &au, const double &mu);
  virtual double
  tauADANdiffa(const double &a, const double &au, const double &mu);

  double
         viscosity(double a0);
  double ep0, epr, p0;

  // Solve junctions
  void
  splittingFunc(int     N,
                double *x,
                double *a,
                double *au,
                double *psi,
                double *a0,
                double *h0,
                double *ee,
                double *ec,
                double *pe,
                double *p0,
                double *ep0,
                double *epr,
                double *Gamma,
                double *T,
                double *alphaM,
                double  rho,
                double *signs,
                double  qIN,
                double *f,
                int    *vessTlType);

  void
  splittingJacobian(int     N,
                    double *x,
                    double *a,
                    double *au,
                    double *psi,
                    double *a0,
                    double *h0,
                    double *ee,
                    double *ec,
                    double *pe,
                    double *p0,
                    double *ep0,
                    double *epr,
                    double *Gamma,
                    double *T,
                    double *alphaM,
                    double  rho,
                    double *signs,
                    double  qIN,
                    double *jac,
                    int    *vessTlType);

  void
  splittingDelta(int     N,
                 double *x,
                 double *dx,
                 double *a,
                 double *au,
                 double *psi,
                 double *a0,
                 double *h0,
                 double *ee,
                 double *ec,
                 double *pe,
                 double *p0,
                 double *ep0,
                 double *epr,
                 double *Gamma,
                 double *T,
                 double *alphaM,
                 double  rho,
                 double *signs,
                 double  qIN,
                 double *f,
                 int     iJunc,
                 int    *vessTlType);
  void
  splitting(int     N,
            double *a,
            double *au,
            double *psi,
            double *a0,
            double *h0,
            double *ee,
            double *ec,
            double *pe,
            double *p0,
            double *ep0,
            double *epr,
            double *Gamma,
            double *T,
            double *alphaM,
            double  rho,
            double *solution,
            double *signs,
            double  qIN,
            int     iJunc,
            int    *vessTlType);


  // Solve junctions with Mynard pressure loss coefficient
  void
  splittingFuncMynard(int     N,
                      double *x,
                      double *a,
                      double *au,
                      double *psi,
                      double *a0,
                      double *h0,
                      double *ee,
                      double *ec,
                      double *pe,
                      double *p0,
                      double *ep0,
                      double *epr,
                      double *Gamma,
                      double *T,
                      double *alphaM,
                      double  rho,
                      double *signs,
                      double  qIN,
                      double *kLoss,
                      double *f,
                      int    *iPtot,
                      int    *jPtot,
                      int     tlTypeVess);

  void
  splittingJacobianMynard(int     N,
                          double *x,
                          double *a,
                          double *au,
                          double *psi,
                          double *a0,
                          double *h0,
                          double *ee,
                          double *ec,
                          double *pe,
                          double *p0,
                          double *ep0,
                          double *epr,
                          double *Gamma,
                          double *T,
                          double *alphaM,
                          double  rho,
                          double *signs,
                          double  qIN,
                          double *kLoss,
                          double *jac,
                          int    *iPtot,
                          int    *jPtot,
                          int     tlTypeVess);


  void
  splittingDeltaMynard(int     N,
                       double *x,
                       double *dx,
                       double *a,
                       double *au,
                       double *psi,
                       double *a0,
                       double *h0,
                       double *ee,
                       double *ec,
                       double *pe,
                       double *p0,
                       double *ep0,
                       double *epr,
                       double *Gamma,
                       double *T,
                       double *alphaM,
                       double  rho,
                       double *signs,
                       double  qIN,
                       double *kLoss,
                       double *f,
                       int     iJunc,
                       int    *iPtot,
                       int    *jPtot);
  void
  splittingMynard(int     N,
                  double *a,
                  double *au,
                  double *psi,
                  double *a0,
                  double *h0,
                  double *ee,
                  double *ec,
                  double *pe,
                  double *p0,
                  double *ep0,
                  double *epr,
                  double *Gamma,
                  double *T,
                  double *alphaM,
                  double  rho,
                  double *solution,
                  double *signs,
                  double  aa0,
                  double  aa1,
                  double *theta,
                  double  qIN,
                  int     iJunc);

  void
  computeJunctionLossCoefficient(int     N,
                                 double  a0,
                                 double  a1,
                                 double *x,
                                 double *signs,
                                 double *theta,
                                 double *kLoss,
                                 int    *iPtot,
                                 int    *jPtot);

  // #############################################


  void
  splittingLinearRHS(int     N,
                     double *a,
                     double *au,
                     double *psi,
                     double *ax,
                     double *aux,
                     double *psix,
                     double *T,
                     double *signs,
                     double *celLin,
                     double *rhou2c2a,
                     double *RIC,
                     double *RHS);



  void
  splittingJacobianLinear(int     N,
                          double *a,
                          double *au,
                          double *psi,
                          double *ax,
                          double *aux,
                          double *psix,
                          double *T,
                          double *signs,
                          double *celLin,
                          double *rhou2c2a,
                          double *RIC,
                          double *jac);

  void
  splittingLinear(int     N,
                  double *a,
                  double *au,
                  double *psi,
                  double *ax,
                  double *aux,
                  double *psix,
                  double *a0,
                  double *h0,
                  double *ee,
                  double *ec,
                  double *pe,
                  double *p0,
                  double *ep0,
                  double *epr,
                  double *Gamma,
                  double *T,
                  double *alphaM,
                  double  rho,
                  double *solution,
                  double *signs,
                  double  qIN,
                  int     iJunc,
                  int     tlTypeVess);


  // Boundary conditions

  void
  outflowBCfixAFunc(double  aL,
                    double  auL,
                    double  psiL,
                    double  aR,
                    double *x,
                    double *f,
                    double  a0,
                    double  h0,
                    double  ee,
                    double  ec,
                    double  pe,
                    double  ep0,
                    double  epr,
                    double  p0,
                    double  Gamma,
                    double  T,
                    double  alphaM,
                    double  rho,
                    int     tlTypeVess);


  void
  outflowBCfixADelta(double  aL,
                     double  auL,
                     double  psiL,
                     double  aR,
                     double *x,
                     double *f,
                     double *dx,
                     double  a0,
                     double  h0,
                     double  ee,
                     double  ec,
                     double  pe,
                     double  ep0,
                     double  epr,
                     double  p0,
                     double  Gamma,
                     double  T,
                     double  alphaM,
                     double  rho);


  void
  outflowBCfixA(double  aL,
                double  auL,
                double  psiL,
                double  aR,
                double *x,
                double  a0,
                double  h0,
                double  ee,
                double  ec,
                double  pe,
                double  ep0,
                double  epr,
                double  p0,
                double  Gamma,
                double  T,
                double  alphaM,
                double  rho,
                int     tlTypeVess);

  void
  inflowBCfixQDelta(double  aR,
                    double  auR,
                    double  psiR,
                    double  auL,
                    double *x,
                    double *f,
                    double  a0,
                    double  h0,
                    double  ee,
                    double  ec,
                    double  ep0,
                    double  epr,
                    double  Gamma,
                    double  T,
                    double  alphaM,
                    double  rho,
                    double *dx,
                    int     tlTypeVess);


  void
  inflowBCfixQFunc(double  aR,
                   double  auR,
                   double  psiR,
                   double  auL,
                   double *x,
                   double *f,
                   double  a0,
                   double  h0,
                   double  ee,
                   double  ec,
                   double  ep0,
                   double  epr,
                   double  Gamma,
                   double  T,
                   double  alphaM,
                   double  rho,
                   int     tlTypeVess);


  void
  inflowBCfixQ(double  aR,
               double  auR,
               double  psiR,
               double  auL,
               double *x,
               double  a0,
               double  h0,
               double  ee,
               double  ec,
               double  ep0,
               double  epr,
               double  Gamma,
               double  T,
               double  alphaM,
               double  rho,
               int     tlTypeVess);


  void
  inflowFlowLTS(double  aR,
                double  auR,
                double  psiR,
                double  auL,
                double *x,
                double  a0,
                double  h0,
                double  ee,
                double  ec,
                double  ep0,
                double  epr,
                double  Gamma,
                double  T,
                double  alphaM,
                double  rho,
                int     iV,
                int     tlTypeVess);

  void
  outflowFlowLTS(double  aL,
                 double  auL,
                 double  psiL,
                 double  auR,
                 double *x,
                 double  a0,
                 double  h0,
                 double  ee,
                 double  ec,
                 double  pe,
                 double  ep0,
                 double  epr,
                 double  p0,
                 double  Gamma,
                 double  T,
                 double  alphaM,
                 double  rho,
                 int     iV);



  void
  outflowBCfixQDelta(double  aL,
                     double  auL,
                     double  psiL,
                     double  auR,
                     double *x,
                     double *f,
                     double *dx,
                     double  a0,
                     double  h0,
                     double  ee,
                     double  ec,
                     double  pe,
                     double  ep0,
                     double  epr,
                     double  p0,
                     double  Gamma,
                     double  T,
                     double  alphaM,
                     double  rho,
                     int     tlTypeVess);


  void
  outflowBCfixQFunc(double  aL,
                    double  auL,
                    double  psiL,
                    double  auR,
                    double *x,
                    double *f,
                    double  a0,
                    double  h0,
                    double  ee,
                    double  ec,
                    double  ep0,
                    double  epr,
                    double  Gamma,
                    double  T,
                    double  alphaM,
                    double  rho,
                    int     tlTypeVess);


  void
  outflowBCfixQ(double  aL,
                double  auL,
                double  psiL,
                double  auR,
                double *x,
                double  a0,
                double  h0,
                double  ee,
                double  ec,
                double  pe,
                double  ep0,
                double  epr,
                double  p0,
                double  Gamma,
                double  T,
                double  alphaM,
                double  rho,
                int     iV);



  void
  inflowBCfixADelta(double  aR,
                    double  auR,
                    double  psiR,
                    double  aL,
                    double *x,
                    double *f,
                    double *dx,
                    double  a0,
                    double  h0,
                    double  ee,
                    double  ec,
                    double  pe,
                    double  ep0,
                    double  epr,
                    double  p0,
                    double  Gamma,
                    double  T,
                    double  alphaM,
                    double  rho);


  void
  inflowBCfixAFunc(double  aR,
                   double  auR,
                   double  psiR,
                   double  aL,
                   double *x,
                   double *f,
                   double  a0,
                   double  h0,
                   double  ee,
                   double  ec,
                   double  pe,
                   double  ep0,
                   double  epr,
                   double  p0,
                   double  Gamma,
                   double  T,
                   double  alphaM,
                   double  rho,
                   int     tlTypeVess);


  void
  inflowBCfixA(double  aR,
               double  auR,
               double  psiR,
               double  aL,
               double *x,
               double  a0,
               double  h0,
               double  ee,
               double  ec,
               double  pe,
               double  ep0,
               double  epr,
               double  p0,
               double  Gamma,
               double  T,
               double  alphaM,
               double  rho,
               int     tlTypeVess);



  void
  terminalRFunc(double  aL,
                double  auL,
                double  psiL,
                double  R1,
                double  pCold,
                double  a0,
                double  h0,
                double  ee,
                double  ec,
                double  pe,
                double  ep0,
                double  epr,
                double  p0,
                double  alphaM,
                double  Gamma,
                double  T,
                double *x,
                double *f,
                int     tlTypeVess);

  void
  terminalRJac(double  aL,
               double  auL,
               double  psiL,
               double  R1,
               double  pCold,
               double  a0,
               double  h0,
               double  ee,
               double  ec,
               double  pe,
               double  ep0,
               double  epr,
               double  p0,
               double  alphaM,
               double  Gamma,
               double  T,
               double *x,
               double *f,
               double *dx,
               int     tlTypeVess);

  void
  terminalR(double *x,
            double  R1,
            double  pC,
            double  a0,
            double  h0,
            double  ee,
            double  ec,
            double  pe,
            double  ep0,
            double  epr,
            double  p0,
            double  alphaM,
            double  Gamma,
            double  T,
            int     iV,
            int     iT);

  void
  terminalRFuncinflow(double  aL,
                      double  auL,
                      double  psiL,
                      double  R1,
                      double  pCold,
                      double  a0,
                      double  h0,
                      double  ee,
                      double  ec,
                      double  pe,
                      double  ep0,
                      double  epr,
                      double  p0,
                      double  alphaM,
                      double  Gamma,
                      double  T,
                      double *x,
                      double *f,
                      int     tlTypeVess);

  void
  terminalRJacinflow(double  aL,
                     double  auL,
                     double  psiL,
                     double  R1,
                     double  pCold,
                     double  a0,
                     double  h0,
                     double  ee,
                     double  ec,
                     double  pe,
                     double  ep0,
                     double  epr,
                     double  p0,
                     double  alphaM,
                     double  Gamma,
                     double  T,
                     double *x,
                     double *f,
                     double *dx,
                     int     tlTypeVess);

  void
  terminalRinflow(double *x,
                  double  R1,
                  double  pC,
                  double  a0,
                  double  h0,
                  double  ee,
                  double  ec,
                  double  pe,
                  double  ep0,
                  double  epr,
                  double  p0,
                  double  alphaM,
                  double  Gamma,
                  double  T,
                  int     iV,
                  int     iT);


  // Functions for the DOT solver
  double
  dotPATH_ZETA(double cL,
               double cR,
               double p0,
               double peL,
               double peR,
               double s);
  double
  dotPATHdiff_ZETA(double a,
                   double a0,
                   double a0L,
                   double a0R,
                   double cL,
                   double cR,
                   double ec,
                   double ecL,
                   double ecR,
                   double ee,
                   double eeL,
                   double eeR,
                   double ep0,
                   double ep0L,
                   double ep0R,
                   double epr,
                   double eprL,
                   double eprR,
                   double h0,
                   double h0L,
                   double h0R,
                   double peL,
                   double peR,
                   int    tlTypeVess);

  double
  getee(double a0, int type, int region);
  double
  getec(double a0, int type, int region);
  double
  getkm(double a0, int type, int region);
  double
  gete0(double a0, int type, int region);
  double
  getp0(int type, int region);
  double
  geter(int type, int region);
  double
  getThickness(double a0, int type, int region);
  double
  a0FpsADAN(double pst, double ast, int type, int region, int tlTypeVess);
  double
  a0FpsElastinADAN(double pst, double ast, int type, int region, int iV);
  double
  a0FpsADANepFixed(double pst, double ast, int type, int region);
  double
  eprFpsFuncADAN(double pst,
                 double ast,
                 double a0,
                 double epr,
                 int    type,
                 int    region);
  double
  eprFpsADAN(double pst, double ast, double a0, int type, int region);
  double
  aFpADAN(double p,
          double a0,
          double h0,
          double ee,
          double ec,
          double pe,
          double ep0,
          double epr,
          double p0,
          double aRef,
          int    tlTypeVess);
  double
  aFpADANrec(double p,
             double a0,
             double h0,
             double ee,
             double ec,
             double pe,
             double ep0,
             double epr,
             double p0,
             int    iE,
             int    iV,
             double aRef,
             int    tlTypeVess);
  double
  aFpADANviscorec(double p,
                  double a0,
                  double h0,
                  double ee,
                  double ec,
                  double pe,
                  double ep0,
                  double epr,
                  double p0,
                  double psi,
                  double Gamma,
                  int    iE,
                  int    iV);

  double
  aFzeta(double zeta,
         double aS,
         double a0,
         double h0,
         double ee,
         double ec,
         double pe,
         double ep0,
         double epr,
         double p0,
         double aRef,
         int    tlTypeVess);
  double
  zetaADAN(double a,
           double a0,
           double h0,
           double ee,
           double ec,
           double ep0,
           double epr,
           int    tlTypeVess);


  // Tube law
  double
  pFaADAN(double a,
          double a0,
          double h0,
          double ee,
          double ec,
          double pe,
          double ep0,
          double epr,
          double p0,
          int    tlTypeVess);
  //
  double
  dzetadaADAN(double a,
              double a0,
              double h0,
              double ee,
              double ec,
              double ep0,
              double epr,
              int    tlTypeVess);
  double
  dzetada0ADAN(double a,
               double a0,
               double h0,
               double ee,
               double ec,
               double ep0,
               double epr,
               int    tlTypeVess);
  double
  dzetadh0ADAN(double a,
               double a0,
               double h0,
               double ee,
               double ec,
               double ep0,
               double epr,
               int    tlTypeVess);
  double
  dzetadeeADAN(double a,
               double a0,
               double h0,
               double ee,
               double ec,
               double ep0,
               double epr,
               int    tlTypeVess);
  double
  dzetadecADAN(double a,
               double a0,
               double h0,
               double ee,
               double ec,
               double ep0,
               double epr,
               int    tlTypeVess);

  double
  dzetadaADANLTS(double a,
                 double a0,
                 double h0,
                 double ee,
                 double ec,
                 double ep0,
                 double epr,
                 double mu,
                 double expmu,
                 double logmu,
                 double sqrta,
                 double sqrta0,
                 double phi);
  double
  dzetada0ADANLTS(double a,
                  double a0,
                  double h0,
                  double ee,
                  double ec,
                  double ep0,
                  double epr,
                  double mu,
                  double expmu,
                  double logmu,
                  double sqrta,
                  double sqrta0,
                  double phi);
  double
  dzetadh0ADANLTS(double a,
                  double a0,
                  double h0,
                  double ee,
                  double ec,
                  double ep0,
                  double epr,
                  double mu,
                  double expmu,
                  double logmu,
                  double sqrta,
                  double sqrta0,
                  double phi);
  double
  dzetadeeADANLTS(double a,
                  double a0,
                  double h0,
                  double ee,
                  double ec,
                  double ep0,
                  double epr,
                  double mu,
                  double expmu,
                  double logmu,
                  double sqrta,
                  double sqrta0,
                  double phi);
  double
  dzetadecADANLTS(double a,
                  double a0,
                  double h0,
                  double ee,
                  double ec,
                  double ep0,
                  double epr,
                  double mu,
                  double expmu,
                  double logmu,
                  double sqrta,
                  double sqrta0,
                  double phi);



  double
  dzetadep0ADANviscoLTS(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        double mu,
                        double expmu,
                        double logmu,
                        double sqrta,
                        double sqrta0,
                        double phi);
  double
  dzetadeprADANviscoLTS(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        double mu,
                        double expmu,
                        double logmu,
                        double sqrta,
                        double sqrta0,
                        double phi);

  double
  dzetadep0ADANLTS(double a,
                   double a0,
                   double h0,
                   double ee,
                   double ec,
                   double ep0,
                   double epr,
                   double mu,
                   double expmu,
                   double logmu,
                   double sqrta,
                   double sqrta0,
                   double phi);
  double
  dzetadeprADANLTS(double a,
                   double a0,
                   double h0,
                   double ee,
                   double ec,
                   double ep0,
                   double epr,
                   double mu,
                   double expmu,
                   double logmu,
                   double sqrta,
                   double sqrta0,
                   double phi);


  // Celerity
  double
  celerityADAN(double a,
               double au,
               double a0,
               double h0,
               double ee,
               double ec,
               double ep0,
               double epr,
               double alphaM,
               double rho,
               int    tlTypeVess);

  double
  riemannInvariantNonLinear(double aL,
                            double psiL,
                            double aR,
                            double psiR,
                            double a0,
                            double h0,
                            double ee,
                            double ec,
                            double ep0,
                            double epr,
                            double alphaM,
                            double rho,
                            double Trel,
                            double Gamma,
                            int    tlTypeVess,
                            double auL,
                            double sign);

  double
  riemannInvariantNonLinearIntegrand(double a,
                                     double psi,
                                     double u,
                                     double a0,
                                     double h0,
                                     double ee,
                                     double ec,
                                     double ep0,
                                     double epr,
                                     double alphaM,
                                     double rho,
                                     double Trel,
                                     double Gamma,
                                     int    tlTypeVess,
                                     double sign);

  void
  jacobianElasticModel(double  a,
                       double  au,
                       double  a0,
                       double  h0,
                       double  ee,
                       double  ec,
                       double  ep0,
                       double  epr,
                       double  alphaM,
                       double  Gamma,
                       double  Trel,
                       double  psi,
                       double  rho,
                       int     tlTypeVess,
                       double *J);

  double
  physicalFluxCeleritySquareTerm(double aL,
                                 double psiL,
                                 double aR,
                                 double psiR,
                                 double a0,
                                 double h0,
                                 double ee,
                                 double ec,
                                 double ep0,
                                 double epr,
                                 double alphaM,
                                 double rho,
                                 double Trel,
                                 double Gamma,
                                 int    tlTypeVess,
                                 double auL,
                                 double sign);
  double
  physicalFluxIntegrand(double a,
                        double psi,
                        double u,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double alphaM,
                        double rho,
                        double Trel,
                        double Gamma,
                        int    tlTypeVess,
                        double sign);

  void
  computeDTvessel(vessel *vessLoc);

  void
  computeDTvesselADANviscoLTS(vessel *vessLoc);

  double
  celerityADANvisco(double a,
                    double au,
                    double a0,
                    double h0,
                    double ee,
                    double ec,
                    double ep0,
                    double epr,
                    double alphaM,
                    double Gamma,
                    double Trel,
                    double psi,
                    double rho,
                    int    tlTypeVess);
  double
  celerityADANvisco0(double a,
                     double au,
                     double a0,
                     double h0,
                     double ee,
                     double ec,
                     double ep0,
                     double epr,
                     double alphaM,
                     double Gamma,
                     double Trel,
                     double psi,
                     double rho,
                     int    tlTypeVess);

  double
  dzetadaADANvisco(double a,
                   double a0,
                   double h0,
                   double ee,
                   double ec,
                   double ep0,
                   double epr,
                   double Gamma,
                   double psi,
                   int    tlTypeVess);
  double
  dzetadpsiADANvisco(double a,
                     double a0,
                     double h0,
                     double ee,
                     double ec,
                     double ep0,
                     double epr,
                     double Gamma,
                     double psi,
                     int    tlTypeVess);


  double
  pFaADANvisco(double a,
               double a0,
               double h0,
               double ee,
               double ec,
               double pe,
               double ep0,
               double epr,
               double p0,
               double psi,
               double Gamma,
               int    tlTypeVess);
  double
  zetaADANvisco(double a,
                double a0,
                double h0,
                double ee,
                double ec,
                double ep0,
                double epr,
                double psi,
                double Gamma,
                int    tlTypeVess);


  void
  jacSourceADANviscoLTS(double   a,
                        double   au,
                        double   mu,
                        double   g_axial,
                        double   rho,
                        double   Trel,
                        double   R,
                        double   L,
                        double   a0,
                        double   h0,
                        double   ee,
                        double   ec,
                        double   ep0,
                        double   epr,
                        double   Gamma,
                        double   psi,
                        int      tlTypeVess,
                        int      doDistributedSource,
                        double **JS);

  void
  dotFLUCT_ZETA_visco_LTS(double *qm,
                          double *qp,
                          double  p0,
                          double  Gamma,
                          double  Trel,
                          double  alphaM,
                          double *fP,
                          double *fM,
                          int     nVar,
                          int     iE,
                          double *rADAN,
                          double *irADAN,
                          double *lADAN,
                          double  rho,
                          int     pathID,
                          double  aRef,
                          int     tlTypeVess);

  void
  dotFLUCT_ZETA_visco_LTS_BOUNDARY_M(double *qm,
                                     double *qp,
                                     double  p0,
                                     double  Gamma,
                                     double  Trel,
                                     double  alphaM,
                                     double *fP,
                                     double *fM,
                                     int     nVar,
                                     int     iE,
                                     double *rADAN,
                                     double *irADAN,
                                     double *lADAN,
                                     double  rho,
                                     int     pathID,
                                     double  aRef,
                                     int     tlTypeVess);
  void
  dotFLUCT_ZETA_visco_LTS_BOUNDARY_P(double *qm,
                                     double *qp,
                                     double  p0,
                                     double  Gamma,
                                     double  Trel,
                                     double  alphaM,
                                     double *fP,
                                     double *fM,
                                     int     nVar,
                                     int     iE,
                                     double *rADAN,
                                     double *irADAN,
                                     double *lADAN,
                                     double  rho,
                                     int     pathID,
                                     double  aRef,
                                     int     tlTypeVess);
  void
  computePhysicalFlux(double *q,
                      double  Gamma,
                      double  T,
                      double  rho,
                      double  alphaM,
                      double *f,
                      int     tlTypeVess);

  void
  computeVesselVolume(vessel *vessLoc);
  void
  computeFluctuationsADANvisco(double *q,
                               double *dq,
                               double  Gamma,
                               double  T,
                               double  rho,
                               double  alphaM,
                               double *fM,
                               double *fP,
                               int     tlTypeVess);


  double
  dzetada0ADANvisco(double a,
                    double a0,
                    double h0,
                    double ee,
                    double ec,
                    double ep0,
                    double epr,
                    double Gamma,
                    double psi,
                    int    tlTypeVess);
  double
  dzetadh0ADANvisco(double a,
                    double a0,
                    double h0,
                    double ee,
                    double ec,
                    double ep0,
                    double epr,
                    double Gamma,
                    double psi,
                    int    tlTypeVess);
  double
  dzetadeeADANvisco(double a,
                    double a0,
                    double h0,
                    double ee,
                    double ec,
                    double ep0,
                    double epr,
                    double Gamma,
                    double psi,
                    int    tlTypeVess);
  double
  dzetadecADANvisco(double a,
                    double a0,
                    double h0,
                    double ee,
                    double ec,
                    double ep0,
                    double epr,
                    double Gamma,
                    double psi,
                    int    tlTypeVess);


  double
  zetaADANLTS(double a,
              double a0,
              double h0,
              double ee,
              double ec,
              double ep0,
              double epr,
              double mu,
              double expmu,
              double logmu,
              double sqrta,
              double sqrta0,
              double phi,
              int    tlTypeVess);

  double
  celerityADANviscoLTS(double a,
                       double au,
                       double a0,
                       double h0,
                       double ee,
                       double ec,
                       double ep0,
                       double epr,
                       double alphaM,
                       double Gamma,
                       double Trel,
                       double psi,
                       double rho,
                       double mu,
                       double expmu,
                       double logmu,
                       double sqrta,
                       double sqrta0,
                       double phi);
  double
  celerityADANvisco0LTS(double a,
                        double au,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double alphaM,
                        double Gamma,
                        double Trel,
                        double psi,
                        double rho,
                        double mu,
                        double expmu,
                        double logmu,
                        double sqrta,
                        double sqrta0,
                        double phi);
  double
  dzetadaADANviscoLTS(double a,
                      double a0,
                      double h0,
                      double ee,
                      double ec,
                      double ep0,
                      double epr,
                      double Gamma,
                      double psi,
                      double mu,
                      double expmu,
                      double logmu,
                      double sqrta,
                      double sqrta0,
                      double phi);

  double
  dzetadpsiADANviscoLTS(double a,
                        double a0,
                        double h0,
                        double ee,
                        double ec,
                        double ep0,
                        double epr,
                        double Gamma,
                        double psi,
                        double mu,
                        double expmu,
                        double logmu,
                        double sqrta,
                        double sqrta0,
                        double phi);

  double
  dzetada0ADANviscoLTS(double a,
                       double a0,
                       double h0,
                       double ee,
                       double ec,
                       double ep0,
                       double epr,
                       double Gamma,
                       double psi,
                       double mu,
                       double expmu,
                       double logmu,
                       double sqrta,
                       double sqrta0,
                       double phi);
  double
  dzetadh0ADANviscoLTS(double a,
                       double a0,
                       double h0,
                       double ee,
                       double ec,
                       double ep0,
                       double epr,
                       double Gamma,
                       double psi,
                       double mu,
                       double expmu,
                       double logmu,
                       double sqrta,
                       double sqrta0,
                       double phi);
  double
  dzetadeeADANviscoLTS(double a,
                       double a0,
                       double h0,
                       double ee,
                       double ec,
                       double ep0,
                       double epr,
                       double Gamma,
                       double psi,
                       double mu,
                       double expmu,
                       double logmu,
                       double sqrta,
                       double sqrta0,
                       double phi);
  double
  dzetadecADANviscoLTS(double a,
                       double a0,
                       double h0,
                       double ee,
                       double ec,
                       double ep0,
                       double epr,
                       double Gamma,
                       double psi,
                       double mu,
                       double expmu,
                       double logmu,
                       double sqrta,
                       double sqrta0,
                       double phi);

  void
  pdeEigenvectorsADANviscoFull(double *q,
                               double  ep0,
                               double  epr,
                               double  Gamma,
                               double  T,
                               double  rho,
                               double  alphaM,
                               double *rADAN,
                               double *irADAN,
                               double *lADAN,
                               int     tlTypeVess);

  void
  pdeEigenvectorsADANvisco2by2(double  cel,
                               double *R,
                               double *L,
                               double *iR,
                               double *Q);
  double
  charVarRight(double *q,
               double *qS,
               double  ep0,
               double  epr,
               double  Gamma,
               double  T,
               double  rho,
               double  alphaM,
               int     tlTypeVess);

  double
  charVarLeft(double *q,
              double *qS,
              double  ep0,
              double  epr,
              double  Gamma,
              double  T,
              double  rho,
              double  alphaM,
              int     tlTypeVess);
};



#endif
