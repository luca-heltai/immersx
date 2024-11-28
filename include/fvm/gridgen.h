// All data needed for each junction
struct juncData
{
  ofstream sample;
  int iT;
  double time;
  double dt;
  double *dtJ;
  double *vessIndex; // index for vessel in the junction
  double *signs; // for each vessel in the junction
  int *ind;
  double *junctions;
  int **evalST;
  double *junctionsHO;
  double *junctionsInt;
  double *junctionsLTS;
  double *junctionsLTSx;
  double *junctionsLTSxx;
  double *aJ;				// auxiliar array of areas and other 
  double *auJ;				// parameters for junctions
  double *psiJ;
  double *aJx;				// auxiliar array of areas and other 
  double *auJx;				// parameters for junctions
  double *psiJx;
  double *aJint;				// auxiliar array of areas and other 
  double *auJint;				// parameters for junctions
  double *peJ;				// parameters for junctions
  double *a0J;
  double *mJ;
  double *nJ;
  double *kJ;
  double *h0J;
  double *eeJ;
  double *ecJ;
  int *vessTlTypeJ;
  double *thetaJ;
  double aa0;
  double aa1;
  double aa0_initial;
  double aa1_initial;
  int juncsNodes;
  double *ep0;
  double *epr;
  double *alphaMJ;
  double a0s;
  double c0s;
  double k0s;
  double h0s;
  double ees;
  double *muJ;
  double *gJ;
  double *aN;
  double *auN;
  double *dx;
  double *p0;
  double v0;
  double vC;
  double *a0sJ;
  double *c0sJ;
  double *k0sJ;
  double *GammabJ;
  double *GammaJ;
  double *TJ;
  double T0s;
  double *psibJ;
  double *epsilonbJ;
  double Gamma0s;
  double epsilon0s;
  double L0s;
};



// All data needed for each valve
struct valve
{
  double valveCoef[13];
  double valveState[7];
};

// All data needed for each starling resistor
struct sr
{
  double srCoef[9];
  double srState[2];
};


