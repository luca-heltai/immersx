#include <fstream>
#include <vector>
#include <math.h>
#include "linalg.h"
#include "cblas.h"
#include "GetPot.h"

using namespace std;

/**
 * @brief Autoregulation model to be applied in any vascular region of interest
 * @author Caterina Dalmaso
 * @note equations are analogous to those presented for cerebral autoregulation in 
    Toro, E. F., Celant, M., Zhang, Q., Contarino, C., Agarwal, N., Linninger, A., & Müller, L. O. (2022). Cerebrospinal fluid dynamics coupled to the global circulation in holistic setting: mathematical models, numerical methods and applications. International Journal for Numerical Methods in Biomedical Engineering, 38(1), e3532.
    
    No transport is required, equations are expressed in terms of average flows
 */

class autoregulationUrsino
{   
    public:

    // time
    double time;
    double dt;
    double dtSample;
    double tSampleIni;
    double tSampleEnd;
    double timeStop;
    double tIni;
    int iT;
    int verbose;    
    int iterationsPerCycle;

    /// @brief cardiac cycle
    double T0; 
    /// @brief  respiratory cycle
    double T;
    /// @brief number of vessels
    int NV;
    /// @brief conversion
    double mmHgDyncm2;

    /// @brief vessel file as in model: remember to delete the pointer


    int OpenClosed;

    // parameters

    /// @brief // total static gain of autoregulation, which is den used to determine the static gain for the ith cerebral terminal artery according to flow distribution
    vector <double> Gtot; 
    /// @brief  time constant of the first-order low-pass dynamics
    double tau;
    /// @brief upper and lower saturation levels of the digmoidal curve
    double sat1;
    double sat2;
    double kmult;
    /// @brief cbfBase in the original, check what it is
    vector <double> GmulBase;

    /// @brief number of regions under the effects of autoregulation
    int nRegions;
    /// @brief IDs of regions under the effects of autoregulation.
    vector <int> regionID; 
    
    /// @brief number of vessels per region that are affected by autoregulation (terminals)
    vector <int> nTerminals; 
    int terminalFullCount;
    /// @brief ids of the vessels that, in each region, are affected by autoregulation: external vector has nRegions components, while each internal one has nTerminals[i] components
    vector <vector <int>> idxTerminals;
    vector <int> idxTerminalsFull;
    /// @brief same as before, but with global ids
    vector <vector <int>> vesselsGlobalID; // id of vessels per region OK
    
    /// @brief reference flow at each considered terminal artery in each considered region. external vector has nRegions components, while each internal one has nTerminals[i] components
    vector <vector <double>> QavgRef; 
    /// @brief reference Volume
    vector <vector <double>> VavgRef;
    /// @brief reference pressure
    vector <vector <double>> PavgRef; 
    /// @brief  reference compliance = central value of the sigmoidal curve
    vector <vector <double>> Cbase;
    /// @brief reference resistance under baseline conditions of the arteriolar-capillaries compartment
    vector <vector <double>> Rbase;
    /// @brief gain for each terminal
    vector <vector <double>> G;
    
    // state variables
    vector <vector <double>> x;  // one component for each considered terminal
    vector <vector <double>> dxdt;

    // algebraic variables, each component correspond to one terminal vessel in the considered region

    /// @brief constant parameter, inversely proportional to the central slope of the sigmoidal curve
    vector <vector <double>> k; 
    /// @brief amplitude of the sigmoidal curve
    vector <vector <double>> DeltaC;
    /// @brief Compliance that is modified by the autoregulation RIGHT NOW WE ARE CONSIDERING ONLY ARTERIES
    vector <vector <double>> C;
    /// @brief Resistance that is modified by the autoregulation RIGHT NOW WE ARE CONSIDERING ONLY ARTERIES
    vector <vector <double>> R; 
    /// @brief Time averaged flow over the period [t-T, t]
    vector <vector <double>> Qavg;  // CFR LINES 8660 ONWARD IN MODEL1D SOLVETIMESTEP
    /// @brief Time averaged volume
    vector <vector <double>> Vavg;  
    /// @brief Time averaged pressure
    vector <vector <double>> Pavg;
    /// @brief Variables required to compute time averages
    vector <vector <vector<double>>> statesMeansQ; // innermost is time
    vector <vector <vector<double>>> statesMeansP;
    vector <vector <vector<double>>> statesMeansV;

    vector <vector <string>> nameAutoregulation;

    // methods
    void init(string ifile, string outDir, string testcase);
    
    /// @brief parametrization of the model
    void getAutoregulationParameters(string _file);
    void printAutoregulationParameters();

    // void getAffectedVessels();    

    void saveInitialCond();   
    
    /// @brief averages required to update the state variable
    //  void getAverages();    

    /// @brief update the state and algebraic variables
    void solveAutoregulation();
    // void getAlgebraicVars(); // k, DeltaC    
    // void updateCompliance();
    // void updateResistance();


    /// @brief print quantities of interest

    void saveState();
    void readState();

    void getState(vector <vector <double>>& state);
    void setState(vector <vector <double>>& state);

    int stateSize;
    string outDir;
    string stateDir;
};