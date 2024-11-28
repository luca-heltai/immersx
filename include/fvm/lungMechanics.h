#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

/**
 * @brief Implementation of lung mechanics model by Albanese et al (2015). 
 * 
 * @author Caterina Dalmaso
 * 
 * @note  Lung mechanics model proposed in: An integrated mathematical model of the human cardiopulmonary system: model development
 * (A Albanese, L Cheng, M Ursino & N. W. Chbat, American Journal of Physiology-Heart and Circulatory Physiology 2016 310, H899-H921)
 * 
 * @note The following denote:
 * - ao -> airway opening
 * - ml -> mouth to larynx
 * - l  -> larynx
 * - lt -> larynx to trachea
 * - tr -> trachea
 * - tb -> trachea to bronchea
 * - b  -> bronchea
 * - bA -> bronchea to alveoli
 * - A  -> alveoli
 * - cw -> chest wall
 * 
 * Pmus is muscolar pressure (influenced by control) 
 * NB: UNITS: cm/s/g
 */
class lungMechAlbanese
{
  public:
    // time
    /// Time in the lungs
    double time;
    /// dt used for the update
    double dt_lungs;
    /// dt used for sampling
    double dtSample;
    /// time at which sampling starts
    double tSampleIni;
    /// time at which sampling ends
    double tSampleEnd;
    /// time at which simulation starts
    double tIni;
    /// number of iterations
    int iT;  
    /// Integer determining whether outputs are printed to teminal/log file
    int verbose;

    /**
     *  @brief Vector with 5 components that contains volumes
     * - double vl: volume in the larynx
     * - double vtr: volume in the trachea
     * - double vb: volume in the bronchea
     * - double vA: volume in the alveoli
     * - double vpl: volume in the pleural cavity (associated to cw compliance)
     */
    vector <double> vol = vector <double> (6);


    /**
     * @brief Vector with 5 components that contains time derivative of volumes
     * - double dvldt: dt volume in the larynx
     * - double dvtrdt: dt volume in the traches
     * - double dvbdt: dt volume in the bronchea
     * - double dvAdt: dt volume in the alveoli
     * - double dvpldt: dt volume in the pleural cavity (associated to cw compliance)
     */
    vector <double> dvdt = vector <double> (6);
    

    // parameters
    /**
     * @brief Vector containing the compliances of the considered compartments
     * - double cl: larynx compliance
     * - double ctr: trachea compliance
     * - double cb: bronchea compliance
     * - double cA: alveoli compliance
     * - double ccw;: chest wall compliance
     */
    vector <double> C = vector <double> (6); 
    
    /**
     * @brief Vector containing the unstressed volumes of the considered compartments
     * - double vul: larynx unstressed volume
     * - double vutr: trachea unstressed volume
     * - double vub: bronchea unstressed volume
     * - double vuA: alveoli unstressed volume
     */
    vector <double> Vu = vector <double> (4);

    /**
     * @brief Vector containing the resistances of the considered compartments
     * - double rml: resistance mouth to larynx
     * - double rlt: resistance larynx to trachea
     * - double rtb: resistance trachea to bronchea
     * - double rbA: resistance bronchea to alveoli
     */
    vector <double> R = vector <double> (4);

    /**
     * @brief Respiratory parameters
     * - double rr: respiratory rate
     * - double ieratio: inspiratory-expiratory time ratio 
     * - double frc: functional residual capacity
     * - double pplee: pleural pressure value at end expiration
     * - double pmusmin: minimum end inspiratory pressure
     * - double tauCoeff: time constant coefficient
     * - double pao: airway opening pressure
     * - double pvent: ventilator pressure
     * - double patm: atmospheric pressure
     * - double IAPee: end-expiratory IAP
    */
    vector <double> resp = vector <double> (10);
    
    // set time constants
    /// respiratory cycle duration
    double T;
    /// expiratory fraction              
    double te;
    /// inspiratory fraction                
    double ti;
    /// time constant of the exponential expiratory profile                
    double tau;
    /// local time variable that denotes the "position" within the respiratory cycle and is used when defining pmus               
    double timeloc;      

    // getPmus
    /**
     * @brief Muscular pressure that drives the respiratory mechanism
     */
    double Pmus;              // muscolar pressure

    // getAlgebraicRelations
    /// total airflow
    double Vdot;   
    /// alveolar airflow           
    double VAdot;
    /// dead space volume             
    double vd;    
    /// total volume            
    double v;                 

    /**
     * @brief Vector with 6 components that contains pressures
     * - double pl: larynx pressure
     * - double ptr: trachea pressure
     * - double pb: bronchea pressure
     * - double pA: alveolar pressure
     * - double ppl1: (1/ccw * vpl) + pplee + pmus
     * - double pabd: abdominal pressure
     */
    vector <double> P = vector <double> (6);


    // numerical method for solution
    /**
     * @brief Numerical method to be used for the solution
     * method = 0 for explicit Euler
     * method = 1 for RK 4
     */
    int method; // 0 if Euler, 1 if rk4

    // vector <double> state = vector <double> (0);
    // output
    /// Volumes output file
    ofstream sampleLungVol;
    /// Pressures output file
    ofstream sampleLungPr;
    /// Flows output file
    ofstream sampleLungFlow;
    string nameLung;

    // methods
    /**
     *  Read lung mechanics parameters from file (passed as input) through the readLungMechParameters function and define the state variable vector through the getState function
     */
    void init(string ifile, string outDir, string testcase);


    /**
     * Default parameter values from Albanese et al (2015). If a variable is not found under [category], then a default value is assigned. Time variables are assigned in the main.
     * Valid categories currently include methods, compliances, unstressed volumes, resistances, respiratory params. Initial conditions are computed assuming that at time t=0 (end-exhalation time), all of the pressures in the lungs equilibrate to Patm = 0 cmH2O, whereas Ppl has a subatmospheric value of -5 cmH2O.
     * @param ifile Input file containing parameters required to run simulations
     */
    void readLungMechParameters(string _file);

    /**
     * @brief Save state variables initial conditions
     */
    void saveInitialCond();

    /**
     * @brief Prints parameters read through readLungMechParameters
     */
    void printLungMechParameters();
    
    /**
     * @brief  Computes the duration of the respiratory cycle, the inspiratory and expiratory fractions and a local time that spans the respiratory cycle
     */
    void getTimeConstants();

    /**
     * @brief  Computes the muscular pressure that drives the respiratory effort. It can be parametrized in terms of respiratory rate and its minimum value.
     */
    void getPmus();

    /**
     * @brief Computes pressures, airflows, total and dead space volumes as functions of volumes.
     */
    void getAlgebraicRelations();

    /**
     * @brief  getTimeDerivative: definition of the ODE system to be solved: requires algebraic relations for pressures, so that pressures and volumes are considered at the same timestep.
     */
    void getTimeDerivative();

    /**
     * @brief  Definition the explicit Euler evolution step 
     */
    void Euler_step();

    /**
     * Definition the RK4 evolution step 
     */
    void RK4_step();

    /**
     * @brief Updates the state of lung mech model by Albanese et al. (2015) from t to t+dt using either explicit Euler (method == 0) or RK4 (method == 1) and saves to file the outputs of interest.
     */
    void updateState();

    /**
     * @brief Save state variables and algebric relations to files each dtSample
     */
    void output();

    // state
    /**
     * @brief Saves the complete set of variables and parameters to file, so that a restart function can be employed to interrupt and resume the simulation 
     */
    void saveState();

    /**
     * @brief Reads the complete set of variables and parameters saved to file through the saveState function
     */
    void readState();

    /**
     * @brief Appends volumes to the state vector
     */
    void getState(vector<double>& state);

    /**
     * @brief Assigns to vol the corresponding state component
     */
    void setState(vector<double>& state);

    /// state size
    int stateSize;

    /// output directory
    string outDir;

    /// state directory
    string stateDir;

}; 