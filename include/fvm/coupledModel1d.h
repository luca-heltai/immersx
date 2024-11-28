// include necessary packages
//#include <functional>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "GetPot.h"
#include "model1d.h"

#ifndef __COUPLEDMODEL1D_H
#define __COUPLEDMODEL1D_H 1

using namespace std;


class CoupledModel1d : public Model1d
{
 public:
  CoupledModel1d(): Model1d(){};
  ~CoupledModel1d(){};

  //
  int totalNumberCells;
  void compute_totalNumberCells();

  // output 
  vector<ofstream> areafile;
  vector<ofstream> presExfile;
  vector<ofstream> presfile;
  vector<ofstream> qfile;
  string outFile;

  void openFilesPlot();
  void closeFilesPlot();
  void writePressure();
  void writeEXTPressure();
  void writeArea();
  void writeFlow();

  // input
  string radiusMidFile; // directory in which the RadiusEnd-RadiusIni file is saved
  string displacementHdf5; // directory in which the RadiusEnd-RadiusIni HDF5file is saved
  string pressure3DFile;//read from 3dPressure file
  string pressure3DFile_hdf5; // read from 3D pressure file in hdf5 format
  string dataset_hdf5; // name of dataset in the pressure3DFile_hdf5 

  int coupling = 0; // if 1 write file with RadiusEnd-RadiusIni for 1d-3d coupling
  
  // external pressure
  double **ExternalPressure;
  void ReadPressure();
  void Read3DPressureFile();
  // void Read3DPressureFile_hdf5();
  //"fake" 3D
  double k; //const k, st P_ext=k*(AreaFinal-AreaIni)
  void Solve3Dk() {};
  void solvePseudo3D();
  void setAreaIni(); // set // area at mid point vessel, before solveTimeStep
  double omega;
  double pBase; // external pressure at the beginning, if not specified otherwise it's 0
  std::vector<double> new_displacement;
  std::vector<double> get_new_displacement_for_coupling();

  // output
   // void plotMid_AREA(int i ); // unused


  void compute_new_displacement_for_coupling(unsigned int);

    void printCouplingParameters();
    void readCouplingParameters(string);
    void init(string ifile);
    void saveFullState(int i) override;

};



#endif
