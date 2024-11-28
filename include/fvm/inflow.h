#include <vector>
#include<iostream>
#include <stdlib.h>
class Inflow
{
 public:
  Inflow(){};
  ~Inflow(){};
  
  ///@brief scaling parameter for relative duration of ejection period (systole)
  double Ts;
  ///@brief scaling parameter for amplitude of systolic flow peak
  double As;
  ///@brief scaling parameter for relative duration of backflow period
  double Tb;
  ///@brief scaling parameter for amplitude of backflow peak
  double Ab;

  ///@brief cardiac cycle dutation
  double T;
  ///@brief average flow
  double qAv;
  ///@brief average flow
  double qAvInit;

  ///@brief time instants for systole
  std::vector<double> timeSystole;

  ///@brief time instants for backflow
  std::vector<double> timeBackflow;

  ///@brief flow for systole
  std::vector<double> flowSystole;

  ///@brief flow for backflow
  std::vector<double> flowBackflow;

  ///@bried time
  std::vector<double> time;
  
  ///@brief flow
  std::vector<double> flow;
  // methods
  void init(int inflowType, double cardiacCycleDuration, double tSys,double tDias, double ampSys, double ampDias);
  void initMynard();
  double getFlow(double tModel);
  void setCurve();
  void setCurve(double qAverage);
  double getAverage();
};



  
