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

  std::vector<double> new_displacement;
  std::vector<double> get_new_displacement_for_coupling();
  void compute_new_displacement_for_coupling(unsigned int);
  void solvePseudo3D(unsigned int);

};



#endif
