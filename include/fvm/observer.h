#include <vector>

#include "fft.h"

class Observer
{
public:
  Observer(){};
  ~Observer(){};

  // variables
  int nObservations; ///@brief number of observations
  std::vector<std::vector<double>> observations; ///@brief observation container

  // methods
  void
  init(int nObs);
  //  double observe(int im, std::string measureClass);
  double
  observe(int im);
  double
  observe(int                  im,
          std::vector<int>     measureHarmonics,
          std::vector<double> &harmonics,
          int                  usePhases);
  void
  forwardDFT(const std::vector<double> &s,
             const int                  Nharm,
             std::vector<double>       &magnitude,
             std::vector<double>       &phase);
  void
  saveObs(int im, double obs);
  void
  clearObs(int im);
};
