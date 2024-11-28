#include <fstream>
#include <vector>
using namespace std;
class cuffTestOccl
{
 public:

  // parameters
  int idxLeftGlobal,idxRightGlobal; // left vessel index, right vessel index
  int idxLeft,idxRight; // left vessel index, right vessel index
  int iCellLeft,iCellRight; // left vessel cell index, right vessel cell index
  double signLeft,signRight; // account for orientation of 1D vessels

  int cuffID;
}; 
