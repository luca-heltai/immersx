// Filename :   FFT.H
// Author   :   Jonathan Valvano
// Date     :   Feb 12, 2008
// derived from Numerical Recipes in C, Cambridge University Press

/*  Example
// v(t) = 2 + cos(2 pi 100t+1.57) + 0.5cos(2 pi 200t), fs=1600 Hz, nn=16
double xyData[32]; // buffer has twice as many points as nn
double v[8];       // voltage in volts
double phase[8];   // phase in radians
double freq[16];   // frequency in Hz
double dB[8];      // magnitude in dB full scale
void TestFFT(void){
  xyData[0]  = 2.501; xyData[1]  = 0.0;
  xyData[2]  = 1.972; xyData[3]  = 0.0;
  xyData[4]  = 1.293; xyData[5]  = 0.0;
  xyData[6]  = 0.723; xyData[7]  = 0.0;
  xyData[8]  = 0.500; xyData[9]  = 0.0;
  xyData[10] = 0.722; xyData[11] = 0.0;
  xyData[12] = 1.292; xyData[13] = 0.0;
  xyData[14] = 1.970; xyData[15] = 0.0;
  xyData[16] = 2.499; xyData[17] = 0.0;
  xyData[18] = 2.736; xyData[19] = 0.0;
  xyData[20] = 2.707; xyData[21] = 0.0;
  xyData[22] = 2.570; xyData[23] = 0.0;
  xyData[24] = 2.500; xyData[25] = 0.0;
  xyData[26] = 2.571; xyData[27] = 0.0;
  xyData[28] = 2.708; xyData[29] = 0.0;
  xyData[30] = 2.737; xyData[31] = 0.0;
  fft(xyData,16);
// approximate results
// xyData[0]  will be 32, xyData[1]  will be 0.0;
// xyData[2]  will be 0,  xyData[3]  will be 8.0;
// xyData[4]  will be 4,  xyData[5]  will be 0.0;
// xyData[6]  will be 0,  xyData[7]  will be 0.0;
// xyData[8]  will be 0,  xyData[9]  will be 0.0;
// xyData[10] will be 0,  xyData[11] will be 0.0;
// xyData[12] will be 0,  xyData[13] will be 0.0;
// xyData[14] will be 0,  xyData[15] will be 0.0;
  for(int k=0; k<8; k++){
  v[k]     = fftMagnitude(xyData,16,k);
  dB[k]    = fftMagdB(xyData,16,k,2.0); // largest component is 2V
  phase[k] = fftPhase(xyData,16,k);
  }
// very approximate results
// v[0] will be 2,   dB[0] will be    0,   phase[0] will be 0.0;
// v[1] will be 1,   dB[1] will be   -6,   phase[1] will be 1.57;
// v[2] will be 0.5, dB[2] will be  -12,   phase[2] will be 0.0;
// v[3] will be 0,   dB[3] will be -200,   phase[3] will be 0.0;
// v[4] will be 0,   dB[4] will be -200,   phase[4] will be 0.0;
// v[5] will be 0,   dB[5] will be -200,   phase[5] will be 0.0;
// v[6] will be 0,   dB[6] will be -200,   phase[6] will be 0.0;
// v[7] will be 0,   dB[7] will be -200,   phase[7] will be 0.0;

  for( k=0; k<16; k++){
  freq[k]  = fftFrequency(16,k,1600);
  }
// expected results
// freq = 0,100,200,300,400,500,600,700,800,-700,-600,-500,-400,-300,-200,-100
}
*/

// Input: nn is the number of points in the data and in the FFT,
//           nn must be a power of 2
// Input: data is sampled voltage v(0),0,v(1),0,v(2),...v(nn-1),0 versus time
// Output: data is complex FFT Re[V(0)],Im[V(0)], Re[V(1)],Im[V(1)],...
// data is an array of 2*nn elements
void
fft(double data[], unsigned long nn);

//-----------------------------------------------------------
// Calculates the FFT magnitude at a given frequency index.
// Input: data is complex FFT Re[V(0)],Im[V(0)], Re[V(1)],Im[V(1)],...
// Input: nn is the number of points in the data and in the FFT,
//           nn must be a power of 2
// Input: k is frequency index 0 to nn/2-1
//        E.g., if nn=16384, then k can be 0 to 8191
// Output: Magnitude in volts at this frequency (volts)
// data is an array of 2*nn elements
// returns 0 if k >= nn/2
double
fftMagnitude(double data[], unsigned long nn, unsigned long k);

//-----------------------------------------------------------
// Calculates the FFT magnitude in db full scale at a given frequency index.
// Input: data is complex FFT Re[V(0)],Im[V(0)], Re[V(1)],Im[V(1)],...
// Input: nn is the number of points in the data and in the FFT,
//           nn must be a power of 2
// Input: k is frequency index 0 to nn/2-1
//        E.g., if nn=16384, then k can be 0 to 8191
// Input: fullScale is the largest possible component in volts
// Output: Magnitude in db full scale at this frequency
// data is an array of 2*nn elements
// returns -200 if k >= nn/2
double
fftMagdB(double data[], unsigned long nn, unsigned long k, double fullScale);

//-----------------------------------------------------------
// Calculates the FFT phase at a given frequency index.
// Input: data is complex FFT Re[V(0)],Im[V(0)], Re[V(1)],Im[V(1)],...
// Input: nn is the number of points in the data and in the FFT,
//           nn must be a power of 2
// Input: k is frequency index 0 to nn/2-1
//        E.g., if nn=16384, then k can be 0 to 8191
// Output: Phase at this frequency
// data is an array of 2*nn elements
// returns 0 if k >= nn/2
double
fftPhase(double data[], unsigned long nn, unsigned long k);

//-----------------------------------------------------------
// Calculates equivalent frequency in Hz at a given frequency index.
// Input: fs is sampling rate in Hz
// Input: nn is the number of points in the data and in the FFT,
//           nn must be a power of 2
// Input: k is frequency index 0 to nn-1
//        E.g., if nn=16384, then k can be 0 to 16383
// Output: Equivalent frequency in Hz
// returns 0 if k >= nn
double
fftFrequency(unsigned long nn, unsigned long k, double fs);
