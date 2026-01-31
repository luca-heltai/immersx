// Linear algebra routines
#include <string.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
//#include <cblas.h>
//#include "clapack.h"
//#include "f2c.h"


#ifndef __LINALG__
#define __LINALG__


extern "C" {
  // LU decomoposition of a general matrix
  void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
}
extern "C" {
  void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}
extern "C" {
  void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* info);
}



void inverse(double* A, int N, double* iA,int* infoExt);
void invertLTS(double* A, int N,int *infoExt);
/*Solve a linear system Ax=b by using LAPACK DGESV subroutine, wrapped to c*/
int dgesv(int N, int NRHS, double *A, int LDA, int *IPIV, double *B, int LDB, int INFO);

void transposeM(double *A,double *AT, int n);
void matrixInv(double *A, int n, double *invA);
void matrixInvNew(double *A, int n, double *invA, int *infoExt);
void matrixInvLTS(double **A, 
		  int n, 
		  int *infoExt);
void matmul(double *A,double *B,double *C, int m, int n, int l);
void matmulLTS(double **A,double **B,double **C, int m, int n, int l);
void matpower(double *A,double *C,int n);
void matsum(double *A,double *B,double *C, int m, int n, double sum);
void matsumLTS(double **A,
	       double **B,
	       double **C, 
	       int m, int n, double sum);
void matsumSame(double *A,double *B, int m, int n, double sum);
void matSameSum(double *A,double *B, int m, int n, double sum);
void vecsumSame(double *A,double *B, int m, double sum);
double vecnorm(double *v, int n, double order);
double vecnormSTD(const std::vector<double> &v, int n, double order);
double matnorm(double *A, int n, int m, double order);
void matabs(double *A, int m, int n);
void matsca(double *A,double sca, int n);
void matscaNewma(double *A,double *B,double sca, int n);
void matscaNS(double *A,double sca, int m, int n);
void vecsca(double *A,double sca, int n);
void solveLinearSystem(double *A, int n, double *b);
void matmulvecLTS(double **A,double *B,double *C, int m, int n);

void matmulvecRot(const std::vector<double> &v, std::vector<double> &v_rot, double alpha, double beta, double gamma);
double dotprod(const std::vector<double> &v1, const std::vector<double> &v2, int n);

#endif
