 
#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>

#ifdef WINDOWS

#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utilities.h"
#include "layers.h"
#include "assert.h"


typedef struct lstm_cache {
  double* c_old;
  double* h_old;
  double* c;
  double* h;
  double* X;
  double* hf;
  double* hi;
  double* ho;
  double* hc;
  double* tanh_c_cache;
} lstm_cache;


typedef struct lstm_rnn
{ 

  /** 
   * X
   * N
   * Y
   * S
    **/
  unsigned int X; 
  unsigned int N;  
  unsigned int Y;  
  unsigned int S;  

  // parameters 
  double* Wf;
  double* Wi;
  double* Wc;
  double* Wo;
  double* Wy;
  double* bf;
  double* bi;
  double* bc;
  double* bo;
  double* by;

  // output probability vector 
  double* probs; 

  // gradients gate
  double* dldh;
  double* dldho;
  double* dldhf;
  double* dldhi;
  double* dldhc;
  double* dldc;
  double* dldXi;
  double* dldXo;
  double* dldXf;
  double* dldXc;
  double* dldy;

  // gradients parameters
  double* dWf;
  double* dWi;
  double* dWc;
  double* dWo;
  double* dWy;
  double* dbf;
  double* dbi;
  double* dbc;
  double* dbo;
  double* dby;

  // cucc. gradients parameters
  double* cdWf;
  double* cdWi;
  double* cdWc;
  double* cdWo;
  double* cdWy;
  double* cdbf;
  double* cdbi;
  double* cdbc;
  double* cdbo;
  double* cdby;

  // gate and memory cell cache for time step
  lstm_cache** cache;
  double* c_prev;
  double* h_prev;


} lstm_rnn;


int lstm_init(int X, int N, int Y, lstm_rnn* lstm);

void lstm_free_model(lstm_rnn *lstm);

void lstm_forward(lstm_rnn* model, int *x ,lstm_cache** cache, Data *data);

void lstm_backforward(lstm_rnn* model , double *y, int l, lstm_cache** cache_in);

void lstm_cache_container_free(lstm_cache* cache_to_be_freed);

void lstm_free_model(lstm_rnn* lstm);

void lstm_zero_the_model(lstm_rnn *model);

void gradients_decend(lstm_rnn* model, lstm_rnn* gradients,float lr, int n);

void sum_gradients(lstm_rnn* model);

void mean_gradients(lstm_rnn* gradients, double d);
 
void copy_lstm(lstm_rnn* lstm, lstm_rnn* secondlstm);

void lstm_store(lstm_rnn* lstm, const char *filename);

void lstm_reset_gradient(lstm_rnn * model, int rc);
 
lstm_cache*  lstm_cache_init(int X, int N, int Y);


#endif
