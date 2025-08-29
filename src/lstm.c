#include "lstm.h"


 
int lstm_init(int X, int N, int Y, lstm_rnn* lstm)
{

  int S = X + N;
  
  lstm->X = X;  
  lstm->N = N;  
  lstm->S = S; 
  lstm->Y = Y; 

  lstm->probs = get_zero_vector(Y);

  lstm->Wf = get_random_vector(N * S, S);
  lstm->Wi = get_random_vector(N * S, S);
  lstm->Wc = get_random_vector(N * S, S);
  lstm->Wo = get_random_vector(N * S, S);
  lstm->Wy = get_random_vector(Y * N, N);

  lstm->dWf = get_zero_vector(N * S);
  lstm->dWi = get_zero_vector(N * S);
  lstm->dWc = get_zero_vector(N * S);
  lstm->dWo = get_zero_vector(N * S);
  lstm->dWy = get_zero_vector(Y * N);

  lstm->cdWf = get_zero_vector(N * S);
  lstm->cdWi = get_zero_vector(N * S);
  lstm->cdWc = get_zero_vector(N * S);
  lstm->cdWo = get_zero_vector(N * S);
  lstm->cdWy = get_zero_vector(Y * N);

  lstm->bf = get_zero_vector(N);
  lstm->bi = get_zero_vector(N);
  lstm->bc = get_zero_vector(N);
  lstm->bo = get_zero_vector(N);
  lstm->by = get_zero_vector(Y);

  lstm->dbf = get_zero_vector(N);
  lstm->dbi = get_zero_vector(N);
  lstm->dbc = get_zero_vector(N);
  lstm->dbo = get_zero_vector(N);
  lstm->dby = get_zero_vector(Y);

  lstm->cdbf = get_zero_vector(N);
  lstm->cdbi = get_zero_vector(N);
  lstm->cdbc = get_zero_vector(N);
  lstm->cdbo = get_zero_vector(N);
  lstm->cdby = get_zero_vector(Y);

  lstm->dldhf = get_zero_vector(N);
  lstm->dldhi = get_zero_vector(N);
  lstm->dldhc = get_zero_vector(N);
  lstm->dldho = get_zero_vector(N);
  lstm->dldc  = get_zero_vector(N);
  lstm->dldh  = get_zero_vector(N);
  lstm->dldy  = get_zero_vector(Y);

  lstm->dldXc = get_zero_vector(S);
  lstm->dldXo = get_zero_vector(S);
  lstm->dldXi = get_zero_vector(S);
  lstm->dldXf = get_zero_vector(S);



  lstm->cache = malloc((200)*sizeof(lstm_cache));
  for (int t = 0; t < 200; t++)
  {
    lstm->cache[t] = lstm_cache_init(X, N, Y);
  }  

  lstm->h_prev = get_zero_vector(N);
  lstm->c_prev = get_zero_vector(N);
 
  return 0;
}

void lstm_free_model(lstm_rnn* lstm)
{

  free_vector(&(lstm)->probs);
  free_vector(&lstm->Wf);
  free_vector(&lstm->Wi);
  free_vector(&lstm->Wc);
  free_vector(&lstm->Wo);
  free_vector(&lstm->Wy);

  free_vector(&lstm->dWf);
  free_vector(&lstm->dWi);
  free_vector(&lstm->dWc);
  free_vector(&lstm->dWo);
  free_vector(&lstm->dWy);

  free_vector(&lstm->cdWf);
  free_vector(&lstm->cdWi);
  free_vector(&lstm->cdWc);
  free_vector(&lstm->cdWo);
  free_vector(&lstm->cdWy);

  free_vector(&lstm->bf);
  free_vector(&lstm->bi);
  free_vector(&lstm->bc);
  free_vector(&lstm->bo);
  free_vector(&lstm->by);
  free_vector(&lstm->h_prev);
  free_vector(&lstm->c_prev);

  free_vector(&lstm->dldhf);
  free_vector(&lstm->dldhi);
  free_vector(&lstm->dldhc);
  free_vector(&lstm->dldho);
  free_vector(&lstm->dldc);
  free_vector(&lstm->dldh);

  free_vector(&lstm->dldXc);
  free_vector(&lstm->dldXo);
  free_vector(&lstm->dldXi);
  free_vector(&lstm->dldXf);
 

  free(lstm);
}


void lstm_forward(lstm_rnn* model, int *x, lstm_cache** cache, Data *data)
{

  double *h_prev = model->h_prev;
  double *c_prev = model->c_prev;
  double  *X_one_hot;
  int N = model->N;
  int S = model->S;
  int n = n = (data->xcol - 1);

  double *tmp;
  if ( init_zero_vector(&tmp, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }
 
  for (int t = 0; t <= (data->xcol-1) ; t++)
  {


    int i = 0 ;
    X_one_hot = cache[t]->X;
    while ( i < S ) 
    {
      if ( i < N ) {
        X_one_hot[i] = h_prev[i];
      } else  {
        X_one_hot[i] = data->embedding[x[t]][i-N];
      }
      ++i;
    }

    /*
     *  Fully connected + sigmoid layers 
     *  ft = sigmoid(Wf*Xt + bf)
     */
    fully_connected_forward(cache[t]->hf, model->Wf, X_one_hot, model->bf, N, S); 
    sigmoid_forward(cache[t]->hf, cache[t]->hf, N);
    
    // it = sigmoid(Wi*Xt + bi)
    fully_connected_forward(cache[t]->hi, model->Wi, X_one_hot, model->bi, N, S);
    sigmoid_forward(cache[t]->hi, cache[t]->hi, N);

    // ot = sigmoid(Wo*Xt + bo)
    fully_connected_forward(cache[t]->ho, model->Wo, X_one_hot, model->bo, N, S);
    sigmoid_forward(cache[t]->ho, cache[t]->ho, N);

    // çt = tanh(Wc*Xt + bc)
    fully_connected_forward(cache[t]->hc, model->Wc, X_one_hot, model->bc, N, S);
    tanh_forward(cache[t]->hc, cache[t]->hc, N);

    // c = hf*c_prev + hi*hc
    copy_vector(cache[t]->c, cache[t]->hf, N);
    vectors_multiply(cache[t]->c, c_prev, N);
    copy_vector(tmp, cache[t]->hi, N);
    vectors_multiply(tmp, cache[t]->hc, N);
    vectors_add(cache[t]->c, tmp, N);

    // h = ho * tanh_c_cache
    tanh_forward(cache[t]->tanh_c_cache, cache[t]->c, N);
    copy_vector(cache[t]->h, cache[t]->ho, N);
    vectors_multiply(cache[t]->h, cache[t]->tanh_c_cache, N);

    copy_vector(cache[t]->c_old, c_prev, N);
    copy_vector(cache[t]->h_old, h_prev, N);
    copy_vector(c_prev, cache[t]->c, N);
    copy_vector(h_prev, cache[t]->h, N);

  }

  /*
   *  we are in many to one LSTM, so we use the last cache to compute the output
   *  probs = softmax (Wy*h + by)
   */
  fully_connected_forward(model->probs, model->Wy, cache[n]->h, model->by, model->Y, model->N);
  softmax_layers_forward(model->probs, model->probs, model->Y);
  
  free_vector(&tmp);

}


void lstm_backforward(lstm_rnn* model, double *y, int n, lstm_cache** cache)
{
 
  lstm_cache* cache_in = NULL;
  double *dldh, *dldy, *dldho, *dldhf, *dldhi, *dldhc, *dldc;
  int N, Y, S;
  
  N = model->N;
  Y = model->Y;
  S = model->S;

  double *bias = malloc(N*sizeof(double));
  double *weigth = malloc((N*S)*sizeof(double));

  // model cache
  dldh  = model->dldh;
  dldc  = model->dldc;
  dldho = model->dldho;
  dldhi = model->dldhi;
  dldhf = model->dldhf;
  dldhc = model->dldhc;
  dldy  = model->dldy;

  copy_vector(dldy, model->probs, model->Y);
  vectors_substract(dldy, y, model->Y);
  fully_connected_backward(dldy, model->Wy, cache[n]->h , model->dWy, dldh, model->dby, Y, N);

  copy_vector(dldc, dldh, N);
  vectors_multiply(dldc, cache[n]->ho , N);
  tanh_backward(dldc, cache[n]->tanh_c_cache, dldc, N);

  for (int t = n ; t >= 0; t--)
  {
    cache_in = cache[t];

    copy_vector(dldho, dldh, N);
    vectors_multiply(dldho, cache_in->tanh_c_cache, N);
    sigmoid_backward(dldho, cache_in->ho, dldho, N);

    copy_vector(dldhf, dldc, N);
    vectors_multiply(dldhf, cache_in->c_old, N);
    sigmoid_backward(dldhf, cache_in->hf, dldhf, N);

    copy_vector(dldhi, cache_in->hc, N);
    vectors_multiply(dldhi, dldc, N);
    sigmoid_backward(dldhi, cache_in->hi, dldhi, N);

    copy_vector(dldhc, cache_in->hi, N);
    vectors_multiply(dldhc, dldc, N);
    tanh_backward(dldhc, cache_in->hc, dldhc, N);

    fully_connected_backward(dldhi, model->Wi, cache_in->X, weigth, model->dldXi, bias, N, S);
    vectors_add(model->dWi, weigth, N*S);
    vectors_add(model->dbi, bias, N);

    fully_connected_backward(dldhc, model->Wc, cache_in->X, weigth, model->dldXc, bias, N, S);
    vectors_add(model->dWc, weigth, N*S);
    vectors_add(model->dbc, bias, N);

    fully_connected_backward(dldho, model->Wo, cache_in->X, weigth, model->dldXo, bias, N, S);
    vectors_add(model->dWo, weigth, N*S);
    vectors_add(model->dbo, bias, N);

    fully_connected_backward(dldhf, model->Wf, cache_in->X, weigth, model->dldXf, bias, N, S);
    vectors_add(model->dWf, weigth, N*S);
    vectors_add(model->dbf, bias, N);

    // dldXi will work as a temporary substitute for dldX (where we get extract dh_next from!)
    vectors_add(model->dldXi, model->dldXc, S);
    vectors_add(model->dldXi, model->dldXo, S);
    vectors_add(model->dldXi, model->dldXf, S);

    copy_vector(dldh, model->dldXi, N);
    vectors_multiply(dldc, cache_in->hf, N);

  }
   
  free_vector(&bias);
  free_vector(&weigth);

}


lstm_cache*  lstm_cache_init(int X, int N, int Y)
{
  int S = N + X;

  lstm_cache* cache = e_calloc(1, sizeof(lstm_cache));

  cache->c = get_zero_vector(N);
  cache->h = get_zero_vector(N);
  cache->X = get_zero_vector(S);

  cache->hf = get_zero_vector(N);
  cache->hi = get_zero_vector(N);
  cache->ho = get_zero_vector(N);
  cache->hc = get_zero_vector(N);

  cache->c_old = get_zero_vector(N);
  cache->h_old = get_zero_vector(N);

  cache->tanh_c_cache = get_zero_vector(N);

  return cache;
}

void lstm_cache_container_free(lstm_cache* cache_to_be_freed)
{
  free_vector(&(cache_to_be_freed)->c);
  free_vector(&(cache_to_be_freed)->h);
  free_vector(&(cache_to_be_freed)->c_old);
  free_vector(&(cache_to_be_freed)->h_old);
  free_vector(&(cache_to_be_freed)->X);
  free_vector(&(cache_to_be_freed)->hf);
  free_vector(&(cache_to_be_freed)->hi);
  free_vector(&(cache_to_be_freed)->ho);
  free_vector(&(cache_to_be_freed)->hc);
  free_vector(&(cache_to_be_freed)->tanh_c_cache);
}


// A = A - alpha * m, m = momentum * m + ( 1 - momentum ) * dldA
void gradients_decend(lstm_rnn* model,  lstm_rnn* gradients, float lr, int n) {

  float R = ( 1/(float)n )*lr;

  // Computing A = A - alpha * m
  vectors_substract_scalar_multiply(model->Wy, gradients->cdWy, model->Y * model->N, R);
  vectors_substract_scalar_multiply(model->Wi, gradients->cdWi, model->N * model->S, R);
  vectors_substract_scalar_multiply(model->Wc, gradients->cdWc, model->N * model->S, R);
  vectors_substract_scalar_multiply(model->Wo, gradients->cdWo, model->N * model->S, R);
  vectors_substract_scalar_multiply(model->Wf, gradients->cdWf, model->N * model->S, R);

  vectors_substract_scalar_multiply(model->by, gradients->cdby, model->Y, R);
  vectors_substract_scalar_multiply(model->bi, gradients->cdbi, model->N, R);
  vectors_substract_scalar_multiply(model->bc, gradients->cdbc, model->N, R);
  vectors_substract_scalar_multiply(model->bf, gradients->cdbf, model->N, R);
  vectors_substract_scalar_multiply(model->bo, gradients->cdbo, model->N, R);

  lstm_reset_gradient(gradients, 1);

}


void lstm_reset_gradient(lstm_rnn* model, int rc)
{

  if (rc)
  {
    vector_set_to_zero(model->cdWy, model->Y * model->N);
    vector_set_to_zero(model->cdWi, model->N * model->S);
    vector_set_to_zero(model->cdWc, model->N * model->S);
    vector_set_to_zero(model->cdWo, model->N * model->S);
    vector_set_to_zero(model->cdWf, model->N * model->S);

    vector_set_to_zero(model->cdby, model->Y);
    vector_set_to_zero(model->cdbi, model->N);
    vector_set_to_zero(model->cdbc, model->N);
    vector_set_to_zero(model->cdbf, model->N);
    vector_set_to_zero(model->cdbo, model->N);
  }

  else 
  {
    
    vector_set_to_zero(model->dWy, model->Y * model->N);
    vector_set_to_zero(model->dWi, model->N * model->S);
    vector_set_to_zero(model->dWc, model->N * model->S);
    vector_set_to_zero(model->dWo, model->N * model->S);
    vector_set_to_zero(model->dWf, model->N * model->S);

    vector_set_to_zero(model->dby, model->Y);
    vector_set_to_zero(model->dbi, model->N);
    vector_set_to_zero(model->dbc, model->N);
    vector_set_to_zero(model->dbf, model->N);
    vector_set_to_zero(model->dbo, model->N);

    set_vector_zero(model->h_prev, model->N);
    set_vector_zero(model->c_prev, model->N);

  }

    vector_set_to_zero(model->dldhf, model->N);
    vector_set_to_zero(model->dldhi, model->N);
    vector_set_to_zero(model->dldhc, model->N);
    vector_set_to_zero(model->dldho, model->N);
    vector_set_to_zero(model->dldc, model->N);
    vector_set_to_zero(model->dldh, model->N);

    vector_set_to_zero(model->dldXc, model->S);
    vector_set_to_zero(model->dldXo, model->S);
    vector_set_to_zero(model->dldXi, model->S);
    vector_set_to_zero(model->dldXf, model->S);
  

}


void lstm_zero_the_model(lstm_rnn * model)
{
  vector_set_to_zero(model->Wy, model->Y * model->N);
  vector_set_to_zero(model->Wi, model->N * model->S);
  vector_set_to_zero(model->Wc, model->N * model->S);
  vector_set_to_zero(model->Wo, model->N * model->S);
  vector_set_to_zero(model->Wf, model->N * model->S);

  vector_set_to_zero(model->by, model->Y);
  vector_set_to_zero(model->bi, model->N);
  vector_set_to_zero(model->bc, model->N);
  vector_set_to_zero(model->bf, model->N);
  vector_set_to_zero(model->bo, model->N);
 
  vector_set_to_zero(model->dldhf, model->N);
  vector_set_to_zero(model->dldhi, model->N);
  vector_set_to_zero(model->dldhc, model->N);
  vector_set_to_zero(model->dldho, model->N);
  vector_set_to_zero(model->dldc, model->N);
  vector_set_to_zero(model->dldh, model->N);

  vector_set_to_zero(model->dldXc, model->S);
  vector_set_to_zero(model->dldXo, model->S);
  vector_set_to_zero(model->dldXi, model->S);
  vector_set_to_zero(model->dldXf, model->S);
}


void sum_gradients(lstm_rnn* model)
{
  vectors_add(model->cdWy, model->dWy, model->Y * model->N);
  vectors_add(model->cdWi, model->dWi, model->N * model->S);
  vectors_add(model->cdWc, model->dWc, model->N * model->S);
  vectors_add(model->cdWo, model->dWo, model->N * model->S);
  vectors_add(model->cdWf, model->dWf, model->N * model->S);

  vectors_add(model->cdby, model->dby, model->Y);
  vectors_add(model->cdbi, model->dbi, model->N);
  vectors_add(model->cdbc, model->dbc, model->N);
  vectors_add(model->cdbf, model->dbf, model->N);
  vectors_add(model->cdbo, model->dbo, model->N);
}

void mean_gradients(lstm_rnn* gradients, double d)
{
  vectors_mean_multiply(gradients->Wy, d,  gradients->Y * gradients->N);
  vectors_mean_multiply(gradients->Wi, d,  gradients->N * gradients->S);
  vectors_mean_multiply(gradients->Wc, d,  gradients->N * gradients->S);
  vectors_mean_multiply(gradients->Wo, d,  gradients->N * gradients->S);
  vectors_mean_multiply(gradients->Wf, d,  gradients->N * gradients->S);

  vectors_mean_multiply(gradients->by, d, gradients->Y);
  vectors_mean_multiply(gradients->bi, d, gradients->N);
  vectors_mean_multiply(gradients->bc, d, gradients->N);
  vectors_mean_multiply(gradients->bf, d, gradients->N);
  vectors_mean_multiply(gradients->bo, d, gradients->N);
}


void copy_lstm(lstm_rnn* lstm, lstm_rnn* receiver)
{
  
  copy_vector(receiver->Wf, lstm->Wf, lstm->N*lstm->S);
  copy_vector(receiver->Wi, lstm->Wi, lstm->N*lstm->S);
  copy_vector(receiver->Wc, lstm->Wc, lstm->N*lstm->S);
  copy_vector(receiver->Wo, lstm->Wo, lstm->N*lstm->S);
  copy_vector(receiver->Wy, lstm->Wy, lstm->Y*lstm->N);

  copy_vector(receiver->bf, lstm->bf, lstm->N);
  copy_vector(receiver->bi, lstm->bi, lstm->N);
  copy_vector(receiver->bc, lstm->bc, lstm->N);
  copy_vector(receiver->bo, lstm->bo, lstm->N);
  copy_vector(receiver->by, lstm->by, lstm->Y);

}


void lstm_store(lstm_rnn* lstm, const char* filename)
{
  FILE * fp;

  fp = fopen(filename, "w");

  if ( fp == NULL ) 
  {
    printf("Failed to open file: %s for writing.\n", filename);
    return;
  }
  
  fprintf(fp, "{");
  fprintf(fp, "\n\t\"InputSize \": %d",  lstm->X);
  fprintf(fp, ",\n\t\"HiddenSize \": %d", lstm->N);
  fprintf(fp, ",\n\t\"OutputSize \": %d", lstm->Y);

  fprintf(fp, ",\n\t\"Wy\": ");
  vector_store_as_matrix_json(lstm->Wy, lstm->Y, lstm->N, fp);
  fprintf(fp, ",\n\t\"Wi\": ");
  vector_store_as_matrix_json(lstm->Wi, lstm->N, lstm->S, fp);
  fprintf(fp, ",\n\t\"Wc\": ");
  vector_store_as_matrix_json(lstm->Wc, lstm->N, lstm->S, fp);
  fprintf(fp, ",\n\t\"Wo\": ");
  vector_store_as_matrix_json(lstm->Wo, lstm->N, lstm->S, fp);
  fprintf(fp, ",\n\t\"Wf\": ");
  vector_store_as_matrix_json(lstm->Wf, lstm->N, lstm->S, fp);

  fprintf(fp, ",\n\t\"by\": ");
  vector_store_json(lstm->by, lstm->Y, fp);
  fprintf(fp, ",\n\t\"bi\": ");
  vector_store_json(lstm->bi, lstm->N, fp);
  fprintf(fp, ",\n\t\"bc\": ");
  vector_store_json(lstm->bc, lstm->N, fp);
  fprintf(fp, ",\n\t\"bf\": ");
  vector_store_json(lstm->bf, lstm->N, fp);
  fprintf(fp, ",\n\t\"bo\": ");
  vector_store_json(lstm->bo, lstm->N, fp);

  fprintf(fp, "\n}");

  fclose(fp);

}

 
