#include <bcm.h>

BCM :: BCM (const int & outputs, const int & batch_size,
            int activation, float mu, float sigma, float epsilon, float interaction_strenght, int seed
            ) : BasePlasticity (outputs, batch_size, activation, mu, sigma, epsilon, seed)
{
  this->init_interaction_matrix(interaction_strenght);
}


BCM :: BCM (const BCM & b) : BasePlasticity (b)
{
}

BCM & BCM :: operator = (const BCM & b)
{
  BasePlasticity :: operator = (b);

  return *this;
}


void BCM :: init_interaction_matrix (const float & interaction_strenght)
{
  this->interaction_matrix.reset(new float[this->outputs * this->outputs]);

  if (interaction_strenght != 0.f)
  {
    for (int i = 0; i < this->outputs; ++i)
      for (int j = 0; j < this->outputs; ++j)
      {
        const int idx = i * this->outputs + j;
        this->interaction_matrix[idx] = i == j ? 1.f : -interaction_strenght;
      }

    // map the matrix to the eigen format
    Eigen :: Map < Eigen :: Matrix < float, Eigen :: Dynamic, Eigen :: Dynamic, Eigen :: RowMajor > > L(interaction_matrix.get(), outputs, outputs);
    // compute the inverse of the matrix
    auto inverse = L.inverse();
    // re-map the result into the member variable
    Eigen :: Map < Eigen :: MatrixXf > ( this->interaction_matrix.get(), inverse.rows(), inverse.cols() ) = inverse;
  }

  else
  {
    for (int i = 0; i < this->outputs; ++i)
      for (int j = 0; j < this->outputs; ++j)
      {
        const int idx = i * this->outputs + j;
        this->interaction_matrix[idx] = i == j ? 1.f : 0.f;
      }
  }
}


void BCM :: weights_update (float * X, const int & n_features, float * weights_update)
{
  static float nc;

  nc = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
    weights_update[i] = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->outputs; ++i)
  {
    const int idx = i * this->batch;
    this->theta[i] = std :: accumulate(this->output.get() + idx,
                                       this->output.get() + idx + this->batch,
                                       0.f,
                                       [](const float & res, const float & xi)
                                       {
                                        return res + xi * xi;
                                       }) / this->batch;
  }

#ifdef _OPENMP
  #pragma omp for collapse (2)
#endif
  for (int i = 0; i < this->outputs; ++i)
    for (int j = 0; j < this->batch; ++j)
    {
      const int idx = i * this->batch + j;
      const float out = this->output[idx];
      const float phi = out * (out - this->theta[i]);
      this->output[idx] = phi * this->gradient(this->activation(out));

      const float A_PART = this->output[idx];
      for (int k = 0; k < n_features; ++k)
        weights_update[i * n_features + k] += A_PART * X[j * n_features + k];
    }

#ifdef _OPENMP
  #pragma omp for reduction (max : nc)
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    const float out = std :: fabs(weights_update[i]);
    nc = nc < out ? out : nc;
  }

  nc = 1.f / std :: max(nc, BasePlasticity :: precision);

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
    weights_update[i] *= nc;
}
