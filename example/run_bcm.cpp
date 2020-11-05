#include <bcm.h>

int main (int argc, char ** argv)
{

  const int n_samples = 1000;
  const int n_features = 100;

  std :: unique_ptr < float[] > data(new float[n_samples * n_features]);
  std :: iota(data.get(), data.get() + n_samples * n_features, 0.f);

  BCM bcm (10, 4, transfer :: _logistic_, 0.f, 1.f, 2e-2f, 42);
  bcm.fit(data.get(), n_samples, n_features, 10);

  return 0;
}
