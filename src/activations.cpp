#include <activations.h>

namespace transfer
{

  float linear (const float & x)
  {
    return x;
  }

  float g_linear (__unused const float & x)
  {
    return 1.f;
  }


  float stair (const float & x)
  {
    const int n = static_cast < int >(std :: floor (x));
    return (n % 2) ? (x - n) + std :: floor(x * .5f) : std :: floor(x * .5f);
  }

  float g_stair (const float & x)
  {
    return (std :: floor( x ) == x ) ? 0.f : 1.f;
  }


  float hardtan (const float & x)
  {
    return ( x < -2.5f ) ? 0.f : ( x > 2.5f ) ? 1.f : .2f * x + .5f;
  }

  float g_hardtan (const float & x)
  {
    return ( x > -2.5f && x < 2.5f ) ? .2f : 0.f;
  }


  float logistic (const float & x)
  {
    return 1.f / (1.f + math :: exp(- x));
  }

  float g_logistic (const float & x)
  {
    return (1.f - x) * x;
  }


  float loggy (const float & x)
  {
    return 2.f / (1.f + math :: exp(- x)) - 1.f;
  }

  float g_loggy (const float & x)
  {
    const float y = ( x + 1.f) * .5f;
    return 2.f * (1.f - y) * y;
  }


  float relu (const float & x)
  {
    return ( x > 0.f ) ? x : 0.f;
  }

  float g_relu (const float & x)
  {
    return (x > 0.f) ? 1.f : 0.f;
  }


  float elu (const float & x)
  {
    return (x >= 0.f ) * x + ( x < 0.f ) * std :: expm1f(x);
  }

  float g_elu (const float & x)
  {
    return (x >= 0.f) + (x < 0.f) * (x + 1.f);
  }


  float relie (const float & x)
  {
    return (x > 0.f ) ? x : 1e-2f * x;
  }

  float g_relie (const float & x)
  {
    return (x > 0.f) ? 1.f : 1e-2f;
  }


  float ramp (const float & x)
  {
    return  x * ( x > 0.f ) + .1f * x;
  }

  float g_ramp (const float & x)
  {
    return (x > 0.f) + .1f;
  }


  float leaky (const float & x)
  {
    return (x > 0.f ) ? x : leaky_coeff * x;
  }

  float g_leaky (const float & x)
  {
    return (x > 0.f) ? 1.f : leaky_coeff;
  }


  float tanhy (const float & x)
  {
    return 2.f / (1. + math :: exp (- (x + x))) - 1.f;
  }

  float g_tanhy (const float & x)
  {
    return 1.f - x * x;
  }


  float plse (const float & x)
  {
    return (x < -4.f ) ?
            1e-2f * ( x + 4.f ) :
            ( x > 4.f ) ?
            1e-2f * ( x - 4.f ) + 1.f :
            .125f * x + .5f;
  }

  float g_plse (const float & x)
  {
    return ( x < 0.f || x > 1.f) ? 1e-2f : .125f;
  }


  float lhtan (const float & x)
  {
    return x < 0.f ? 1e-3f * x : x > 1.f ? 1e-3f * (x - 1.f) + 1.f : x;
  }

  float g_lhtan (const float & x)
  {
    return ( x > 0.f && x < 1.f) ? 1.f : 1e-3f;
  }


  float selu (const float & x)
  {
    return (x >= 0.f) * 1.0507f * x + (x < 0.f) * 1.0507f * 1.6732f * std :: expm1f(x);
  }

  float g_selu (const float & x)
  {
    return (x >= 0.f) * 1.0507f + (x < 0.f) * (x + 1.0507f * 1.6732f);
  }

  float elliot (const float & x)
  {
    return .5f * steepness * x / (1.f + std :: fabs(x + steepness)) + .5f;
  }

  float g_elliot (const float & x)
  {
    const float last_forward = 1.f + std :: fabs(x * steepness); // miss null condition
    return .5f * steepness / (last_forward * last_forward);
  }

  float symm_elliot (const float & x)
  {
    return x * steepness / (1.f + std :: fabs(x * steepness));
  }

  float g_symm_elliot (const float & x)
  {
    const float last_forward = 1.f + std :: fabs(x * steepness); // miss null condition
    return steepness / (last_forward * last_forward);
  }

  float softplus (const float & x)
  {
    return std :: log1pf(math :: exp(x));
  }

  float g_softplus (const float & x)
  {
    const float ex = math :: exp(x);
    return ex / (1.f + ex);
  }

  float softsign (const float & x)
  {
    return x / (std :: fabs(x) + 1.f);
  }

  float g_softsign (const float & x)
  {
    const float fx = std :: fabs(x) + 1.f;
    return 1.f / (fx * fx);
  }

  float asymm_logistic (const float & x)
  {
    return x < 0.f ? -1.f * (2.f / (1.f + math :: exp(2.f * x)) - 1.f) : 50.f * (2.f / (1.f + math :: exp(-2.f * x / 50.f)) - 1.f);
  }

  float g_asymm_logistic (const float & x)
  {
    const float par = x < 0 ? -1.f : 50.f;
    const float denom = 1.f + math :: exp(-2.f * x / par);
    return 4.f * math :: exp(-2.f * x / par) / (denom * denom);
  }

  void swish_array (const float * x, const int & n, float * output_sigmoid, float * output)
  {

#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < n; ++i)
    {
      output_sigmoid[i] = 1.f / (1.f + math :: exp(- x[i]));
      output[i] = x[i] * output_sigmoid[i];
    }
  }

  void swish_gradient (const float * x, const int & n, const float * sigmoid, float * delta)
  {

#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < n; ++i)
      delta[i] *= x[i] + sigmoid[i] * (1.f - x[i]);

  }

  void mish_array (const float * x, const int & n, float * input_activation, float * output)
  {
    // const float MISH_THRESHOLD = 20.0f;

#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < n; ++i)
    {
      const float x_val = x[i];
      input_activation[i] = x[i]; // store value before activation

      output[i] = x_val * (2.f / (1.f + math :: exp( -2.f * std :: log1pf(math :: exp(x_val)) )) - 1.f);
    }
  }

  void mish_gradient (const int & n, const float * activation_input, float * delta)
  {
    // const float MISH_THRESHOLD = 20.0f;

#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < n; ++i)
    {
      // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
      // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
      float inp = activation_input[i];
      const float sp = std :: log1pf(math :: exp(inp));
      const float grad_sp = 1 - math :: exp(-sp);
      const float tsp = math :: tanh(sp);
      const float grad_tsp = (1.f - tsp * tsp) * grad_sp;
      const float grad = inp * grad_tsp + tsp;
      delta[i] *= grad;
    }
  }

  std :: function < float(const float &) > activate ( const int & active)
  {
    switch (active)
    {
      case _logistic_:       return transfer :: logistic;
      case _loggy_:          return transfer :: loggy;
      case _relu_:           return transfer :: relu;
      case _elu_:            return transfer :: elu;
      case _relie_:          return transfer :: relie;
      case _ramp_:           return transfer :: ramp;
      case _linear_:         return transfer :: linear;
      case _tanh_:           return transfer :: tanhy;
      case _plse_:           return transfer :: plse;
      case _leaky_:          return transfer :: leaky;
      case _stair_:          return transfer :: stair;
      case _hardtan_:        return transfer :: hardtan;
      case _lhtan_:          return transfer :: lhtan;
      case _selu_:           return transfer :: selu;
      case _elliot_:         return transfer :: elliot;
      case _symm_elliot_:    return transfer :: symm_elliot;
      case _softplus_:       return transfer :: softplus;
      case _softsign_:       return transfer :: softsign;
      case _asymm_logistic_: return transfer :: asymm_logistic;
      default:               return nullptr;
    }
  }

  std :: function < float(const float &) > gradient ( const int & active)
  {
    switch (active)
    {
      case _logistic_:       return transfer :: g_logistic;
      case _loggy_:          return transfer :: g_loggy;
      case _relu_:           return transfer :: g_relu;
      case _elu_:            return transfer :: g_elu;
      case _relie_:          return transfer :: g_relie;
      case _ramp_:           return transfer :: g_ramp;
      case _linear_:         return transfer :: g_linear;
      case _tanh_:           return transfer :: g_tanhy;
      case _plse_:           return transfer :: g_plse;
      case _leaky_:          return transfer :: g_leaky;
      case _stair_:          return transfer :: g_stair;
      case _hardtan_:        return transfer :: g_hardtan;
      case _lhtan_:          return transfer :: g_lhtan;
      case _selu_:           return transfer :: g_selu;
      case _elliot_:         return transfer :: g_elliot;
      case _symm_elliot_:    return transfer :: g_symm_elliot;
      case _softplus_:       return transfer :: g_softplus;
      case _softsign_:       return transfer :: g_softsign;
      case _asymm_logistic_: return transfer :: g_asymm_logistic;
      default:               return nullptr;
    }
  }


}