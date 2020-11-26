#ifndef __weights_h__
#define __weights_h__

#include <fmath.h>       // fast math functions

#include <unordered_map> // std :: unordered_map
#include <numeric>       // std :: inner_product

#define ERROR_WEIGHTS_INITIALIZATION 001

enum weights_init_t{ _zeros_ = 0, _ones_, _uniform_, _normal_, _lecun_uniform_, _glorot_uniform_, _glorot_normal_, _he_uniform_, _he_normal_
};///< weights initialization types


namespace weights_init
{
  static const std :: unordered_map < std :: string, int > get {
                                                                 {"zeros"          , _zeros_},
                                                                 {"ones"           , _ones_},
                                                                 {"uniform"        , _uniform_},
                                                                 {"normal"         , _normal_},
                                                                 {"lecun_uniform"  , _lecun_uniform_},
                                                                 {"glorot_uniform" , _glorot_uniform_},
                                                                 {"glorot_normal"  , _glorot_normal_},
                                                                 {"he_uniform"     , _he_uniform_},
                                                                 {"he_normal"      , _he_normal_},
                                                               }; ///< Utility for the weight initialization functions management
} // end namespace


/**
* @class weights_initialization
* @brief Abstract type representing a weights initialization algorithm.
* The object implements different weights initialization algorithms, in
* particular:
*   - Zeros
*   - Ones
*   - Uniform
*   - Normal
*   - Lecun Uniform
*   - Glorot Uniform
*   - Glorot Normal
*   - He Uniform
*   - He Normal
*
* @details The desired weights initialization algorithm can be set using
* the type variable in the constructor signature.
* The core functionality of the object is given by the
* 'init' member function which applies the desired initialization
* algorithm using the member parameters.
*
*/
class weights_initialization
{
  // Private members

  std :: mt19937 engine; ///< Random number generator

  int type; ///< Initialization type to use

  float mu; ///< mean of the weights distribution (used in normal distribution initializations)
  float sigma; ///< standard deviation of the weights distribution (used in normal distribution initialization)
  float scale; ///< bound of the distribution domain (used in uniform distribution initialization)


public:

  // Constructor

  /**
  * @brief Default constructor.
  *
  */
  weights_initialization ();

  /**
  * @brief Construct the object using the list of parameters.
  *
  * @details The constructor takes the parameters related to any initialization
  * function but they will be used only if the selected algorithm requires them.
  *
  * @note The type variable determines the desired initialization method.
  *
  * @param type Initialization type to apply.
  * @param mu Mean of the gaussian distribution that initializes the weights.
  * @param sigma Standard deviation of the gaussian distribution that initializes the weights.
  * @param scale Dimension of the uniform distribution that initializes the weights.
  * @param seed Random number generator seed.
  *
  */
  weights_initialization (const int & type, float mu=0.f, float sigma=1.f, float scale=1.f, int seed=42);

  // Destructors

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory.
  *
  */
  ~weights_initialization() = default;

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param args weights_initialization object
  *
  */
  weights_initialization & operator = (const weights_initialization & args);

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param args weights_initialization object
  *
  */
  weights_initialization (const weights_initialization & args);

  /**
  * @brief Init the member arrays using the given number of weights
  *
  * @details This function init the member arrays used for the
  * optimization steps.
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void init (float * weights, const int & inputs, const int & outputs);

private:

  /**
  * @brief Initialize weights with zero values
  *
  * @details The initialization function follows the equation:
  *
  * ```python
  * w = np.zeros(shape=size, dtype=float)
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void zeros (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief Initialize weights with one values
  *
  * @details The initialization function follows the equation:
  *
  * ```python
  * w = np.ones(shape=size, dtype=float)
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void ones (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief Sample initial weights from the uniform distribution.
  *
  * @details Parameters are sampled from U(a, b).
  * The initialization function follows the equation:
  *
  * ```python
  * w = np.random.uniform(low=-scale, high=scale, size=size)
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void uniform (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief Sample initial weights from the Gaussian distribution.
  *
  * @details Initial weight parameters are sampled from N(mean, std).
  * The initialization function follows the equation:
  *
  * ```python
  * w = np.random.normal(loc=mu, scale=std, size=size)
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void normal (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief LeCun uniform initializer.
  *
  * @details It draws samples from a uniform distribution within [-limit, limit]
  * where `limit` is `sqrt(3 / inputs)` where `inputs` is the number of input
  * units in the weight matrix.
  *
  * ```c++
  * w = uniform(w, size, std :: sqrt(3.f / inputs))
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void lecun_uniform (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief Glorot uniform initializer, also called Xavier uniform initializer.
  *
  * @details It draws samples from a uniform distribution within [-limit, limit]
  * where `limit` is `sqrt(6 / (inputs + outputs))` and `inputs` is the number
  * of input units in the weight matrix and `outputs` is the number of output
  * units in the weight matrix.
  *
  * ```c++
  * w = uniform(w, size, std :: sqrt(6.f / (inputs + outputs))
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void glorot_uniform (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief Glorot normal initializer, also called Xavier normal initializer.
  *
  * @details It draws samples from a truncated normal distribution centered on 0
  * with `stddev = sqrt(2 / (inputs + outputs))` and `inputs` is the number of
  * input units in the weight matrix and `outputs` is the number of
  * output units in the weight matrix.
  *
  * ```c++
  * w = normal(w, size, 0.f, std :: sqrt(2.f / (inputs + outputs))
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void glorot_normal (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief He uniform variance scaling initializer.
  *
  * @details It draws samples from a uniform distribution within [-limit, limit]
  * where `limit` is `sqrt(6 / inputs)` and `inputs` is the number
  * of input units in the weight matrix.
  *
  * ```c++
  * w = uniform(w, size, std :: sqrt(6.f / inputs)
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void he_uniform (float * weights, const int & inputs, const int & outputs);

  /**
  * @brief He normal initializer.
  *
  * @details It draws samples from a truncated normal distribution centered on 0
  * with `stddev = sqrt(2 / inputs)` and `inputs` is the number of input units
  * in the weight matrix.
  *
  * ```c++
  * w = normal(w, size, 0.f, std :: sqrt(2.f / inputs)
  * ```
  *
  * @param weights Matrix of weights in ravel format.
  * @param inputs Number of rows of the weight matrix.
  * @param outputs Number of columns of the weight matrix.
  */
  void he_normal (float * weights, const int & inputs, const int & outputs);

}; // end class

#endif //__weights_h__