#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}


void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x; //val
  __pp_vec_int y; // exp
  __pp_vec_float result;
  __pp_vec_int count;
  __pp_vec_int zero_int = _pp_vset_int(0);
  __pp_vec_int one_int = _pp_vset_int(1);
  __pp_vec_float limit = _pp_vset_float(9.999999f);
  __pp_mask maskAll, maskEqualZero, maskNotZero, maskIsNotNegative, maskCount, maskIsLimit;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskEqualZero = _pp_init_ones(0);
    maskNotZero = _pp_init_ones(0);
    maskCount = _pp_init_ones(0);
    maskIsLimit = _pp_init_ones(0);
    maskAll = _pp_init_ones();

    if(i + VECTOR_WIDTH > N){
      maskAll = _pp_init_ones(N % VECTOR_WIDTH);
    }

    // load
    _pp_vload_float(x, values + i, maskAll);          // x = values[i]
    _pp_vload_int(y, exponents + i , maskAll);        // y = exponents[i]
    
    // check if zero
    _pp_veq_int(maskEqualZero, y, zero_int, maskAll); // if(y == 0){
    _pp_vset_float(result, 1.f, maskEqualZero);       // result[i] = 1.f;
                                                      // }
    maskNotZero = _pp_mask_not(maskEqualZero);        // else{
    _pp_vmove_float(result, x, maskNotZero);          // result = x
    _pp_vsub_int(count, y, one_int, maskNotZero);     // count = y - 1
                                                      // }
    
    _pp_vgt_int(maskCount, count, zero_int, maskNotZero); // count > 0
    // multiply
    while(_pp_cntbits(maskCount)){                    // while(count > 0)
      _pp_vmult_float(result, result, x, maskCount);  //result *= x;
      _pp_vsub_int(count, count, one_int, maskCount); // count--;
      _pp_vgt_int(maskCount, count, zero_int, maskCount);      
    }

    // limit
    _pp_vgt_float(maskIsLimit, result, limit, maskAll); // if(result > limit)
    _pp_vset_float(result, 9.999999f, maskIsLimit);     // result = 9.999999f

    // store
    _pp_vstore_float(output + i, result, maskAll);
    
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float x;
  __pp_vec_float result = _pp_vset_float(0.);  

  __pp_mask maskAll;
  maskAll = _pp_init_ones();

  int time = VECTOR_WIDTH;
  float output[VECTOR_WIDTH];
  
  // time complexity = O(N/vector_width)
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll); 
    _pp_vadd_float(result, result, x, maskAll);
  }

  // time complexity = O(log(vector_width))
  while(time > 1){
    _pp_hadd_float(result, result);
    _pp_interleave_float(result, result);
    time /= 2;
  }
  _pp_vstore_float(output, result, maskAll);

  return output[0];
}
