functions {

real clayton_copula(real u, real v, real theta) {
  if (theta <= 0.0) 
    reject("clayton_copula: theta must > 0");
  
  return log1p(theta) - (theta + 1) * (log(u) + log(v))
         - (1 + 2 * theta) / theta * log(pow(u, -theta) + pow(v, -theta) - 1);
}


}
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real mu1;
  real mu2;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=0> theta;
}
model {
  target += normal_lpdf(x | mu1, sigma1);
  target += normal_lpdf(y | mu2, sigma2);
  
  for (n in 1:N)
    target += clayton_copula(normal_cdf(x[n] | mu1, sigma1),
                             normal_cdf(y[n] | mu2, sigma2), theta);
}

