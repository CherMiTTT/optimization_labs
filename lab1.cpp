#include <ios>
#include <iostream>
#include <type_traits>
#include <vector>
#include <cmath>
#include <utility>

double f(std::vector<double> x);
std::vector<double> grad_f(std::vector<double> x);
double norm(std::vector<double> grad_f);

std::vector<double> mult(double lambda, std::vector<double> v);
std::vector<double> v_sum(std::vector<double> v1, std::vector<double> v2);

std::pair<std::vector<double>, double> grad_descend(std::vector<double> x0, double eps, double t0);
std::pair<std::vector<double>, double> fastest_grad_descend(std::vector<double> x0, double eps, double t0);
double phi(double t, std::vector<double> x_k);
double gold(std::pair<double, double> interval, double eps, std::vector<double> x_k);

int main() {
  // Исходные данные
  std::vector<double> x0 = {0.5, 1};
  const double eps = 0.001;
  const double t0 = 0.1;

  auto res = grad_descend(x0, eps, t0);
  //std::cout.precision(2);
  //std::cout << std::fixed;

  std::cout << "Grad descend: " << std::endl;
  std::cout << "x* = {";
  for(int i = 0; i < res.first.size(); i++) {
    std::cout << res.first[i];
    if (i != res.first.size() - 1) {std::cout << "; ";}
  }

  std::cout << "}" << std::endl;
  std::cout << "f(x*) = " << res.second << std::endl;
  
  
  std::cout << "______________________________________________" << std::endl;

  res = fastest_grad_descend(x0, eps, t0);

  std::cout << "Fastest grad descend: " << std::endl;
  std::cout << "x* = {";
  for(int i = 0; i < res.first.size(); i++) {
    std::cout << res.first[i];
    if (i != res.first.size() - 1) {std::cout << "; ";}
  }

  std::cout << "}" << std::endl;
  std::cout << "f(x*) = " << res.second << std::endl;


  return 0;
}

[[nodiscard]] double f(std::vector<double> x) {
  double x1 = x[0];
  double x2 = x[1];
  return 2*x1*x1 + x1*x2 + x2*x2;
}

[[nodiscard]] std::vector<double> grad_f(std::vector<double> x) {
  double x1 = x[0];
  double x2 = x[1];
  double f_x1 = 4*x1 + x2;
  double f_x2 = x1 + 2*x2;
  return {f_x1, f_x2};
}


// Евклидова норма
[[nodiscard]] double norm(std::vector<double> grad_f) {
  double x1 = grad_f[0];
  double x2 = grad_f[1];

  return std::sqrt(x1*x1 + x2*x2);
  //return std::max(std::fabs(x1), std::fabs(x2));
}

[[nodiscard]] std::pair<std::vector<double>, double> grad_descend(std::vector<double> x0, double eps, double t0) {
  int k = 0; // Номер интерации
  std::vector<double> x_k = x0;
  std::vector<double> x_next;

  while (true) {
    std::cout << "k = " << k << std::endl;
    std::vector<double> grad = grad_f(x_k);
    
    if (norm(grad) < eps) {
      return {x_k, f(x_k)};
    }
    double t_k = t0;

    while(true) {
      x_next = v_sum(x_k, mult((-1.0) * t_k, grad_f(x_k)));
      //std::cout<<"f(x_k+1) = "<<f(x_next)<<std::endl;
      //std::cout<<"f(x_k) = "<<f(x_k)<<std::endl;
      //std::cout<<"(x_k+1) - (x_k) = "<<f(x_next) - f(x_k)<<std::endl;
      if (f(x_next) - f(x_k) >= 0) {
        //std::cout << "t_k updated" << std::endl;
        t_k = t_k / 2;
      }
      else {break;}
    }

    //std::cout << "t_k = " << t_k << std::endl;
    //std::cout << "norm: " <<  norm(v_sum(x_next,  mult(-1.0, x_k))) << std::endl <<
    //(norm(v_sum(x_next,  mult(-1.0, x_k))) < eps) << std::endl;
    //std::cout << "std::fabs(f(x_next) - f(x_k)) = " << std::fabs(f(x_next) - f(x_k)) << std::endl;

    if (
      norm(v_sum(x_next,  mult(-1.0, x_k))) < eps &&
      std::fabs(f(x_next) - f(x_k)) < eps){
    
        return {mult( 0.1, x_next), f(x_next)};
      }
    else {
      k++;
      x_k = x_next;
    }
  }
  return {};
}

[[nodiscard]] std::pair<std::vector<double>, double> fastest_grad_descend(std::vector<double> x0, double eps, double t0) {
  int k = 0; // Номер интерации
  std::vector<double> x_k = x0;
  std::vector<double> x_next;
  double t_k = t0;


  while (true) {
    std::cout << "k = " << k << std::endl;
    std::vector<double> grad = grad_f(x_k);
    if (norm(grad) < eps) {
      return {x_k, f(x_k)};
    }

    //auto phi = f(v_sum(x_k, mult(-1.0 * t_k, grad)));
    t_k = gold({-1, 1}, eps, x_k);
    x_next = v_sum(x_k, mult((-1.0) * t_k, grad));

    if (
      norm(v_sum(x_next,  mult(-1.0, x_k))) < eps &&
      std::fabs(f(x_next) - f(x_k)) < eps) {
        return {x_next, f(x_next)};
      }
    else {
      k++;
      x_k = x_next;
    }
  }

  return {};
}

[[nodiscard]] std::vector<double> mult(double lambda, std::vector<double> v) {
  std::vector<double> result;
  for (int i = 0; i < v.size(); i++) {
    result.push_back(v[i] * lambda);
  }
  return result;
}

[[nodiscard]] std::vector<double> v_sum(std::vector<double> v1, std::vector<double> v2) {
  std::vector<double> result;
  for (int i = 0; i < v1.size(); i++) {
    result.push_back(v1[i] + v2[i]);
  }
  return result;
}

[[nodiscard]] double phi(double t, std::vector<double> x_k) {
  double res = f(v_sum(x_k, mult(-1.0 * t, grad_f(x_k))));
  return res;
}

[[nodiscard]] double gold(std::pair<double, double> interval, double eps, std::vector<double> x_k) {
  double a = interval.first;
  double b = interval.second;
  double tau = (std::sqrt(5) + 1) / 2;
  
  while(std::fabs(b - a) >= eps) {
    double x1 = b - (b - a) / tau;
    double x2 = a + (b - a) / tau;
    double y1 = phi(x1, x_k);
    double y2 = phi(x2, x_k);

    if(y1 >= y2) a = x1;
    else b = x2;
  }
  return (a + b)/2;
}