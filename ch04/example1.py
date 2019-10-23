import numpy as np
import matplotlib.pylab as plt

def function_1(x):
  return 0.01*x**2 + 0.1*x

def function_2(x):
  # return np.sum(x**2)
  return x[0]**2 + x[1]**2

# x0 = 3, x1 = 4 (x0으로 편미분)
def function_3(x0):
  return x0**2 + 4.0**2

def numerical_diff(f, x):
  h = 1e-4
  return (f(x+h) - f(x-h)) / (2*h)

# 모든 변수의 편미분을 벡터로 정리한 것 -> 기울기(gradient)
def numerical_gradient(f, x):
  h = 1e-4 #0.0001
  grad = np.zeros_like(x)

  for idx in range(x.size):
    tmp_val = x[idx]
    # f(x+h) 계산
    x[idx] = tmp_val + h
    fxh1 = f(x)
    # f(x-h) 계산
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val
  
  return grad

# 경사 하강법
# f - 함수
# init_x - 초기값
# lr - learning rate
# step_nuim - step 반복값
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
    
    return x

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)

# plt.xlabel("x")
# plt.ylable("f(x)")
# plt.plot(x, y)
# plt.show()

# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))

# print(numerical_diff(function_3, 3.0))

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))

init_x = np.array([-3.0, 4.0])

print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))