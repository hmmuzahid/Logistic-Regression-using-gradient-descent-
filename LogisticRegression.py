import numpy as np
try:
  from mat_opr_python import *
  import mat_opr_python as obj
except:
  import numpy as obj

class LogisticRegression:
  def __init__(self, lambda_ = 0):
    self.lambda_ = lambda_
  
  #Create a method to fit the logistic regression model
  def sigmoid(self, fx, obj):
    mask = fx>0
    out = obj.zeros(fx.shape)
    out[mask] = 1/(1+obj.exp(-fx[mask]))
    exp_neg = obj.exp(fx[~mask])
    out[~mask] = exp_neg/(1+exp_neg)
    return out
  
  
  def fit(self, x, y, obj=obj, max_iter=500):
    m, n = x.shape
    y = y.reshape(-1, 1)
    self.obj = obj
    #Calculate the mean of each features
    means = x.mean(axis=0, keepdims=True)
    
    #Calculate the std of each features
    stds = x.std(axis=0, keepdims=True) + 1e-4
    
    #Subtract the mean and divide by the std
    #It removes the bias(intercept) out of the equation and can reduce the variance
    x = (x - means)/stds
    
    
    #Create initial weight(s) and bias
    #Start with 0.1
    w = obj.array([0.1]*n).reshape(-1, 1)
    b = .1
    #Create initial rate(s)
    #Start with 0.1
    rate = obj.array([.1]*n).reshape(-1, 1)
    b_rate = .1
    #Create initial acceleration for rate(s)
    #We increase the rate(s) by 10% until we find the maximum rate(s) that doesn't overshoot(s)
    accel = obj.array([.1]*n).reshape(-1, 1)
    b_accel = .1
    #Keep track of the previous derivative(s) to compare it with the new one(s)
    old_grd = None
    b_old_grd = None
    for _ in range(max_iter):
      #Calculate new gradient
      p = self.sigmoid(x @ w + b, obj) - y
      new_grd = obj.array((x.T @ p)/m) + (self.lambda_/m)*w
      b_new_grd = p.mean()
      #Update rate
      rate += rate*accel
      b_rate += b_rate*b_accel
      #Calculate the very first derivatives and set it to old_grd and continue
      if old_grd is None:
        old_grd = new_grd
        b_old_grd = b_new_grd
        continue
      
      #Use masking to check overshooting
      mask = (new_grd>0) != (old_grd>0)
      #print(mask.shape)
      accel[mask] = 0
      rate[mask] /= 1.1
      w += obj.where(mask , rate*old_grd, -rate*new_grd)
      rate[mask] /= 1.1
      
      old_grd = new_grd
      
      if (b_new_grd>0) != (b_old_grd>0):
        b_accel = 0
        b_rate /= 1.1
        b += b_rate * b_old_grd
        b_rate /= 1.1
      else:
        b -= b_rate * b_new_grd
      b_old_grd = b_new_grd
      
      #If any all derivative is less than or equal to 1e-6, we reached the minimum. Return the weights
      if (abs(new_grd) <= 1e-4).all() and abs(b_new_grd) <= 1e-4:
        #Scale the weight(s) back to the original form and return them
        self.w = w / stds.T
        self.b = b - ((w*means) / stds.T).sum()
        break
        
  
  def predict(self, x):
    p = self.sigmoid(x @ self.w + self.b, self.obj)
    p[p>.5] = 1
    p[p<=.5] = 0
    return p



if __name__=="__main__":
  from sklearn.linear_model import LogisticRegression as skl
  
  data = np.load("train-test.npz")
  x_train = data["x_train"]# (700, 10)
  y_train = data["y_train"]# (700,)
  x_test = data["x_test"]# (300, 10)
  y_test = data["y_test"]# (300,)
  
  custom_model = LogisticRegression()
  custom_model.fit(array(x_train), array(y_train))
  
  sk_model = skl(penalty=None)
  sk_model.fit(x_train, y_train)
  
  sk_pred = sk_model.predict(x_test)
  custom_pred = custom_model.predict(array(x_test))
  
  print("sklearn      custom")
  for cu, sk in zip(custom_pred[:20], sk_pred[:20]):
    print(sk, cu)
  
  