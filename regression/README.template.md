# Linear Regression

We assume a linear relationship between data $x$ and label $y$:
$$y=f(x)=wx+b$$

We call $w$ the weight and $b$ the bias.

We want the loss function $L(w, b)$ to be as small as possible:
$$L=\frac{1}{m}\sum_{i=1}^m(f^{(i)}-y^{(i)})^2=\frac{1}{m}\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})^2$$

For the sake of cleaner computations, we usually divide the loss function by 2:
$$L(w,b)=\frac{1}{2m}\sum_{i=1}^m(f^{(i)}-y^{(i)})^2=\frac{1}{2m}\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})^2$$

We want to find $w$ and $b$ through linear regression.

## Normal equation

$$\frac{\partial L}{\partial w}=\frac{1}{2m}\frac{\partial(\sum(wx^{(i)}+b-y^{(i)})^2)}{\partial w}=\frac{1}{m}\sum(wx^{(i)}+b-y^{(i)})x^{(i)}$$

$$\frac{\partial L}{\partial b}=\frac{1}{2m}\frac{\partial(\sum(wx^{(i)}+b-y^{(i)})^2)}{\partial b}=\frac{1}{m}\sum(wx^{(i)}+b-y^{(i)})$$

For $L$ to be minimum, the following equations must be satisfied:

$$\frac{\partial L}{\partial w}=\sum(wx^{(i)}+b-y^{(i)})x^{(i)}=0$$

$$\frac{\partial L}{\partial b}=\sum(wx^{(i)}+b-y^{(i)})=0$$

Denote
$$X=\left(\begin{matrix}1&w^{(1)}\\1&x^{(2)}\\1&\vdots\\1&w^{(m)}\end{matrix}\right),\qquad W=\left(\begin{matrix}b\\w\end{matrix}\right),\qquad y=\left(\begin{matrix}y^{(1)}\\y^{(2)}\\\vdots\\y^{(m)}\end{matrix}\right)$$

The equations can be rewritten as:
$$\left(\begin{matrix}1&1&\cdots&1\\x^{(1)}&x^{(2)}&\cdots&x^{(m)}\end{matrix}\right)\left(\begin{matrix}b+wx^{(1)}-y^{(1)}\\b+wx^{(2)}-y^{(2)}\\\vdots\\b+wx^{(m)}-y^{(m)}\end{matrix}\right)=0$$
$$X^T(XW-y)=0$$
$$X^XW=x^Ty$$
$$W=(X^X)^{-1}X^Ty$$

Therefore,
$$\left(\begin{matrix}b\\w\end{matrix}\right)=(X^X)^{-1}X^Ty$$

### Example
We have a small dataset:
![](../assets/food_truck_plot.png)
$[[ -regression.snippets.food_truck_plot ]]

We can use normal equation to find $w$ and $b$ in $f(x)=wx+b$ that best fit the data:
```python
${{ food_truck_normal_equation }}
```
$[[ +regression.snippets.normal_equation_example ]]

## Gradient Descent
As normal equation involves computing matrix inverse, using normal equation is slow for datasets with a lot of samples or a lot of features. Instead, we can apply gradient descent algorithm to find the minimum for the loss function $L$.

Starting from $(w_0, b_0)$, the update formulae:
$$w:=w-\alpha\frac{\partial L}{\partial w}$$
$$b:=w-\alpha\frac{\partial L}{\partial b}$$

Denote
$$x=\left(\begin{matrix} x^{(1)} \\\ x^{(2)} \\\ \vdots \\\ x^{(m)} \end{matrix}\right),\qquad y=\left(\begin{matrix} y^{(1)} \\\ y^{(2)} \\\ \vdots \\\ y^{(m)} \end{matrix}\right),\qquad b=\left(\begin{matrix} b \\\ b \\\ \vdots \\\ b \end{matrix}\right)$$

We can write
$$\frac{\partial L}{\partial w}=\text{mean}((wx+b-y)\odot x)$$
$$\frac{\partial L}{\partial b}=\text{mean}(wx+b-y)$$

Code to use gradient descent algorithm to solve linear regression:
```python
${{ linear_regression }}
```

### Example
Continuing from our previous example, we can use gradient descent to find $w$ and $b$ that best fit the data:
```python
${{ food_truck_gradient_descent }}
```
$[[ +regression.snippets.gradient_descent_example ]]

![](../assets/food_truck_linear_regression_plot.png)

We can plot the loss function to see how the loss changes throughout the process:

![](../assets/food_truck_linear_regression_loss.png)

# Multivariate linear regression
Assume that now our dataset has $K$ features: $x=\left(\begin{matrix}x_1&x_2&\cdots&x_K\end{matrix}\right)$. The coefficient for each of the features can be grouped into vector form: $w=\left(\begin{matrix}w_1&w_2&\cdots&w_K\end{matrix}\right)$. The objective function is now
$$f(x)=\left(\begin{matrix}x_1&x_2&\cdots&x_K\end{matrix}\right)\left(\begin{matrix}w_1\\w_2\\\vdots\\w_K\end{matrix}\right)+b=xw+b$$

To simplify the equation further, we can write $b=w_0$. Let $w=\left(\begin{matrix}w_0&w_1&w_2&\cdots&w_K\end{matrix}\right)$ and $x=\left(\begin{matrix}1&x_1&x_2&\cdots&x_K\end{matrix}\right)$. Then
$$f_w(x)=\left(\begin{matrix}1&x_1&x_2&\cdots&x_K\end{matrix}\right)\left(\begin{matrix}w_0\\w_1\\w_2\\\vdots\\w_K\end{matrix}\right)=xw$$

If we have $m$ number of data points in the dataset, we can put them into a matrix:
$$X=\left(\begin{matrix}x^{(1)}\\x^{(2)}\\\vdots\\x^{(m)}\end{matrix}\right)=\left(\begin{matrix}x_0^{(1)}&x_1^{(1)}&\cdots&x_K^{(1)}\\x_0^{(2)}&x_1^{(2)}&\cdots&x_K^{(2)}\\\vdots&\vdots&&\vdots\\x_0^{(m)}&x_1^{(m)}&\cdots&x_K^{(m)}\end{matrix}\right)$$
$$f_w(X)=Xw$$

The loss function is the same as before:
$$L(w)=\frac{1}{2m}\sum_{i=1}^m(f_w(x^{(i)})-y^{(i)})^2$$
We want to find the minimum solution $w^\ast$ for the loss function. It must satisfy for all $j$:
$$\left.\frac{\partial L(w)}{\partial w_j}\right|_{w^\ast}=0$$

## Normal equation
Let $f^{(i)}=f_w(x^{(i)})=x^{(i)}w$ and $\delta^{(i)}=f^{(i)}-y^{(i)}$. By Chain rule,
$$\begin{aligned}\frac{\partial L(w)}{\partial\delta^{(i)}}&=\sum_{i=1}^m\frac{\partial L(w)}{\partial\delta^{(i)}}\times\frac{\partial\delta^{(i)}}{\partial f^{(i)}}\times\frac{\partial f^{(i)}}{\partial w_j}\\&=\frac{1}{m}\sum_{i=1}^m\delta^{(i)}\times1\times x_j^{(i)}\\&=\frac{1}{m}\sum_{i=1}^m(f^{(i)}-y^{(i)})x_j^{(i)}\\&=\frac{1}{m}\sum_{i=1}^m(f_w(x^{(i)}-y^{(i)})x_j^{(i)}\\&=\frac{1}{m}\left(\begin{matrix}x_j^{(1)}&x_j^{(2)}&\cdots&x_j^{(m)}\end{matrix}\right)\left(\begin{matrix}f_w(x^{(1)})-y^{(1)}\\f_w(x^{(2)})-y^{(2)}\\\vdots\\f_w(x^{(m)})-y^{(m)}\end{matrix}\right)\\&=\frac{1}{m}\left(\begin{matrix}x_j^{(1)}&x_j^{(2)}&\cdots&x_j^{(m)}\end{matrix}\right)\left(\begin{matrix}x^{(1)}w-y^{(1)}\\x^{(2)}w-y^{(2)}\\\vdots\\x^{(m)}w-y^{(m)}\end{matrix}\right)\\&=\frac{1}{m}X_{:,j}^T(Xw-y)\end{aligned}$$

We therefore have $\frac{\partial L(w)}{\partial w_j}=\frac{1}{m}X_{:,j}^T(Xw-y)$. The gradient of $L$ with respect to $w$ is:
$$\nabla L(w)=\left(\begin{matrix}\frac{\partial L(w)}{\partial w_1},\cdots,\frac{\partial L(w)}{\partial w_j},\cdots\end{matrix}\right)^T=\frac{1}{m}X^T(Xw-y)$$

To make $\nabla L(w)=0$,
$$\begin{aligned}\nabla L(w)&=0\\X^T(Xw-y)&=0\\w&=(X^TX)^{-1}X^Ty\end{aligned}$$

### Example
We can create a toy dataset sampled from $z=3x+2y+c$.
```python
${{ multivariate_toy_dataset }}
```
![](../assets/multivariate_plane.png)

Finding the best matching plane with linear regression:
```python
${{ multivariate_toy_dataset_normal_equation }}
```
$[[ +regression.snippets.multivariate_linear_regression ]]

![](../assets/multivariate_plane_normal_equation.png)

## Gradient Descent
Again, normal equation is slow for big datasets as it involves computing the matrix inverse. A better approach is gradient descent (with momentum):
```python
${{ linear_regression_vec }}
```

### Example
```python
${{ multivariate_linear_regression_gradient_descent }}
```
$[[ +regression.snippets.multivariate_linear_regression_gradient_descent ]]

The loss value decreases over time:

![](../assets/multivariate_plane_loss_history.png)