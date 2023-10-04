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

$$X=\left(\begin{matrix}1&w^{(1)} \\
1&x^{(2)} \\
1&\vdots \\
1&w^{(m)}\end{matrix}\right),\qquad W=\left(\begin{matrix}b \\
w\end{matrix}\right),\qquad y=\left(\begin{matrix}y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}\end{matrix}\right)$$

The equations can be rewritten as:

$$\left(\begin{matrix}1&1&\cdots&1 \\
x^{(1)}&x^{(2)}&\cdots&x^{(m)}\end{matrix}\right)\left(\begin{matrix}b+wx^{(1)}-y^{(1)} \\
b+wx^{(2)}-y^{(2)} \\
\vdots \\
b+wx^{(m)}-y^{(m)}\end{matrix}\right)=0$$

$$X^T(XW-y)=0$$

$$X^XW=x^Ty$$

$$W=(X^X)^{-1}X^Ty$$

Therefore,

$$\left(\begin{matrix}b \\
w\end{matrix}\right)=(X^X)^{-1}X^Ty$$

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

$$x=
 \begin{pmatrix}
 x^{(1)} \\
 x^{(2)} \\
 \vdots \\
 x^{(m)} \\
 \end{pmatrix}
 ,\qquad y=
 \begin{pmatrix}
 y^{(1)} \\
 y^{(2)} \\
 \vdots \\
 y^{(m)} \\
 \end{pmatrix}
 ,\qquad b=
 \begin{pmatrix}
 b \\
 b \\
 \vdots \\
 b \\
 \end{pmatrix} $$

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
Assume that now our dataset has $K$ features:

$$x=\begin{pmatrix}x_1&x_2&\cdots&x_K\end{pmatrix}$$

The coefficient for each of the features can be grouped into vector form:

$$w=\begin{pmatrix}w_1&w_2&\cdots&w_K\end{pmatrix}$$

The objective function is now

$$f(x)=\left(\begin{matrix}x_1&x_2&\cdots&x_K\end{matrix}\right)\left(\begin{matrix}w_1 \\
w_2 \\
\vdots \\
w_K\end{matrix}\right)+b=xw+b$$

To simplify the equation further, we can write $b=w_0$. Let

$$w=\left(\begin{matrix}w_0&w_1&w_2&\cdots&w_K\end{matrix}\right)\text{ and }x=\left(\begin{matrix}1&x_1&x_2&\cdots&x_K\end{matrix}\right)$$

Then

$$f_w(x)=\left(\begin{matrix}1&x_1&x_2&\cdots&x_K\end{matrix}\right)\left(\begin{matrix}w_0 \\
w_1 \\
w_2 \\
\vdots \\
w_K\end{matrix}\right)=xw$$

If we have $m$ number of data points in the dataset, we can put them into a matrix:

$$X=\left(\begin{matrix}x^{(1)} \\
x^{(2)} \\
\vdots \\
x^{(m)}\end{matrix}\right)=\left(\begin{matrix}x_0^{(1)}&x_1^{(1)}&\cdots&x_K^{(1)} \\
x_0^{(2)}&x_1^{(2)}&\cdots&x_K^{(2)} \\
\vdots&\vdots&&\vdots \\
x_0^{(m)}&x_1^{(m)}&\cdots&x_K^{(m)}\end{matrix}\right)$$

$$f_w(X)=Xw$$

The loss function is the same as before:
$$L(w)=\frac{1}{2m}\sum_{i=1}^m(f_w(x^{(i)})-y^{(i)})^2$$
We want to find the minimum solution $w^\ast$ for the loss function. It must satisfy for all $j$:
$$\left.\frac{\partial L(w)}{\partial w_j}\right|_{w^\ast}=0$$

## Normal equation
Let $f^{(i)}=f_w(x^{(i)})=x^{(i)}w$ and $\delta^{(i)}=f^{(i)}-y^{(i)}$. By Chain rule,

$$\begin{aligned}\frac{\partial L(w)}{\partial\delta^{(i)}}&=\sum_{i=1}^m\frac{\partial L(w)}{\partial\delta^{(i)}}\times\frac{\partial\delta^{(i)}}{\partial f^{(i)}}\times\frac{\partial f^{(i)}}{\partial w_j} \\
&=\frac{1}{m}\sum_{i=1}^m\delta^{(i)}\times1\times x_j^{(i)} \\
&=\frac{1}{m}\sum_{i=1}^m(f^{(i)}-y^{(i)})x_j^{(i)} \\
&=\frac{1}{m}\sum_{i=1}^m(f_w(x^{(i)}-y^{(i)})x_j^{(i)} \\
&=\frac{1}{m}\left(\begin{matrix}x_j^{(1)}&x_j^{(2)}&\cdots&x_j^{(m)}\end{matrix}\right)\left(\begin{matrix}f_w(x^{(1)})-y^{(1)} \\
f_w(x^{(2)})-y^{(2)} \\
\vdots \\
f_w(x^{(m)})-y^{(m)}\end{matrix}\right) \\
&=\frac{1}{m}\left(\begin{matrix}x_j^{(1)}&x_j^{(2)}&\cdots&x_j^{(m)}\end{matrix}\right)\left(\begin{matrix}x^{(1)}w-y^{(1)} \\
x^{(2)}w-y^{(2)} \\
\vdots \\
x^{(m)}w-y^{(m)}\end{matrix}\right) \\
&=\frac{1}{m}X_{:,j}^T(Xw-y)\end{aligned}$$

We therefore have $\frac{\partial L(w)}{\partial w_j}=\frac{1}{m}X_{:,j}^T(Xw-y)$. The gradient of $L$ with respect to $w$ is:

$$\nabla L(w)=\left(\begin{matrix}\frac{\partial L(w)}{\partial w_1},\cdots,\frac{\partial L(w)}{\partial w_j},\cdots\end{matrix}\right)^T=\frac{1}{m}X^T(Xw-y)$$

To make $\nabla L(w)=0$,

$$\begin{aligned}\nabla L(w)&=0 \\
X^T(Xw-y)&=0 \\
w&=(X^TX)^{-1}X^Ty\end{aligned}$$

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

# Caveats
We have a slightly more complicated dataset:

![](../assets/water_dataset.png)

Using linear regression to fit the dataset:
```python
${{ water_linear_regression }}
```
$[[ +regression.snippets.water_plot ]]

![](../assets/water_linear_regression.png)

Clearly, using a linear function to fit the data is not the best choice for this dataset. This is a phenomenon called underfitting when we use a too simple model. We can use more complex models such as polynomial functions to represent non-linear relationships. We now assume

$$f(x)=w_3x^3+w_2x^2+w_1x+w_0=\begin{pmatrix}1&x&x^2&x^3\end{pmatrix}\begin{pmatrix}w_0&w_1&w_2&w_3\end{pmatrix}^T=xw$$

We try to solve it:
```python
${{ water_linear_regression_nan }}
```
$[[ +regression.snippets.water_nan ]]

We can see from the result that it did not converge. This is because the values in our features are large, causing the large gradients.

## Standardization
One solution to the above caveat is standardization.

$$x\leftarrow\frac{x-\text{mean}(x)}{\text{std}(x)}$$

```python
${{ water_linear_regression_standardization }}
```
$[[ +regression.snippets.water_standardization ]]

![](../assets/water_standardization.png)

## Underfitting and Overfitting
We create a toy dataset with sine curve:

![](../assets/sine_toy_plot.png)

We try to fit the data with different degree of polynomials:
```python
${{ sine_underfitting_overfitting }}
```
$[[ +regression.snippets.overfitting ]]

![](../assets/sine_underfitting_overfitting.png)

We can see from the plots that 0- and 1-degree polynomials underfits the data, while 9-degree polynomial overfits the training data: it fits the training data very well, while it cannot predict well for other unseen data.

One of the ways to combat the overfitting problem it to increase the training data size. Here, we re-train the 9-degree polynomial linear regression model with more data:

![](../assets/sine_more_data.png)

In general, there are a few ways to relieve the underfitting issue:
1. Increase number of features
2. Increase the complexity of model

A few ways to solve the overfitting issue:
1. Increase training dataset size
2. Reduce the complexity of model: use a simpler model
3. Regularization: will be described below

## Regularization
Weights having huge magnitudes is a overfitting indication. In regularization, we penalize large weights in the loss function:

$$L(x;w)=\frac{1}{2m}\sum_{i=1}^m\|x^{(i)}w-y^{(i)}\|^2+\lambda\|w^2\|$$

As our objective is to minimize the loss function, the regularization term $\lambda\|w^2\|$ would need to be minimized. $\lambda$ is a regularization hyperparameter. The larger the $\lambda$ is, the stronger the effect of the regularization term becomes, and the smaller the weight $w$ remains.

The gradient of our new loss function becomes:

$$\nabla L(w)=\frac{1}{m}\sum_{i=1}^m(x^{(i)}w-y^{(i)})x^{(i)}+2\lambda w$$

```python
${{ gradient_descent_reg }}
```

We now use it to fit our 9-degree polynomial linear regression model:
```python
${{ water_regularization }}
```
$[[ +regression.snippets.regularization ]]

![](../assets/water_regularization.png)

We can see that the weights in our trained model is much smaller than before, yet it is able to produce a good result, relieving the overfitting problem.

# Logistic Regression
In linear regression, the target values are continuous. However, sometimes our target values have to be discrete. For example in classification problems, we have to group a point into a class. We can consider logistic regression as an extension of linear regression, specializing in binary classification problems.

Logistic regression sets the value of $f_w(x)$ to be between 0 and 1, representing the probability of $x$ being in a class. If we use $f_w(x)$ to represent the probability of $x$ being class $y=1$, then the probability of $x$ being class $y=0$ is $1-f_w(x)$.

$$P(y=1|x)=f_w(x)=\frac{1}{1+e^{-xw}}=\sigma(xw)$$

$$P(y=0|x)=1-f_w(x)=1-\frac{1}{1+e^{-xw}}=1-\sigma(xw)$$

For a data $x$, it has a $P(y=1|x)$ probability of being $y=1$, and a $P(y=0|x)$ probability of being $y=0$. We can say that it has a probability of $P(y=1|x)^yP(y=0|x)^{1-y}=f_w(x)^y(1-f_w(x))^{1-y}$ being sample $(x,y)$. For a dataset with $m$ samples, the probability of forming an exact dataset is
$$\prod_{i=1}^m\left(f_w(x^i)^{y^i}(1-f_w(x^i))^{1-y^i}\right)$$

We want to find the $w$ that makes the dataset most likely to happen. By taking logarithm, it is equivalent to maximizing:
$$\sum_{i=1}^m\left(y^i\log(f_w(x^i))+(1-y^i)\log(1-f_w(x^i))\right)$$

Hence, our loss function can be written as:
$$L(w)=-\frac{1}{m}\sum_{i=1}^m\left(y^i\log(f_w(x^i))+(1-y^i)\log(1-f_w(x^i))\right)$$

$-\left(y^i\log(f_w(x^i))+(1-y^i)\log(1-f_w(x^i))\right)$ is known as the cross-entropy loss. Only when the predicted value $f_w(x^i)$ for a sample $x^i$ matches its label $y^i$ can the cross-entropy loss becomes 0.

To find the minimum of $L(w)$, we need gradient descent, thus we need to know how to find the gradient of $L(w)$. Denote
$$z^i=w\odot x^i,\quad f^i=\sigma(z^i),\quad L^i=-\left(y^i\log(f^i)+(1-y^i)\log(1-f^i)\right)$$
Then
$$L(w)=\frac{1}{m}\sum_{i=1}^mL^i$$

For a given $i$ and $j$,
$$\frac{\partial L(w)}{\partial L^i}=\frac{1}{m}$$
$$\frac{\partial L^i}{\partial f^i}=-\left(\frac{y^i}{f^i}-\frac{1-y^i}{1-f^i}\right)=\frac{f^i-y^i}{f^i(1-f^i)}$$
$$\frac{\partial f^i}{\partial z^i}=\sigma(z^i)(1-\sigma(z^i))=f^i(1-f^i)$$
$$\frac{\partial z^i}{\partial w_j}=x_j^i$$

$$\begin{aligned}\frac{\partial L(w)}{\partial w_j}&=\sum_{i=1}^m\frac{\partial L(w)}{\partial L^i}\times\frac{\partial L^i}{\partial f^i}\times\frac{\partial f^i}{\partial z^i}\times\frac{\partial z^i}{\partial w_j}\\&=\frac{1}{m}\sum_{i=1}^m\frac{f^i-y^i}{f^i(1-f^i)}\times f^i(1-f^i)\times x_j^i\\&=\frac{1}{m}\sum_{i=1}^m(f^i-y^i)x_j^i\\&=\frac{1}{m}\sum_{i=1}^mx_j^i(f_w(x^i)-y^i)\end{aligned}$$

$$\begin{aligned}\nabla_w L(w)&=\begin{bmatrix}\frac{\partial L(w)}{\partial w_0}&\frac{\partial L(w)}{\partial w_1}&\frac{\partial L(w)}{\partial w_2}&\cdots&\frac{\partial L(w)}{\partial w_n}\end{bmatrix}\\&=\begin{bmatrix}\frac{1}{m}\sum_{i=1}^mx_0^i(f_w(x^i)-y^i)&\frac{1}{m}\sum_{i=1}^mx_1^i(f_w(x^i)-y^i)&\cdots&\frac{1}{m}\sum_{i=1}^mx_n^i(f_w(x^i)-y^i)\end{bmatrix}\\&=\frac{1}{m}\sum_{i=1}^m\begin{bmatrix}x_0^i(f_w(x^i)-y^i)&x_1^i(f_w(x^i)-y^i)&\cdots&x_n^i(f_w(x^i)-y^i)\end{bmatrix}\\&=\frac{1}{m}\sum_{i=1}^m\begin{bmatrix}x_0^i&x_1^i&\cdots&x_n^i\end{bmatrix}(f_w(x^i)-y^i)\\&=\frac{1}{m}\sum_{i=1}^m(f_w(x^i)-y^i)x^i\\&=\frac{1}{m}\begin{bmatrix}f_w(x^1)-y^1&f_w(x^2)-y^2&\cdots&f_w(x^m)-y^m\end{bmatrix}\begin{bmatrix}x^1 \\
x^2 \\
\vdots \\
x^m\end{bmatrix}\\&=\frac{1}{m}(f_w(x)-y)^TX=\frac{1}{m}(\sigma(Xw)-y)^TX\end{aligned}$$

We can also add a regularization term in the loss function:
$$L(w)=-\frac{1}{m}\sum_{i=1}^m\left(y^i\log(f_w(x^i))+(1-y^i)\log(1-f_w(x^i))\right)+\lambda\|w\|^2$$
Correspondingly, the gradient of $L(w)$ with respect to $w$ is
$$\nabla_wL(w)=\frac{1}{m}(f-y)^TX+2\lambda w=\frac{1}{w}(\sigma(Xw)-y)^TX+2\lambda w$$

We have another simple toy dataset:

![](../assets/logistic_toy_plot.png)

Logistic regression implementation with regularization and momentum:
```python
${{ gradient_descent_logistic_regression }}
```

Apply logistic regression model onto our toy dataset:
```python
${{ logistic_regression_toy }}
```
$[[ +regression.snippets.logistic_toy_plot ]]

## Decision boundary
We use $f_w(x)=0.5$ to separate the two classes. This is equivalent to $xw=0$.

Our dataset is in two dimensions, so we can write the decision boundary as $w_0+w_1x_1+w_2x_2=0$. Given $w$ and $x_1$, we can find $x_2=-w_0/w_2-w_1x_1/w_2$.

```python
${{ logistic_regression_decision_boundary }}
```

![](../assets/logistic_toy_decision_boundary.png)
