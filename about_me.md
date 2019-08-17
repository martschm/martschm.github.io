# Aggregation of Integer-Valued Risks with Copula-Induced Dependency Structure

---

## 1. Abstract

In this diploma thesis we investigate a portfolio of d integer-valued risks and calculate the distribution of the aggregate loss S, which is the sum of these. To generalize the popular assumption of independence used in practice, we model the dependency structure of the individual risks using copulas, allowing for a wide range of flexibility. After a rather detailed introduction to copula theory, the main part of this thesis starts with a formula for the distribution function of S. In addition, a recursion formula for the probability mass function of S is provided. Bounds on the distribution of S determined by the Rearrangement Algorithm serve to quantify the model risk caused by feasible scenarios of dependency. To illustrate the theoretical considerations, the final chapter contains a multitude of numerical examples in which, besides the distribution and probability mass function, common risk measures such as Value-at-Risk and Expected Shortfall for S are calculated under various dependency structures.

---

## 2. Full Text

View the thesis [here](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/3559554?originalFilename=true).

---

## 3. Summary of the Main Results

View the summary [here](/pdf/diploma_thesis_presentation.pdf).

---

## 4. Some Numerical Results

- Distribution Function of the sum S of three Poisson-distributed random variables under different dependency scenarios (using Gaussian Copulas).

<img src="images/sum_poisson_variables.png?raw=true"/>

- Probability Mass Function of the sum S of three Poisson-distributed random variables under different dependency scenarios (using Gaussian Copulas).

<img src="images/sum_poisson_variables_pmf.png?raw=true"/>

- Value-at-Risk of the sum S of three Poisson-distributed random variables under different dependency scenarios (using Gaussian Copulas).

<img src="images/sum_poisson_variables_var.png?raw=true"/>

- Expected Shortfall of the sum S of three Poisson-distributed random variables under different dependency scenarios (using Gaussian Copulas).

<img src="images/sum_poisson_variables_es.png?raw=true"/>

---

## 5. R-Code

In this section I provide different R codes for calculation of the numerical results presented in the diploma thesis.
- 5.1. Calculation of the Distribution Function of the Aggregate Loss S
- 5.2. Recursion for the Calculation of the Probability Mass Function of the Aggregate Loss S
- 5.3. Calculation of the Probability Mass Function of the Aggregate Loss S by Integration over Copula Densities
- 5.4. Rearrangement Algorithm for the Calculation of Sharp Bounds on the Distribution of the Aggregate Loss S
- 5.5. Calculation of Sharp Bounds on Value-at-Risk and Expected Shortfall

<br><br>

### 5.1. Calculation of the Distribution Function of the Aggregate Loss S

<img src="images/R_calculation_distribution_function.png?raw=true"/>

```r
helper.evalCopula=function(j,copula,margins) {
    valMargins=numeric(length(j))
    for (i in 1:length(valMargins)) {
        valMargins[i]=margins[[i]](j[i])
    }
    return(copula(valMargins))
}

helper.setNextCombination=function(j,target) {
    for (i in 1:length(j)) {
        j[i]=j[i]+1
        if (j[i]<=target&&sum(j)<=target) {
            break
        } else {
            j[i]=0
        }
    }
    return(j)
}

S.distribution=function(d,n,copula,margins) {
    if (n<0) {
        return(0)
    }
    n=floor(n)
    j=numeric(d)
    result=0
    for (k in 0:min(d-1,n)) {
        j=numeric(d)
        if (sum(j)==n-k) {
            result=result+((-1)^k)*choose(d-1,k)*helper.evalCopula(j,copula,margins)           
        }
        repeat {
            j=helper.setNextCombination(j,n-k)
            if (sum(j)==0) {
                break
            } else if (sum(j)==n-k) {
                result=result+((-1)^k)*choose(d-1,k)*helper.evalCopula(j,copula,margins)
            }
        }
    }
    return(result)
}
```
