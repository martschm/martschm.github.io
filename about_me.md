# Dipl.-Ing. Martin Schmidt

---

**View My CV [here](/pdf/20190808_CV_Martin_Schmidt.pdf).**

---

## Personal Overview

- Date of birth: December 31, 1992
- Nationality: Austria
- Period of notice: 3 months (mid and end of month)

---

## Work Experience

Credit Risk Manager at [ING in Austria](https://ing.at) (July 2016 - Ongoing)

- Development and validation of retail risk models and scorecards in SAS and Python using advanced statistical methods (close agile collaboration with advanced analytics colleagues from head office)
- Execution of scenario-based simulations for various test & learn business experiments (e.g. Preapproved Loans and Early Collections customer segmentation)
- Led the development of automated reporting from internal databases in the financial and non-financial risk area via SAS/SQL and VBA from scratch
- Backtesting of external rating models (CRIF, KSV, Credify.AT)
- Member of the local Credit Risk Committee

---

## Academic Education

MSc Financial- and Actuarial Mathematics at [Vienna University of Technology](https://www.tuwien.at/) (March 2016 - March 2019)

- Passed with distinction
- Diploma Thesis: [Aggregation of Integer-Valued Risks with Copula Induced Dependency Structure](/diploma_thesis) (nominated for the [AVÖ-price](http://avoe.at/wp-content/uploads/2014/09/AVOe_Foerderung_Abschlussarbeiten_2016.pdf) in 2020)

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
