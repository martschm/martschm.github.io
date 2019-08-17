# Aggregation of Integer-Valued Risks with Copula-Induced Dependency Structure

---

## Abstract

In this diploma thesis we investigate a portfolio of d integer-valued risks and calculate the distribution of the aggregate loss S, which is the sum of these. To generalize the popular assumption of independence used in practice, we model the dependency structure of the individual risks using copulas, allowing for a wide range of flexibility. After a rather detailed introduction to copula theory, the main part of this thesis starts with a formula for the distribution function of S. In addition, a recursion formula for the probability mass function of S is provided. Bounds on the distribution of S determined by the Rearrangement Algorithm serve to quantify the model risk caused by feasible scenarios of dependency. To illustrate the theoretical considerations, the final chapter contains a multitude of numerical examples in which, besides the distribution and probability mass function, common risk measures such as Value-at-Risk and Expected Shortfall for S are calculated under various dependency structures.

---

## Full Text

View the thesis [here](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/3559554?originalFilename=true).

---

## Summary of the Main Results

View the summary [here](/pdf/diploma_thesis_presentation.pdf)

---

## R-Code

### 1. Calculation of the Distribution Function of the Aggregate Loss S

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

### 2. Recursion for the Calculation of the Probability Mass Function of the Aggregate Loss S

<img src="images/R_calculation_probability_mass_function_recursion.png?raw=true"/>

```r
helper.setNextIndex=function(ind,maxValues) {
    for (i in 1:length(ind)) {
        ind[i]=ind[i]+1
        if (ind[i]<=maxValues[i]) {
            break
        } else {
            ind[i]=1
        }
    }
    return(ind)
}

helper.calcCombinations=function(d,n,support) {
    j=numeric(d)
    ind=rep(1,d)
    number=1
    maxValues=numeric(d)
    combinations=list()
    for (i in 1:d) {
        maxValues[i]=length(support[[i]])
    }
    repeat {
        for (i in 1:d) {
            j[i]=support[[i]][ind[i]]
        }
        if (sum(j)<=n) {
            combinations[[number]]=c(sum(j),j)
            number=number+1
        }
        ind=helper.setNextIndex(ind,maxValues)
        if (sum(ind)==d) {
            break
        }
    }
    return(combinations)
}

S.probabilityMassFunctionRecursion=function(d,n,copula,margins,support,combinations=list(),firstRun=TRUE) {
    result=NULL
    valMargins=numeric(d)
    combinationsN = list()
    ind=1
    if(firstRun==TRUE) {
        combinations=helper.calcCombinations(d,n,support)
    }
    uniqueComb=NULL
    for (i in 1:length(combinations)) {
        uniqueComb[i]=combinations[[i]][1]
    }
    if (sum(uniqueComb==n)==0) {
        return(0)
    }
    if (n==0) {
        for (i in 1:d) {
            valMargins[i]=margins[[i]](0)
        }
        result=copula(valMargins)
        return(result)
    } else {
        for (i in 1:length(combinations)) {
            if(combinations[[i]][1]==n) {
                combinationsN[[ind]]=combinations[[i]][2:(d+1)]
                ind=ind+1
            }
        }
        if (length(combinationsN)==0) {
            return(0)
        } else {
            tmpCopula=0
            tmpSum=0
            for (i in 1:length(combinationsN)) {
                for (k in 1:d) {
                    valMargins[k]=margins[[k]](combinationsN[[i]][k])
                }
                tmpCopula=tmpCopula+copula(valMargins)
            }
            for (k in 1:n) {
                tmpSum=tmpSum+choose(k+d-1,d-1)*S.probabilityMassFunctionRecursion(d,n-k,copula,margins,support,combinations,FALSE)
            }
            result=tmpCopula-tmpSum
            return(result)
        }
    }
}
```

---
