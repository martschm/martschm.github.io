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

### 1. Calculation of the Distribution Function


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

---

**Project description:** Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### 1. Suggest hypotheses about the causes of observed phenomena

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

```R
if (isAwesome){
  return true
}
```

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
