# Inference in Linear Regression: From Standard Errors to Hypothesis Testing

## A Unified Lecture Covering the Theory of Precision, Distribution, and Testing in Econometrics

---

## I. The Big Picture: Why This Matters

You have learned how to obtain a point estimate $\hat{\beta}$ from data using OLS. That answered the question: *What is our best guess of $\beta$?* But a point estimate by itself is almost useless. Suppose I tell you the return to education is 0.05. Your immediate follow-up should be: *How confident should I be in that number?* Could it plausibly be 0.01? Could it be 0.10? Could it even be negative?

This entire block of material is about answering two further questions. First, **how precise is $\hat{\beta}$ as a measure of $\beta$?** This is the standard error question. Second, **can we rule out that $\beta$ equals some specific value, say zero?** This is the inference question. Both questions require us to understand the *distribution* of $\hat{\beta}$ — not just its expected value — and to have a reliable estimate of the *variance* of that distribution.

The conceptual arc is this: we start by deriving formulas for the variance of $\hat{\beta}$ under different assumptions about the error structure (homoskedasticity, heteroskedasticity, clustering). We then ask what distribution $\hat{\beta}$ follows, first under the strong assumptions of the normal regression model, then under the weaker assumptions of large-sample theory. Once we know the distribution, we construct test statistics and conduct hypothesis tests — for single parameters, for combinations of parameters, and for nonlinear functions of parameters using the delta method.

Throughout, a recurring theme is the **tension between the strength of your assumptions and the reliability of your inference**. Stronger assumptions (normality, homoskedasticity) give you exact, finite-sample distributions but are often implausible. Weaker assumptions (just finite moments and i.i.d. sampling) give you only *approximate* distributions that require the sample to be "large enough" — and how large is large enough depends entirely on the problem at hand.

---

## II. Coefficients Are Random Variables: Foundations

### The fundamental insight

Before we talk about standard errors, we must internalize one thing: **estimated coefficients are random variables**. If you could draw a different sample of the same size from the same population and re-run the same regression, you would get a different $\hat{\beta}$. This variation is not a defect of OLS; it is an inherent feature of estimation from finite samples.

Why does $\hat{\beta}$ vary across samples? Consider the population model:

$$Y = X\beta + e, \quad E[e \mid X] = 0.$$

The condition $E[e \mid X] = 0$ means the error terms are uncorrelated with $X$ *in expectation* — that is, if we had infinitely many draws. In any finite sample, however, the realized errors will be correlated with $X$ to some extent simply by chance. Different samples produce different realizations of the errors, and hence different estimates.

Here is a tiny example to make this visceral. Suppose the true model is $\log(w) = 10 + 0.05 \cdot \text{ed} + e$, so the true return to education is 0.05. Consider two samples of three workers:

| Sample 1 | ed | e    | log(w) |   | Sample 2 | ed | e    | log(w) |
|----------|----|------|--------|---|----------|----|------|--------|
| Worker 1 | 10 | +0.10| 10.60  |   | Worker 1 | 10 | −0.05| 10.45  |
| Worker 2 | 12 | 0.00 | 10.60  |   | Worker 2 | 12 | 0.00 | 10.60  |
| Worker 3 | 14 | −0.05| 10.65  |   | Worker 3 | 14 | +0.10| 10.80  |

In Sample 1, the error happens to be negatively correlated with education (the most educated person draws a negative error), pulling the slope estimate down: $\hat{\beta}_1 \approx 0.0125$. In Sample 2, the opposite happens and $\hat{\beta}_1 \approx 0.0875$. The true effect is 0.05 in both cases. The difference is entirely due to the random realization of the error terms.

### Standard errors: what they are

The **standard error** of $\hat{\beta}$ is the estimated standard deviation of $\hat{\beta}$ across hypothetical repeated samples. It is our best guess, from the data we have, of how much $\hat{\beta}$ would vary if we could re-run the estimation many times.

A critical point: standard errors are themselves random variables. Just as $\hat{\beta}$ varies from sample to sample, so does $s(\hat{\beta})$. The standard error has its own mean and variance. Even if $s(\hat{\beta})$ is unbiased on average, in any particular sample it could be too large or too small. This has a direct consequence for inference: the $t$-statistic $T = \hat{\beta}/s(\hat{\beta})$ depends on *both* the numerator and the denominator, and a small denominator (an underestimated standard error) inflates $T$ and leads to false rejections.

When assessing different standard error estimators, we should care not only about their bias or consistency, but also about their **variance**. An estimator that is consistent under weaker assumptions but has much higher variance can perform worse in finite samples than a less general but lower-variance alternative. This is a central theme of the course.

---

## III. The Variance of the OLS Estimator: The Sandwich Formula

### Setup and notation

We work with the linear regression model:

$$Y = X\beta + e, \quad E[e \mid X] = 0,$$

where $Y$ is $n \times 1$, $X$ is $n \times k$, $\beta$ is $k \times 1$, and $e$ is $n \times 1$. We assume i.i.d. sampling and that $Q_{XX} = E[XX']$ is positive definite (no perfect multicollinearity).

The OLS estimator is $\hat{\beta} = (X'X)^{-1}X'Y$, which can be decomposed as:

$$\hat{\beta} = \beta + (X'X)^{-1}X'e.$$

This expression is the key to everything: $\hat{\beta}$ equals the truth plus a "noise" term $(X'X)^{-1}X'e$ that depends on the sample realizations of $X$ and $e$.

### Conditional variance

Conditioning on $X$ (treating $X$ as fixed), the only source of randomness in $\hat{\beta}$ is $e$. Using the rule $\text{Var}(AZ) = A \cdot \text{Var}(Z) \cdot A'$ for a constant matrix $A$:

$$V_{\hat{\beta}} = \text{Var}(\hat{\beta} \mid X) = (X'X)^{-1} X' \text{Var}(e \mid X) \, X (X'X)^{-1}.$$

Now, what is $\text{Var}(e \mid X)$? Under independence of the observations, the covariance between $e_i$ and $e_j$ for $i \neq j$ is zero. So $\text{Var}(e \mid X)$ is a diagonal matrix $D$ with the individual error variances $\sigma_i^2 = \text{Var}(e_i \mid X_i)$ on the diagonal:

$$V_{\hat{\beta}} = (X'X)^{-1}(X'DX)(X'X)^{-1}.$$

This is the **sandwich formula** — so called because the "meat" $X'DX$ is sandwiched between two "slices of bread" $(X'X)^{-1}$. This formula is completely general under independence. Everything that follows is about what happens under different assumptions about $D$.

### Why we condition on $X$

Two reasons. First, it simplifies the math: $X$ and $X'X$ become constants, and variance comes only from $e$. Second, and more substantively, it answers the right question. Having estimated a regression on a particular sample, you want to know how much $\hat{\beta}$ would vary across other samples *with the same distribution of the regressors*. Conditioning on $X$ gives you exactly that.

---

## IV. Homoskedastic Standard Errors

### The assumption

Homoskedasticity means the conditional variance of the error is the same for every observation:

$$E[e_i^2 \mid X_i] = \sigma^2 \quad \text{for all } i.$$

This does *not* mean every error is the same — it means they are drawn from distributions with equal variance. Under this assumption, $D = I_n \sigma^2$, and the sandwich formula collapses:

$$V_{\hat{\beta}} = (X'X)^{-1}(X' \cdot I_n \sigma^2 \cdot X)(X'X)^{-1} = (X'X)^{-1} \sigma^2.$$

This is the **classical** variance formula. It is much simpler than the general sandwich — instead of estimating the $n \times n$ matrix $D$, we need to estimate only a single scalar $\sigma^2$.

### What determines precision: the univariate case

To build intuition, consider a simple regression of $Y$ on a single demeaned variable $X$ (no constant needed). Then $X'X = \sum(X_i - \bar{X})^2 = n\hat{\sigma}_X^2 \equiv SST_X$. The variance of $\hat{\beta}_1$ is:

$$\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{SST_X} = \frac{\sigma^2}{n\hat{\sigma}_X^2}.$$

Three things make $\hat{\beta}_1$ more precise: (1) smaller error variance $\sigma^2$ — there is less noise in the outcome; (2) larger sample size $n$ — you have more data; (3) larger variance of $X$ — you have more "identifying" variation. The standard error is:

$$s(\hat{\beta}_1) = \sqrt{\frac{s^2}{SST_X}} \quad \text{where} \quad s^2 = \frac{1}{n-k}\sum_{i=1}^n \hat{e}_i^2.$$

We divide by $n-k$ rather than $n$ because the sample variance of the residuals is biased downward as an estimator of the true error variance — each estimated coefficient "uses up" one degree of freedom.

### The multivariate case and the role of $(X'X)^{-1}$

With multiple regressors, $X'X$ is a $k \times k$ matrix. If the variables in $X$ are demeaned, $(X'X)/n$ is essentially the sample variance-covariance matrix of the regressors. Inverting it performs crucial mathematical operations:

- **Demeaning**: If $X$ includes a constant (a column of ones), inversion effectively demeans the other variables.
- **Residualization**: If the regressors are correlated, inversion extracts the *unique* variation in each variable — the residual variation after regressing each $X_j$ on all other covariates.

This is why, in the multivariate case, the variance of $\hat{\beta}_j$ depends on $\sigma_j^2$, the residual variance from regressing $X_j$ on all other regressors. More correlation among the regressors means less unique variation, which means less precise estimates.

When you add a new control variable to a regression, two things happen simultaneously: $s^2$ weakly decreases (you explain more variation in $Y$), but $\sigma_j^2$ also weakly decreases (there is less unique variation left in your variable of interest). The net effect on $s(\hat{\beta}_j)$ is ambiguous — this is why standard errors can increase when you add controls.

---

## V. Heteroskedastic Standard Errors (HC0–HC3)

### Why homoskedasticity fails

In many applications, the variability of the outcome differs systematically across observations. Consider a wage equation: the variance of unobserved ability might be larger for highly educated workers (who sort into diverse occupations) than for less educated workers. If $\sigma_i^2$ varies with $X_i$, the classical formula $(X'X)^{-1}\sigma^2$ is wrong — it is not that it gives biased point estimates (OLS is still unbiased and consistent under heteroskedasticity), but that it misestimates the *variance* of those estimates.

### The heteroskedastic variance formula

Returning to the full sandwich formula:

$$V_{\hat{\beta}} = (X'X)^{-1}(X'DX)(X'X)^{-1}$$

where $D$ is diagonal with potentially different $\sigma_i^2$ on the diagonal. We can write the meat as:

$$X'DX = \sum_{i=1}^n X_i X_i' \sigma_i^2.$$

In the univariate case, this becomes $\sum(X_i - \bar{X})^2 \sigma_i^2$, giving:

$$\text{Var}(\hat{\beta}_1) = \frac{\sum(X_i - \bar{X})^2 \sigma_i^2}{SST_X^2}.$$

Compare this to the homoskedastic case: $\sigma^2 / SST_X$. The heteroskedastic formula weights each observation's error variance by $(X_i - \bar{X})^2$ — observations far from $\bar{X}$ get more weight because they have more *leverage*, meaning they are more informative about the slope.

### When does heteroskedasticity inflate or deflate standard errors?

If $(X_i - \bar{X})^2$ and $\sigma_i^2$ are positively correlated — high-leverage observations also have high error variance — then the true variance is *larger* than the classical formula suggests. Assuming homoskedasticity would understate your uncertainty. This is the more common case in practice (for example, when studying income, richer people have both more extreme $X$ values and more variable outcomes).

If the correlation is negative — high-leverage observations have low error variance — the true variance is *smaller* than the classical formula, and homoskedastic standard errors would be conservative. This is rarer but possible.

### Estimating heteroskedastic standard errors

Since the true $\sigma_i^2$ are unobservable, we replace them with estimates based on the residuals. The most common estimators are:

**HC0**: Plug in squared residuals $\hat{e}_i^2$ for $\sigma_i^2$ directly:
$$\hat{V}^{HC0}_{\hat{\beta}} = (X'X)^{-1}\left(\sum_{i=1}^n X_i X_i' \hat{e}_i^2\right)(X'X)^{-1}.$$

**HC1**: Apply a degrees-of-freedom correction $n/(n-k)$:
$$\hat{V}^{HC1}_{\hat{\beta}} = \frac{n}{n-k}(X'X)^{-1}\left(\sum_{i=1}^n X_i X_i' \hat{e}_i^2\right)(X'X)^{-1}.$$
HC1 is the default "robust" standard error in Stata and most packages.

**HC2 and HC3**: These use the concept of **leverage**. The leverage of observation $i$ is $h_{ii}$, the $i$-th diagonal element of the projection (hat) matrix $P = X(X'X)^{-1}X'$. In the univariate case, $h_{ii} = (X_i - \bar{X})^2 / \sum(X_j - \bar{X})^2$: it is the share of total $X$-variation attributable to observation $i$.

A key result: even if the true errors are homoskedastic, the *residuals* are heteroskedastic:
$$\text{Var}(\hat{e}_i \mid X) = (1 - h_{ii})\sigma^2.$$

This means $\hat{e}_i^2$ is a downward-biased estimate of $\sigma_i^2$, especially for high-leverage observations. The standardized residuals $\tilde{e}_i = \hat{e}_i / \sqrt{1 - h_{ii}}$ correct for this and have $\text{Var}(\tilde{e}_i \mid X) = \sigma^2$. The prediction errors $\bar{e}_i = \hat{e}_i / (1 - h_{ii})$ overcorrect.

HC2 plugs in $\tilde{e}_i^2 = \hat{e}_i^2/(1-h_{ii})$; HC3 plugs in $\bar{e}_i^2 = \hat{e}_i^2/(1-h_{ii})^2$. This produces a strict ordering: $\hat{V}^{HC3} > \hat{V}^{HC2} > \hat{V}^{HC0}$. Moreover, HC2 is unbiased for the true variance under homoskedasticity.

### The variance trade-off

Here is a crucial practical point. Heteroskedasticity-robust standard errors are consistent under weaker assumptions, but they have **higher variance** than classical standard errors. The reason is intuitive: robust SEs rely on individual squared residuals $\hat{e}_i^2$, which are noisy proxies for the true $\sigma_i^2$. Classical SEs pool all residuals into a single $s^2$, which averages out the noise.

In simulations with a homoskedastic DGP, $n=100$, and normal errors, the average robust SE is close to the average classical SE, but the *standard deviation* of the robust SEs across samples is about 40% larger. This means that in any given sample, the robust SE is more likely to be far from the truth — and when it happens to be too small, the $t$-statistic is inflated, leading to false rejections. We will see this issue reappear, amplified, when we discuss clustering.

---

## VI. The Normal Regression Model: Exact Finite-Sample Inference

### The assumptions

The **normal regression model** adds two strong assumptions on top of what we already had:

1. **Homoskedasticity**: $E[e_i^2 \mid X_i] = \sigma^2$.
2. **Normality**: $e \mid X \sim N(0, I_n \sigma^2)$.

Together with independence, this gives us $e \mid X \sim N(0, I_n \sigma^2)$: the error vector is multivariate normal, independent of $X$.

### Normality of $\hat{\beta}$

Since $\hat{\beta} - \beta = (X'X)^{-1}X'e$ is a *linear function* of $e$, and linear functions of normal random variables are normal, we immediately get:

$$\hat{\beta} \mid X \sim N\left(\beta, \sigma^2(X'X)^{-1}\right).$$

This is the payoff: under these assumptions, $\hat{\beta}$ is exactly normally distributed, centered at the truth, with variance we can compute. For the $j$-th coefficient:

$$\hat{\beta}_j \mid X \sim N\left(\beta_j, \sigma^2 [(X'X)^{-1}]_{jj}\right).$$

Standardizing:

$$\frac{\hat{\beta}_j - \beta_j}{\sqrt{\sigma^2 [(X'X)^{-1}]_{jj}}} \sim N(0,1).$$

### The $t$-statistic and the $t$-distribution

We do not know $\sigma^2$, so we replace it with $s^2$. This introduces additional randomness. The crucial result is:

$$\frac{(n-k)s^2}{\sigma^2} \sim \chi^2_{n-k}$$

and this quantity is *independent* of $\hat{\beta}$ (both results require homoskedasticity and normality). Now:

$$T = \frac{\hat{\beta}_j - \beta_j}{s(\hat{\beta}_j)} = \frac{\hat{\beta}_j - \beta_j}{\sqrt{\sigma^2[(X'X)^{-1}]_{jj}}} \bigg/ \sqrt{\frac{s^2}{\sigma^2}} = \frac{N(0,1)}{\sqrt{\chi^2_{n-k}/(n-k)}} \sim t_{n-k}.$$

The $t$-distribution with $n-k$ degrees of freedom is essentially a standard normal with fatter tails: the extra randomness from estimating $\sigma^2$ widens the distribution. As $n-k$ grows, $s^2/\sigma^2 \to 1$ in probability, and $t_{n-k} \to N(0,1)$.

### Why this is important

This gives us an **exact, finite-sample** distribution for the $t$-statistic — no approximation, no "large sample" caveat. From this distribution, we can compute exact $p$-values and confidence intervals.

### Why this is fragile

The result fails if either assumption is violated:

- If errors are **non-normal**, $\hat{\beta}$ is not normally distributed in finite samples, and the $t$-statistic does not follow a $t$-distribution.
- If the model is **heteroskedastic**, $(n-k)s^2/\sigma^2$ no longer has a $\chi^2$ distribution, and the independence between $s^2$ and $\hat{\beta}$ breaks down.

A particularly important case: using heteroskedasticity-robust standard errors (HC1, HC2, HC3) in the denominator of the $t$-statistic does *not* restore the $t$-distribution, even if the errors are normal and homoskedastic. The reason is that robust SEs have higher variance, making the $t$-statistic have fatter tails than $t_{n-k}$. This is a finite-sample problem that only disappears asymptotically — which motivates the next section.

Think about what this means with a concrete case: binary treatment $X$ with $E(X) = 0.1$ (a rare treatment), normal errors, and $n=100$. In simulations, the share of $|T| > 2$ using HC1 standard errors is about 8.6% instead of the nominal 5%. HC2 gives about 7.5%, and HC3 gives about 6.4%. None of these are "right" in finite samples under these conditions, even though the DGP is homoskedastic and the errors are normal. The problem is the high variance of the robust SE estimator when leverage is unequal. This is the kind of reasoning relevant for understanding the simulations in the problem set.

---

## VII. Large-Sample (Asymptotic) Inference

When the assumptions of the normal regression model fail — as they usually do — we rely on asymptotic theory. The idea is to show that, as $n \to \infty$, $\hat{\beta}$ and its associated test statistics have nice distributional properties regardless of the shape of the error distribution.

### Building blocks

We need four ingredients.

**1. Convergence in probability.** A sequence $Z_n$ converges in probability to $Z$, written $Z_n \xrightarrow{p} Z$, if for every $\delta > 0$, $P(\|Z_n - Z\| \leq \delta) \to 1$ as $n \to \infty$. Intuitively: $Z_n$ gets arbitrarily close to $Z$ with probability approaching one.

**2. The Weak Law of Large Numbers (WLLN).** If $Y_i$ are i.i.d. with $E\|Y\| < \infty$, then $\bar{Y} \xrightarrow{p} E[Y]$. The sample mean converges to the population mean. This works for any random variable (or vector) with a finite first moment.

**3. The Continuous Mapping Theorem (CMT).** If $Z_n \xrightarrow{p} c$ and $g$ is continuous at $c$, then $g(Z_n) \xrightarrow{p} g(c)$. This lets us combine convergence results. For instance, if $\hat{Q}_{XX} \xrightarrow{p} Q_{XX}$, then $\hat{Q}_{XX}^{-1} \xrightarrow{p} Q_{XX}^{-1}$.

**4. The Central Limit Theorem (CLT).** If $Y_i$ are i.i.d. with $E\|Y\|^2 < \infty$, then:
$$\sqrt{n}(\bar{Y} - \mu) \xrightarrow{d} N(0, V)$$
where $\mu = E[Y]$ and $V = E[(Y-\mu)(Y-\mu)']$. The sample mean, after rescaling by $\sqrt{n}$, converges *in distribution* to a normal, regardless of the shape of $Y$'s distribution.

Why the $\sqrt{n}$ rescaling? Because $\text{Var}(\hat{\beta})$ shrinks to zero as $n$ grows — without rescaling, $\hat{\beta} - \beta$ would converge to a degenerate point mass at zero. Multiplying by $\sqrt{n}$ keeps the variance constant in the limit.

### Consistency of OLS

Write the OLS estimator as:

$$\hat{\beta} = \hat{Q}_{XX}^{-1}\hat{Q}_{XY} \quad \text{where} \quad \hat{Q}_{XX} = \frac{1}{n}\sum X_i X_i', \quad \hat{Q}_{XY} = \frac{1}{n}\sum X_i Y_i.$$

By the WLLN (since $X_i X_i'$ and $X_i Y_i$ are i.i.d. with finite expectations under the assumption $E\|X\|^2 < \infty$ and $E[Y^2] < \infty$):

$$\hat{Q}_{XX} \xrightarrow{p} Q_{XX}, \quad \hat{Q}_{XY} \xrightarrow{p} Q_{XY}.$$

By the CMT:
$$\hat{\beta} = \hat{Q}_{XX}^{-1}\hat{Q}_{XY} \xrightarrow{p} Q_{XX}^{-1}Q_{XY} = \beta.$$

This is **consistency**: $\hat{\beta}$ converges to the true value as $n \to \infty$. Note this requires only finite second moments — much weaker than normality.

Equivalently, from $\hat{\beta} - \beta = \hat{Q}_{XX}^{-1}\hat{Q}_{Xe}$ where $\hat{Q}_{Xe} = \frac{1}{n}\sum X_i e_i$, we need $\hat{Q}_{Xe} \xrightarrow{p} E[Xe] = 0$, which holds by the WLLN under $E[Xe] = 0$.

### An important aside on estimator properties

The problem set asks you to evaluate an estimator $\bar{Y}^* = \frac{\sum Y_i}{n+1}$ for $E[Y]$. This is a good occasion to distinguish three properties:

- **Unbiasedness**: $E[\bar{Y}^*] = E[Y]$? Compute: $E[\bar{Y}^*] = \frac{n \cdot E[Y]}{n+1} \neq E[Y]$. So it is biased, with the bias being $-E[Y]/(n+1)$.

- **Consistency**: $\bar{Y}^* \xrightarrow{p} E[Y]$? As $n \to \infty$, $\frac{n}{n+1} \to 1$, so $\bar{Y}^* = \frac{n}{n+1}\bar{Y} \to E[Y]$ by the WLLN and CMT. Yes, it is consistent.

- **Asymptotic normality**: $\sqrt{n}(\bar{Y}^* - E[Y]) \xrightarrow{d} N(0, V)$? Write $\sqrt{n}(\bar{Y}^* - E[Y]) = \frac{n}{n+1}\sqrt{n}(\bar{Y} - E[Y]) - \frac{\sqrt{n}}{n+1}E[Y]$. By the CLT, $\sqrt{n}(\bar{Y} - E[Y]) \xrightarrow{d} N(0, \text{Var}(Y))$. The factor $\frac{n}{n+1} \to 1$. The second term $\frac{\sqrt{n}}{n+1}E[Y] \to 0$. By Slutsky's theorem, $\sqrt{n}(\bar{Y}^* - E[Y]) \xrightarrow{d} N(0, \text{Var}(Y))$.

The lesson: an estimator can be biased in finite samples yet still consistent and asymptotically normal. Unbiasedness is a finite-sample property; consistency and asymptotic normality are large-sample properties.

### Asymptotic normality of $\hat{\beta}$

For the asymptotic distribution, we need slightly stronger assumptions: **finite fourth moments** of $X$ and $Y$ (Assumption 7.2 in Hansen). This is because the CLT applies to the product $X_i e_i$, whose variance involves $E[X^2 e^2]$, which in turn requires control over fourth moments.

Start from:
$$\sqrt{n}(\hat{\beta} - \beta) = \left(\frac{1}{n}\sum X_i X_i'\right)^{-1}\left(\frac{1}{\sqrt{n}}\sum X_i e_i\right).$$

The first factor converges in probability: $\hat{Q}_{XX}^{-1} \xrightarrow{p} Q_{XX}^{-1}$.

The second factor converges in distribution by the CLT: since $X_i e_i$ is i.i.d. with mean zero and covariance matrix $\Omega = E[X X' e^2]$,

$$\frac{1}{\sqrt{n}}\sum X_i e_i \xrightarrow{d} N(0, \Omega).$$

By **Slutsky's theorem** (which lets us combine probability limits with distributional limits):

$$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} Q_{XX}^{-1} \cdot N(0, \Omega) = N(0, V_\beta)$$

where $V_\beta = Q_{XX}^{-1}\Omega Q_{XX}^{-1}$.

This is the **asymptotic normality of OLS**. Notice: no assumption about the distribution of $e$ was needed — only i.i.d. sampling and finite fourth moments. The normality comes entirely from the CLT.

### Connecting asymptotic and exact variance

$V_\beta$ is the asymptotic variance of $\sqrt{n}(\hat{\beta} - \beta)$, while $V_{\hat{\beta}} = (X'X)^{-1}(X'DX)(X'X)^{-1}$ is the exact conditional variance of $\hat{\beta}$. They are related by:

$$nV_{\hat{\beta}} = \left(\frac{X'X}{n}\right)^{-1}\left(\frac{X'DX}{n}\right)\left(\frac{X'X}{n}\right)^{-1} \xrightarrow{p} V_\beta.$$

So $V_{\hat{\beta}} \approx V_\beta/n$ for large $n$.

### Consistent variance estimation

Under homoskedasticity, the classical estimator $\hat{V}^0_\beta = s^2 \hat{Q}_{XX}^{-1} \cdot n$ is consistent for $V_\beta$.

Under heteroskedasticity, we need the HC estimators: $\hat{V}^{HC1}_\beta \xrightarrow{p} V_\beta$ under finite fourth moments. All the HC variants (HC0 through HC3) are consistent for $V_\beta$.

The key difference: if the model happens to be homoskedastic, the classical estimator converges *faster* (has lower variance in finite samples) than the HC estimators. If the model is heteroskedastic, the classical estimator is inconsistent while the HC estimators remain consistent.

---

## VIII. The Delta Method: Functions of Parameters

### Motivation

Often we are interested not in a coefficient $\beta_j$ directly but in some function $\theta = r(\beta)$. For example: the ratio of two coefficients $\beta_1/\beta_0$, the difference $\beta_1 - \beta_2$, or a marginal effect from a nonlinear specification. We need to know the asymptotic distribution of $\hat{\theta} = r(\hat{\beta})$.

### The scalar delta method: intuition

Suppose $X_n$ is asymptotically normal: $\sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, \sigma^2)$. We want the distribution of $g(X_n)$.

Taylor-expand $g(X_n)$ around $\theta$:

$$g(X_n) \approx g(\theta) + g'(\theta)(X_n - \theta).$$

Then:
$$\sqrt{n}(g(X_n) - g(\theta)) \approx g'(\theta) \cdot \sqrt{n}(X_n - \theta) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2).$$

The variance of the transformation is the original variance, scaled by the square of the derivative. This is the **delta method**: it propagates uncertainty through smooth functions.

### The multivariate delta method

For $\theta = r(\beta)$ where $r : \mathbb{R}^k \to \mathbb{R}^q$, define the Jacobian:

$$R = R(\beta) = \frac{\partial r(\beta)'}{\partial \beta} \quad (k \times q).$$

If $\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} N(0, V_\beta)$, then:

$$\sqrt{n}(\hat{\theta} - \theta) \xrightarrow{d} N(0, V_\theta) \quad \text{where} \quad V_\theta = R'V_\beta R.$$

### Concrete example: $\theta = \beta_1/\beta_0$

This arises naturally in treatment effect estimation. Suppose you run $Y = \beta_0 + \beta_1 D + e$, where $D$ is a treatment dummy. Then $\beta_0$ is the control group mean and $\beta_1$ is the treatment effect. The *relative* treatment effect is $\theta = \beta_1/\beta_0$.

Here $r(\beta) = \beta_1/\beta_0$, so:

$$R = \begin{pmatrix} \partial(\beta_1/\beta_0)/\partial\beta_0 \\ \partial(\beta_1/\beta_0)/\partial\beta_1 \end{pmatrix} = \begin{pmatrix} -\beta_1/\beta_0^2 \\ 1/\beta_0 \end{pmatrix}.$$

Therefore:

$$V_\theta = R' V_\beta R = \frac{1}{\beta_0^2}V_{11} + \frac{\beta_1^2}{\beta_0^4}V_{00} - \frac{2\beta_1}{\beta_0^3}V_{01}$$

where $V_{jk}$ are elements of $V_\beta$.

**Why is $\hat{\theta}$ consistent?** Because $\hat{\beta} \xrightarrow{p} \beta$, and $r(\cdot) = \beta_1/\beta_0$ is a continuous function (assuming $\beta_0 > 0$), the CMT gives $\hat{\theta} = r(\hat{\beta}) \xrightarrow{p} r(\beta) = \theta$.

**Practical computation.** Replace $\beta$ with $\hat{\beta}$ and $V_\beta$ with $\hat{V}_\beta$ in all the formulas above. The asymptotic standard error is $s(\hat{\theta}) = \sqrt{\hat{V}_{\hat{\theta}}} = \sqrt{\hat{R}'\hat{V}_{\hat{\beta}}\hat{R}}$.

For a concrete numerical illustration: if $\hat{\beta}_0 = 10$, $\hat{\beta}_1 = -2$, and $\hat{V}_{\hat{\beta}} = \begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix}$, then $\hat{\theta} = -2/10 = -0.2$ and $\hat{R} = (-(-2)/100, \; 1/10)' = (0.02, \; 0.1)'$. So $\hat{V}_{\hat{\theta}} = (0.02, 0.1) \begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix} (0.02, 0.1)' = 0.02^2 \cdot 1 + 2 \cdot 0.02 \cdot 0.1 \cdot 0.3 + 0.1^2 \cdot 1 = 0.0004 + 0.0012 + 0.01 = 0.0116$, giving $s(\hat{\theta}) \approx 0.1077$.

### Linear vs. nonlinear functions: a warning

When $r(\beta)$ is linear — say $\theta = \beta_1 - \beta_2$ — the delta method is exact (no approximation error from the Taylor expansion), and inference tends to be well-behaved.

When $r(\beta)$ is nonlinear — say $\theta = \beta_1/\beta_2$ — the Taylor approximation can be poor in finite samples, especially when $\hat{\beta}_2$ is close to zero. The lectures illustrate a dramatic case: testing $H_0: \beta_1/\beta_2 = 10$ with $\beta_2 = 0.1$ leads to a rejection rate of 44% instead of the nominal 5%. The problem is that when $\hat{\beta}_2$ is near zero, both $|\hat{\theta}|$ and $s(\hat{\theta})$ explode, and they do so in a correlated way that distorts the $t$-statistic.

The solution? Reformulate the null as a linear restriction: instead of $H_0: \beta_1/\beta_2 = 10$, test $H_0: \beta_1 - 10\beta_2 = 0$. This is algebraically equivalent but produces a well-behaved $t$-statistic. Always prefer linear formulations when possible.

---

## IX. Hypothesis Testing

### The general framework

Given $\hat{\theta}$ and $s(\hat{\theta})$, the **$t$-statistic** for testing $H_0: \theta = \theta_0$ is:

$$T(\theta_0) = \frac{\hat{\theta} - \theta_0}{s(\hat{\theta})}.$$

Under $H_0$, using either the normal regression model or large-sample theory:

$$T(\theta_0) \xrightarrow{d} N(0,1) \quad \text{or equivalently} \quad T(\theta_0) \sim t_{n-k} \text{ (exact, under normality + homoskedasticity)}.$$

Since $t_{n-k} \to N(0,1)$ as $n-k \to \infty$, either reference distribution works asymptotically. In practice, we use $t_{n-k}$ because it is more conservative (fatter tails, larger critical values).

### Two-sided tests

For $H_0: \theta = \theta_0$ vs. $H_1: \theta \neq \theta_0$, we reject when $|T| > c_{\alpha/2}$. With $\alpha = 0.05$ and using $N(0,1)$, $c = 1.96$. The $p$-value is:

$$p = 2(1 - \Phi(|T|)).$$

### One-sided tests

For $H_1: \theta > \theta_0$, we reject when $T > c_\alpha$ where $c_{0.05} = 1.645$. For $H_1: \theta < \theta_0$, we reject when $T < -c_\alpha$.

One-sided tests yield smaller $p$-values for the same $T$, which is why they are viewed with suspicion unless the direction of the alternative was specified *before* seeing the data.

### Confidence intervals

The $1-\alpha$ confidence interval is:

$$\hat{C} = \left[\hat{\theta} - c_{\alpha/2} \cdot s(\hat{\theta}), \quad \hat{\theta} + c_{\alpha/2} \cdot s(\hat{\theta})\right].$$

It contains all values $\theta_0$ for which the two-sided test at level $\alpha$ would not reject. The duality between testing and confidence intervals is exact.

### The Wald statistic: joint testing

When $\theta$ is a vector of dimension $q > 1$, we test $H_0: \theta = \theta_0$ using the **Wald statistic**:

$$W = (\hat{\theta} - \theta_0)' \hat{V}_{\hat{\theta}}^{-1} (\hat{\theta} - \theta_0).$$

Under $H_0$ and large-sample theory: $W \xrightarrow{d} \chi^2_q$.

The intuition: $W$ measures the "squared distance" of $\hat{\theta}$ from $\theta_0$, weighted by the inverse of the covariance matrix. The covariance weighting is essential — it accounts for the fact that different components of $\hat{\theta}$ may be estimated with different precision and may be correlated.

When $q = 1$: $W = T^2$, and $\chi^2_1$ is the distribution of a squared standard normal. The Wald test reduces to the squared $t$-test.

### The $F$ version of the Wald test

Divide $W$ by $q$ to get $F = W/q$, and compare to an $F_{q, n-k}$ distribution. When $q=1$, $F_{1, n-k}$ equals $t_{n-k}^2$. The $F$ version is slightly more conservative and is the standard output in most software packages.

For **linear restrictions** $R'\beta = \theta_0$ under homoskedasticity, the $F$-statistic can equivalently be computed as:

$$F = \frac{(SSR_r - SSR_{ur})/q}{SSR_{ur}/(n-k)}$$

where $SSR_r$ and $SSR_{ur}$ are the sums of squared residuals from the restricted and unrestricted models. This is the classical $F$-test from undergraduate econometrics.

### An example: the covariance matrix and joint tests

Consider the model $\ln(w) = \beta_0 + \beta_1 \text{age} + \beta_2 \text{hours} + \beta_3 \text{female} + e$.

The **covariance matrix** $\hat{V}_{\hat{\beta}}$ tells you how each pair of coefficient estimates covaries across samples. To test $H_0: \beta_1 = -\beta_2$ (i.e., $\beta_1 + \beta_2 = 0$), define $\theta = \beta_1 + \beta_2$ and $R' = (0, 1, 1, 0)$. Then:

$$\text{Var}(\hat{\theta}) = R' \hat{V}_{\hat{\beta}} R = \hat{V}_{11} + \hat{V}_{22} + 2\hat{V}_{12}$$

where $\hat{V}_{11} = \text{Var}(\hat{\beta}_1)$, $\hat{V}_{22} = \text{Var}(\hat{\beta}_2)$, and $\hat{V}_{12} = \text{Cov}(\hat{\beta}_1, \hat{\beta}_2)$. The $t$-statistic is:

$$T = \frac{\hat{\beta}_1 + \hat{\beta}_2}{\sqrt{\hat{V}_{11} + \hat{V}_{22} + 2\hat{V}_{12}}}.$$

The sign of the covariance matters: if $\hat{\beta}_1$ and $\hat{\beta}_2$ are positively correlated, the variance of their sum is larger; if negatively correlated, it is smaller. This is why you cannot just compute the standard error of a sum from individual standard errors alone — you need the covariance.

For a **joint test** such as $H_0: \gamma_5 = 0, \gamma_6 = 0$ in a model with interaction terms, you would construct the Wald statistic using the $2 \times 2$ submatrix of $\hat{V}_{\hat{\beta}}$ corresponding to $\hat{\gamma}_5$ and $\hat{\gamma}_6$, and compare $W$ to $\chi^2_2$ (or $F = W/2$ to $F_{2, n-k}$).

---

## X. Clustering

### The problem

So far we have assumed independence across observations: $\text{Cov}(e_i, e_j) = 0$ for $i \neq j$. In many real settings this fails. Students in the same school, workers in the same firm, or residents in the same town may have correlated unobservables. When this correlation exists, treating observations as independent leads to incorrect (usually too small) standard errors.

### The setup

Index observations as $(Y_{ig}, X_{ig})$ where $g = 1, \ldots, G$ denotes clusters and $i = 1, \ldots, n_g$ denotes observations within cluster $g$. The total sample size is $n = \sum_g n_g$.

The OLS estimator is numerically identical whether you think of the data as having $n$ independent observations or $G$ clusters:

$$\hat{\beta} = (X'X)^{-1}X'Y = \left(\sum_g X_g'X_g\right)^{-1}\left(\sum_g X_g'Y_g\right).$$

However, the **variance** changes because the matrix $\Sigma = \text{Var}(e \mid X)$ is no longer diagonal. Within each cluster $g$, we allow arbitrary correlations among the errors; across clusters, we maintain independence. So $\Sigma$ is block-diagonal, with each block being the $n_g \times n_g$ covariance matrix of errors within cluster $g$.

### Cluster-robust variance estimation

The variance formula becomes:

$$V_{\hat{\beta}} = (X'X)^{-1}\Omega_n(X'X)^{-1}, \quad \text{where} \quad \Omega_n = X'\Sigma X = \sum_{g=1}^G X_g' \Sigma_g X_g.$$

The cluster-robust estimator (CR1) replaces $\Sigma_g$ with $\hat{e}_g \hat{e}_g'$ (the outer product of the residual vector within cluster $g$):

$$\hat{V}^{CR1}_{\hat{\beta}} = a_n (X'X)^{-1}\hat{\Omega}_n(X'X)^{-1}$$

where $\hat{\Omega}_n = \sum_g X_g'\hat{e}_g\hat{e}_g'X_g$ and $a_n = \frac{n-1}{n-k} \cdot \frac{G}{G-1}$ is a finite-sample correction.

### When does clustering matter? Intuition

The cross-terms in $\hat{\Omega}_n$ involve products like $X_{ig} X_{lg}' \hat{e}_{ig} \hat{e}_{lg}$ for two observations $i, l$ in the same cluster $g$. These terms are absent from the HC1 formula (which only uses the diagonal $X_{ig}X_{ig}'\hat{e}_{ig}^2$).

Whether clustering *increases* or *decreases* standard errors relative to HC1 depends on the signs of these cross-terms. There are two factors:

1. **Within-cluster correlation of $X$**: Are observations in the same cluster similar in $X$? If treatment is assigned at the cluster level (e.g., school-level policy), then $X_{ig}$ is identical within clusters — maximal positive correlation.

2. **Within-cluster correlation of errors**: Are unobservables correlated within clusters? Typically yes — students in the same school share common unobserved school quality.

If both correlations are positive, the cross-terms are positive, $\hat{\Omega}_n > X'DX$, and clustered SEs are larger than non-clustered SEs. This is the standard case in applied work and is why clustering is so important.

### A powerful special case: treatment at the cluster level

When treatment does not vary within clusters, with equal cluster sizes $N$, homoskedasticity, and within-cluster error correlation $\rho$, the variance simplifies to:

$$V_{\hat{\beta}} = (X'X)^{-1}\sigma^2(1 + \rho(N-1)).$$

The factor $(1 + \rho(N-1))$ is sometimes called the **design effect** or **variance inflation factor (VIF)** due to clustering. If $\rho = 0$ (no within-cluster error correlation) or $N = 1$ (one observation per cluster, so no "cluster" to speak of), there is no inflation. If $\rho = 1$, the effective sample size drops to $n/N = G$, the number of clusters. The message: when treatment varies at the group level, the *number of clusters* — not the number of observations — determines your precision.

### The variance cost of clustering

Just as robust SEs have higher variance than classical SEs, clustered SEs have higher variance than robust SEs. The reason is the same: we are estimating more parameters (the within-cluster covariances) from the same data.

The fewer clusters you have, the worse this problem becomes. With $G = 10$ clusters of size 100 each (total $n = 1000$), simulations show that even when there is *no* within-cluster correlation in the population, the rejection rate of $|T| > 2$ using clustered SEs is about 7.8% instead of 5%. The average SE is slightly below the truth (downward bias from the finite-sample correction), and the variance of the SEs is enormously inflated. With $G = 100$ clusters of size 10, the distortion is much smaller.

This is one of the central lessons: **the number of clusters is crucial**, not the total sample size.

### What level to cluster at?

A deep question. The answer depends on the structure of the DGP — specifically, at what level the correlations in $X$ and $e$ arise. Some key principles:

- **Cluster at the level at which treatment varies.** If a policy is assigned at the school level, cluster at the school level, even if you observe individual students.

- **Cluster at the level at which errors are correlated.** If unobservable shocks are common to all firms in the same industry, you need industry-level clustering.

- **Clustering at too fine a level is insufficient.** If the true correlation is at the firm level but you cluster at the establishment level (where each firm has multiple establishments), you miss the firm-level correlation. The cross-terms between establishments within the same firm are set to zero, and your SEs will be too small.

- **Clustering at too coarse a level wastes statistical power.** If you cluster at the industry level when there are only 4 industries, you have effectively 4 independent observations for estimating the variance — your SEs will be very imprecise (high variance), even if they are unbiased on average.

- **When treatment varies within clusters and errors are independent within clusters** (e.g., a within-school randomization where students in the same school are assigned to treatment or control), clustering can actually *reduce* standard errors relative to robust SEs. The $X$'s are negatively correlated within clusters (if some are treated, others must be control), and if residuals are positively correlated, the product is negative, reducing $\hat{\Omega}_n$.

The problem set asks you to explore this systematically with simulations at multiple clustering levels: employee, establishment, firm, and industry. The key insight is that the "right" level to cluster at depends on which level generates the correlated component in the outcome. If there is a firm-level unobservable $\phi_f$ in the DGP but no industry-level unobservable, then clustering at the firm level is sufficient and clustering at the industry level is unnecessarily conservative. If there is *also* an industry-level unobservable $\delta_g$, you need to cluster at the industry level — but with only 4 industries, you face the "few clusters" problem and your rejection rates will be distorted.

---

## XI. Power, Type M, and Type S Errors

### Statistical power

Power is the probability of rejecting a false null. It is a function of: (1) the true effect size $\theta - \theta_0$; (2) the standard error $s(\hat{\theta})$, which depends on sample size and the variance of $X$; and (3) the significance level $\alpha$.

The asymptotic power formula for a one-sided test $H_1: \theta > \theta_0$ is:

$$\pi(\delta) = 1 - \Phi(c_\alpha - \delta) \quad \text{where} \quad \delta = \frac{\theta - \theta_0}{s(\hat{\theta})}.$$

Here $\delta$ is the "noncentrality parameter" — the true effect measured in units of standard errors. Power increases monotonically in $\delta$. For a two-standard-error effect ($\delta = 2$), one-sided power at $\alpha = 0.05$ is about 64%. For a three-standard-error effect, it is about 91%.

The critical quantity is not just $n$ but $n$ times the *relevant* variation in the regressor. Adding 1000 controls to a sample when you have only 10 treated units does not help nearly as much as adding 10 treated units.

### The precision allocation problem

Suppose you are running an experiment and can add either 10 treated units or 100 controls. Under homoskedasticity, the variance of the treatment-control difference in a regression $Y = \beta_0 + \beta_1 D + e$ is:

$$\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{n \cdot \hat{p}(1 - \hat{p})}$$

where $\hat{p} = n_T/n$ is the treatment share. This is minimized when $\hat{p} = 0.5$. Starting from 100 treated and 300 controls ($n = 400$, $\hat{p} = 0.25$), adding 10 treated units gives $n = 410$, $\hat{p} = 110/410 \approx 0.268$, while adding 100 controls gives $n = 500$, $\hat{p} = 100/500 = 0.20$. You should compute $n \cdot \hat{p}(1 - \hat{p})$ for each case to determine which is more precise. Intuitively, you are far from 50-50 balance, so adding treated units (moving toward balance) is likely to help more than adding controls (moving further from balance), even though you add fewer units.

Under **heteroskedasticity**, the answer depends on the relative error variances in the treated and control groups. If $\sigma_T^2 \neq \sigma_C^2$, the optimal allocation shifts toward oversampling the group with higher variance.

### Type M and Type S errors

When power is low, statistically significant results are misleading in two ways:

**Type M (magnitude) error**: To achieve $|T| > 2$ with a large $s(\hat{\theta})$, you need $|\hat{\theta}|$ to be large. So among the significant results, $|\hat{\theta}|$ systematically overstates $|\theta|$. In simulations with $\beta = 1$, $\sigma_e = 10$, and $n = 10$, the average $|\hat{\beta}|$ conditional on $p < 0.05$ is about 7 — a sevenfold exaggeration.

**Type S (sign) error**: With enough noise, $\hat{\beta}$ can be significantly positive even though $\beta > 0$, or significantly negative even though $\beta > 0$. At $n = 10$ in the same simulation, about 25% of significant results have the wrong sign.

Both problems vanish as power increases: with $n = 500$, the Type M ratio approaches 1 and Type S probability approaches 0.

### Publication bias

If journals preferentially publish statistically significant results, the literature becomes a biased sample of all studies. Low-powered studies that happen to reach significance will report inflated effects (Type M) and sometimes wrong signs (Type S). High-powered studies are both more likely to be published and more accurate when they are. This is a systemic problem in empirical social science.

---

## XII. Multiple Hypothesis Testing

When testing $k$ hypotheses simultaneously, all of which are true nulls, the probability that *at least one* is rejected at level $\alpha$ is much higher than $\alpha$. Under independence of the tests, this probability is $1 - (1-\alpha)^k$. For $k = 10$ and $\alpha = 0.05$, this is about 40%.

### Bonferroni correction

Multiply each $p$-value by $k$ (or equivalently, use $\alpha/k$ as the threshold). This controls the **familywise error rate (FWER)** — the probability of any false rejection — at level $\alpha$. It is simple but conservative, especially when tests are positively correlated.

### Bonferroni-Holm

A step-down improvement: sort $p$-values from smallest to largest as $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$. Test $p_{(j)}$ against $\alpha/(m + 1 - j)$. Reject all hypotheses up to the first non-rejection. This is uniformly more powerful than Bonferroni.

### False Discovery Rate (FDR)

Instead of controlling the probability of *any* false rejection, the **Benjamini-Hochberg** procedure controls the *expected proportion* of false rejections among all rejections. This is less conservative: you accept some false positives in exchange for many more true positives.

The procedure: sort $p$-values, and find the largest $k$ such that $p_{(k)} \leq \frac{k}{m}q$ where $q$ is the target FDR. Reject hypotheses $1, \ldots, k$.

The intuition for why BH is more lenient at higher ranks: a single small $p$-value among many tests is unsurprising (even under all true nulls), but *many* small $p$-values are strong evidence that some nulls are false.

---

## XIII. Synthesis: How It All Fits Together

Let me pull back and connect the major threads.

The entire enterprise of inference rests on knowing — or at least approximating — the **distribution of estimators**. We have two routes to this distribution: assuming normality of the errors (giving exact finite-sample results but requiring strong assumptions) or invoking the CLT (giving approximate results under weak assumptions but requiring a "large enough" sample).

In either case, the mechanics of testing are the same: estimate $\hat{\theta}$, compute $s(\hat{\theta})$ using the appropriate variance formula, form $T = (\hat{\theta} - \theta_0)/s(\hat{\theta})$, and compare to the relevant reference distribution ($t_{n-k}$ or $N(0,1)$).

The choice of variance formula — classical, HC1, HC2, HC3, or cluster-robust — is not just a computational detail. It reflects your assumptions about the error structure: how error variances relate to the regressors, and whether errors are correlated across observations. Getting this wrong does not bias your point estimates, but it makes your standard errors wrong, which in turn makes your $p$-values and confidence intervals wrong. And since these are what drive inference and policy conclusions, the stakes are high.

For functions of parameters, the delta method provides the bridge: given the asymptotic distribution of $\hat{\beta}$, it gives you the asymptotic distribution of $r(\hat{\beta})$. But beware of nonlinear functions in small samples — the Taylor approximation can fail, and reformulating hypotheses as linear restrictions is almost always safer.

For joint tests, the Wald statistic generalizes the $t$-test and accounts for covariances among the estimated parameters. The covariance matrix is not a nuisance — it is essential information that determines how to weight evidence from multiple coefficients.

For clustering, the key message is that the effective sample size for inference is determined not by the total number of observations, but by the number of independent units at the level where both the regressor and the unobservables exhibit correlation. With few clusters, even correctly specified cluster-robust SEs have high variance and produce distorted rejection rates.

Finally, power is not a luxury — it determines whether your study can produce reliable evidence. Low power does not just mean "failure to reject"; it means that when you do reject, your estimates are likely to be exaggerated and may even have the wrong sign.

All of these topics interact. The simulation-based questions in the problem set are designed to make you *see* these interactions by watching them unfold in artificial data where you know the truth. The applied questions ask you to wield these tools on real data, where you must make judgment calls about the appropriate error structure and the reliability of your inference.
