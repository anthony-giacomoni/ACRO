# Model and contract notes

These notes describe the assumptions behind the demo. They are not a market term sheet.

## Reference contract

The starting example uses a Phoenix on three equities, with a worst-of reference basket and €100 nominal. The note has a maximum maturity of five years, semiannual conditional coupons without memory, a 70% coupon barrier, annual autocall observations in years 1–4 at 100% and a 60% European principal barrier.

On an observation date, the same normalized basket reference determines whether a coupon is paid or an early call happens. At maturity, if the protection barrier is breached, the redemption follows the reference performance (capped at 100%); otherwise the principal is repaid in full, subject to issuer credit risk which is not modelled.

ACRO is purchased separately. If the original note calls on an eligible date, the investor can buy a *new* Phoenix at a pre-agreed strike and with the original conditional coupon rate. Its references are reset to 100% on the renewal date. Otherwise no renewal option exists. The right is not exercised when the new note's model value is below the strike.

The option payoff at first eligible call $\tau$ is

$$
\mathbf{1}_{\{\tau<T\}}\,u\max\left(V_{\rm new}(\tau;c_0)-K,0\right).
$$

Here $u$ is the number of renewal notes, $K$ is the subscription price per note and $V_{\rm new}$ is the conditional model value at that time. ACRO's price today is the discounted risk-neutral expectation of this payoff.

## Market model

The base case simulates correlated equity log returns under the risk-neutral measure with constant short rate and dividend yields. Each stock has a base volatility multiplied by a *shared* exponential Ornstein–Uhlenbeck log-volatility factor. Equity and volatility shocks may be negatively correlated. The model is synthetic and is not calibrated to an implied volatility surface.

The code also contains an optional annual stochastic-correlation experiment. It is separate from the main GUI pricer, exploratory and not calibrated to market correlation data.

## Coupon and renewal pricing

For fixed barriers and observation rules, the Phoenix price can be written as $V=B+cA$: discounted principal payments $B$ plus an exotic conditional-coupon annuity $A$ times annual coupon rate $c$. The par coupon is found from $(N-B)/A$ when $A>0$.

The outer simulation draws the initial Phoenix paths and detects eligible autocalls. Conditional simulations estimate the renewal note's principal/coupon components under the market state at call. A conditional curve with interpolation reduces the cost of evaluating many outer paths. The exercise decision then uses the positive part of the renewed Phoenix value minus the contract strike.

## Basket selection

The number of equities is configurable from one to six. Reference choices are:

- **Worst-of:** minimum normalized equity performance.
- **Best-of:** maximum normalized equity performance.
- **Equally weighted:** average of normalized equity performances, not average raw stock prices.

With one underlying, these choices are identical. The initial and renewed notes may use different reference rules, though that changes the renewal contract rather than simply reproducing the original note. Each simulated market correlation matrix is checked for positive definiteness.

## Important limitations

The model does not include issuer credit/funding terms, distribution fees or a market hedging programme. The shared volatility factor is not assumed perfectly hedgeable. Output standard errors based on outer paths do not capture inner-simulation error, interpolation bias, time-grid bias or model risk. Sensitivities reflect the assumed model and shock definitions, not a universal ranking of economic risks.
