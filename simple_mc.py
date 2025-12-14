import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
sns.set_theme(style="whitegrid", palette="deep")

def MC(TICKER="MSFT", DAYS_FORWARD=200, DAYS_BACKWARD=50, SIMS=200) -> None:
    s = yf.download(tickers=TICKER, period="10y", auto_adjust=True).Close
    s.columns = ["Close"]

    s["Log Change"] = np.log(s["Close"] / s["Close"].shift(1))
    log_mean = s["Log Change"].mean()
    log_std = s["Log Change"].std()

    # PLOT DISTRIBUTION OF LOG RETURNS
    sns.kdeplot(data=s["Log Change"], fill=True)
    plt.title(f"{TICKER} DENSITY PLOT FOR 2015-2025")
    plt.ylabel("DENSITY")
    plt.xlabel("LOG CHANGE")
    plt.show()

    print("\n-----------------------------------------------------------------------------------\n")

    # QQ-PLOT TO INFER DISTRIBUTION OF LOG RETURNS
    z = (s["Log Change"] - log_mean) / log_std  # get z-scores of sampling
    sm.qqplot(z, scipy.stats.t, distargs=(5,), line="45")  # compare with standard distribution
    plt.title("QQ-PLOT")
    plt.show()

    print("\n-----------------------------------------------------------------------------------\n")

    start_price = s.iloc[-1].iloc[0]

    pre = []
    for i in range(s["Close"].shape[0] - 1 - DAYS_BACKWARD, s["Close"].shape[0] - 1):
        pre.append(s["Close"].iloc[i])

    # ADD PLOT OF SECURITY'S PAST TREND
    sns.lineplot(x=np.arange(-DAYS_BACKWARD, 0), y=pre)

    long_returns = []

    for i in range(SIMS):
        price = [start_price]
        time = [0]

        for i in range(1, DAYS_FORWARD + 1):
            # change = np.random.standard_t(loc = log_mean, scale = log_std)

            # STUDENT-T DISTRIBUTION TO MODEL FAT TAILS FOUND MADE BY FINANCIAL SHOCKS
            change = scipy.stats.t.rvs(df=5, loc=log_mean, scale=log_std)
            y = price[-1] * np.exp(change)
            price.append(y)
            time.append(i)

        # COLLECT SIMPLE RETURN FOR EACH PATH
        long_returns.append((price[-1] / price[0]) - 1)

        # ADD DIFFERENT TIMELINES TO THE PLOT
        sns.lineplot(x=time, y=price)

    plt.title(f"{DAYS_FORWARD}-DAY {TICKER} MC FORECAST")
    plt.xlabel("DAYS")
    plt.ylabel("PRICE ($)")
    plt.show()

    print("\n-----------------------------------------------------------------------------------\n")

    # PLOT DISTRIBUTION OF SIMPLE RETURNS
    sns.histplot(data=long_returns, kde=True, color="m")
    plt.axvline(x=0, color="r", lw=2.5)
    plt.title(f"DISTRIBUTION OF SIMPLE RETURNS ON DAY {DAYS_FORWARD}")
    plt.ylabel("COUNT")
    plt.xlabel("FINAL RETURN")
    plt.show()

MC()
