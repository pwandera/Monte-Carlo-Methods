# Monte-Carlo-Methods
Generating future price movements of different stocks and their return distributions.

### Procedures:  
-> yfinance is used to import stock prices  
-> The σ and μ Log Change of the stock price is captured using numpy  
-> A t-distribution is used to account for fat tails in financial markets using scipy and statsmodels  
-> Price paths are drawn using log changes grabbed from the t-distribution with df = 5  
-> Returns of each path are plotted to infer an expected value of longing the stock  
-> Visual plots are generated using seaborn and matplotlib methods  
