from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd

from pypfopt.expected_returns import returns_from_prices
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

app = Flask(__name__)

STOCK_CATEGORIES = {
    "Aggressive": ["TSLA", "ADANIENT.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "RELIANCE.NS", 
                   "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "AFFLE.NS", "NVDA"],
    "Moderate": ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "WIPRO.NS", "TCS.NS",
                 "HCLTECH.NS", "SUNPHARMA.NS", "CIPLA.NS", "LT.NS", "BHARTIARTL.NS"],
    "Conservative": ["AAPL", "MSFT", "WMT", "KO", "PG", "JNJ", "XOM", "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]
}

@app.route("/allocate_portfolio", methods=["POST"])
def allocate_portfolio():
    data = request.get_json()
    category = data.get("category", "Moderate")
    investment_amount = data.get("investment_amount", 50000)

    if category not in STOCK_CATEGORIES:
        return jsonify({"error": "Invalid category. Choose from 'Aggressive', 'Moderate', or 'Conservative'."}), 400

    assets = STOCK_CATEGORIES[category]
    
    # Fetch stock data from Yahoo Finance
    prices_df = yf.download(assets, start="2023-01-01", end="2023-12-31")

    if "Adj Close" in prices_df:
        prices_df = prices_df["Adj Close"]
    elif "Close" in prices_df:
        prices_df = prices_df["Close"]
    else:
        return jsonify({"error": "No 'Adj Close' or 'Close' data available for selected stocks."}), 400

    rtn_df = returns_from_prices(prices_df)

    # Portfolio optimization
    hrp = HRPOpt(returns=rtn_df)
    hrp.optimize()
    weights = hrp.clean_weights()

    # Portfolio performance
    expected_return, volatility, sharpe_ratio = hrp.portfolio_performance()

    # Stock allocations
    latest_prices = get_latest_prices(prices_df)
    allocation_finder = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = allocation_finder.lp_portfolio()

    portfolio_performance = []
    portfolio_performance.append(expected_return*100)
    portfolio_performance.append(volatility*100)
    portfolio_performance.append(sharpe_ratio)  

    # Prepare the response, ensuring data is serializable
    response = {
        "investment_amount": investment_amount,
        "portfolio_performance": portfolio_performance,
        "leftover_cash": leftover
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
