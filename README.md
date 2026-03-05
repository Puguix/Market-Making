# Market-Making
Market Making project for Electronic Markets course @ 203


Ideas : le mid A est le VWAP de B (75%) et C (25%)

# 1. Design carnet d'ordre


# 2. Arrivage d'ordres

## A. Market ordres design

- créer class Market Order (enum OrderType ?)
- implémenter une méthode "aks hit" / "bid hit"
- design intensity arrival function with different methods : exp or power law


# 3. Market Maker Agent

- build inventory property and cash property
- utility problem
- bid-ask optim and order book adjustment (cancelation, new limit order etc).