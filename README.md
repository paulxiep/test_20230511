## How to use

1. Trim the transaction tables according to [trim_transactions.py](trim_transactions.py)

2. Use the following additional tables (equivalent tables from other years haven't been tested, may work, may not.)

    - income: ACSST5Y2010.S1903
    - demographics: ACSDP5Y2010.DP05
    - residents_commute: ACSDT5Y2010.B08141
    - workers_commute: ACSDT5Y2010.B08541

3. Obtain the tables by entering above code as search term [on US census bureau page here](https://data.census.gov/)

It is very likely that the provided cloud app doesn't have enough (1 GB) memory, 

so it is recommended you clone the repo to run locally