# credit_card_customer_segmentation
Develop a customer segmentation to define market strategy. The sample dataset summarizes the usage behaviour of about 9000 active credit card holders during the last 6 months.
## Objectives Of Data Analysis:
### Advanced data preparation: Build an ‘enriched’ customer profile by deriving “intelligent” KPIs such as:
1- Monthly average purchase and cash advance amount

2- Purchases by type (one-off, installments)

3- Average amount per purchase and cash advance transaction,

4- Limit usage (balance to credit limit ratio),

5- Payments to minimum payments ratio etc.

6- Advanced reporting: Use the derived KPIs to gain insight on the customer profiles.

7- Identification of the relationships/ affinities between services.

8- Clustering: Apply a data reduction technique factor analysis for variable reduction technique and a clustering algorithm to reveal the behavioural segments of credit card holders

9- Identify cluster characterisitics of the cluster using detailed profiling.

10- Provide the strategic insights and implementation of strategies for given set of cluster characteristics

Data Dictionary-
1- CUST_ID : Identification of Credit Card holder (Categorical)
2- BALANCE : Balance amount left in their account to make purchases (
3- BALANCE_FREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
4- PURCHASES : Amount of purchases made from account
5- ONEOFF_PURCHASES : Maximum purchase amount done in one-go
6- INSTALLMENTS_PURCHASES : Amount of purchase done in installment
7- CASH_ADVANCE : Cash in advance given by the user
8- PURCHASES_FREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
9- ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
10- PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
11- CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
12- CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
13- PURCHASES_TRX : Numbe of purchase transactions made
14- CREDIT_LIMIT : Limit of Credit Card for user
15- PAYMENTS : Amount of Payment done by user
16- MINIMUM_PAYMENTS : Minimum amount of payments made by user
17- PRCFULLPAYMENT : Percent of full payment paid by user
18- TENURE : Tenure of credit card service for user
