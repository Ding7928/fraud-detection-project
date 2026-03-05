
-- Basic metrics
.headers on
.mode column

SELECT
  COUNT(*) AS total_transactions,
  SUM(Class) AS fraud_transactions,
  ROUND(SUM(Class) * 1.0 / COUNT(*), 6) AS fraud_rate
FROM transactions;

-- Fraud rate by hour bucket (Time is seconds since first transaction)
SELECT
  CAST(Time / 3600 AS INT) AS hour_bucket,
  COUNT(*) AS transactions,
  SUM(Class) AS frauds,
  ROUND(SUM(Class) * 1.0 / COUNT(*), 6) AS fraud_rate
FROM transactions
GROUP BY hour_bucket
ORDER BY hour_bucket;

-- High amount transactions and fraud rate
SELECT
  CASE
    WHEN Amount < 10 THEN '<10'
    WHEN Amount < 50 THEN '10-50'
    WHEN Amount < 200 THEN '50-200'
    WHEN Amount < 1000 THEN '200-1000'
    ELSE '>=1000'
  END AS amount_bucket,
  COUNT(*) AS transactions,
  SUM(Class) AS frauds,
  ROUND(SUM(Class) * 1.0 / COUNT(*), 6) AS fraud_rate
FROM transactions
GROUP BY amount_bucket
ORDER BY
  CASE amount_bucket
    WHEN '<10' THEN 1
    WHEN '10-50' THEN 2
    WHEN '50-200' THEN 3
    WHEN '200-1000' THEN 4
    ELSE 5
  END;
