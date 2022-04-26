# auto_trading
Auto Trading (DSAI HW2)

## Flow
1. Train LSTM model
2. Predict next day open price
3. Determine action based on predicted price and current price

## Model
One-to-one LSTM model, using early stopping to avoid overfitting.

## Strategy
1. 預測之股價高於目前股價
      1. 目前沒有持股：買入，做多
      1. 目前有持股：不動作，持續做多
      1. 目前有借券：買入，回補空單 
2. 預測之股價低於目前股價
      3. 目前沒有持股：賣出，放空
      3. 目前有持股：不動作，持續放空
      3. 目前有持股：賣出
