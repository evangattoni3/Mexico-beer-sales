
#===============================================
# Time Series Forecasting - Evan Gattoni
# Aggregate Daily Beer Sales (All Cities, All Products)
# Models on log scale + NEW seasonal ARIMA model
#===============================================

data <- Mexico.beer.data   

################################################
# 1. Data cleaning and aggregation
################################################

data_clean <- data

# Parse date: day/month/year (e.g. "1/1/2013")
data_clean$date  <- as.Date(data_clean$date, format = "%d/%m/%Y")

# Ensure sales numeric
data_clean$sales <- as.numeric(data_clean$sales)

# Drop rows with missing/invalid date or sales
data_clean <- data_clean[is.finite(data_clean$sales) & !is.na(data_clean$date), ]

# Aggregate sales over ALL cities and ALL beers by date
agg <- aggregate(sales ~ date, data = data_clean, FUN = sum)
agg <- agg[order(agg$date), ]

# Raw and log series
y_agg_raw <- agg$sales              # original scale
n_agg     <- length(y_agg_raw)
cat("Aggregate series length (days):", n_agg, "\n")

# Log transform (avoid log(0))
y_agg <- log(pmax(y_agg_raw, 1))

################################################
# 2. Helper functions: rolling evaluation & metrics
################################################

# Rolling one-step-ahead errors:
#   - y: series (numeric)
#   - initial_window: size of first training set
#   - forecast_fun: function(y_train) -> scalar forecast
rolling_errors <- function(y, initial_window, forecast_fun) {
  n <- length(y)
  err <- rep(NA_real_, n)
  if (initial_window >= n) return(err)
  
  for (t in initial_window:(n - 1)) {
    y_train <- y[1:t]
    y_hat   <- forecast_fun(y_train)
    err[t + 1] <- y[t + 1] - y_hat
  }
  err
}

mse <- function(e) mean(e^2, na.rm = TRUE)
mae <- function(e) mean(abs(e), na.rm = TRUE)

################################################
# 3. Plot aggregate series (raw scale, date axis)
################################################

plot(agg$date, y_agg_raw / 1e6, type = "l", lwd = 1.2,
     main = "Aggregate Daily Sales (All Cities, All Beers)",
     xlab = "Date", ylab = "Total sales (millions)")
grid()

# Optional: 30-day moving average to highlight trend/seasonality
lines(agg$date,
      stats::filter(y_agg_raw / 1e6, rep(1/30, 30), sides = 2),
      col = "red", lwd = 2)
legend("topleft",
       legend = c("Daily sales", "30-day moving average"),
       col = c("black", "red"), lwd = c(1.2, 2), bty = "n")

################################################
# 4. Model definitions (all on log scale)
################################################

#-------------------------
# Model 1: ARIMA(1,0,1)
#-------------------------
fit_forecast_arima <- function(y_train) {
  fit <- try(arima(y_train, order = c(1, 0, 1)), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  pr <- predict(fit, n.ahead = 1)$pred[1]
  as.numeric(pr)
}

#-------------------------
# Model 2: Simple exponential smoothing (manual)
#-------------------------
simple_exp_smooth <- function(y, alpha = 0.3) {
  level <- y[1]
  if (length(y) > 1) {
    for (i in 2:length(y)) {
      level <- alpha * y[i] + (1 - alpha) * level
    }
  }
  level         # forecast next value
}

fit_forecast_ets <- function(y_train) {
  simple_exp_smooth(y_train, alpha = 0.3)
}

#-------------------------
# Model 3: Local Level (StructTS)
#-------------------------
fit_forecast_local_level <- function(y_train) {
  fit <- try(StructTS(y_train, type = "level"), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  pr <- predict(fit, n.ahead = 1)$pred[1]
  as.numeric(pr)
}

#-------------------------
# Model 4: Discounted local-level DLM
#         y_t = mu_t + e_t
#         mu_t = mu_{t-1} + w_t,  Var(w_t) controlled by delta
#-------------------------
discount_dlm <- function(y, delta = 0.95, v0 = 1) {
  n <- length(y)
  m <- numeric(n)
  C <- numeric(n)
  
  # prior
  m[1] <- y[1]
  C[1] <- v0
  
  for (t in 2:n) {
    # prior
    a <- m[t - 1]
    R <- C[t - 1] / delta
    
    # forecast
    f <- a
    Q <- R + v0
    
    # update
    e <- y[t] - f
    A <- R / Q
    
    m[t] <- a + A * e
    C[t] <- (1 - A) * R
  }
  
  # one-step forecast for time n+1
  a_next <- m[n]
  R_next <- C[n] / delta
  f_next <- a_next   # mean forecast for y_{n+1}
  as.numeric(f_next)
}

fit_forecast_dlm <- function(y_train, delta = 0.95) {
  discount_dlm(y_train, delta = delta)
}

#-------------------------
# Model 5: Seasonal ARIMA (manual seasonal differencing)
#         Seasonality = yearly (365 days)
#         Fit ARIMA(1,0,1) to seasonally differenced data
#-------------------------
fit_forecast_sarima <- function(y_train, season = 365) {
  n <- length(y_train)
  if (n <= season + 5) return(NA_real_)  # not enough data
  
  # Seasonal differencing
  ds <- y_train[(season + 1):n] - y_train[1:(n - season)]
  
  fit <- try(arima(ds, order = c(1, 0, 1), include.mean = TRUE),
             silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  
  # Forecast next differenced value
  pr_diff <- predict(fit, n.ahead = 1)$pred[1]
  
  # Reconstruct forecast on original (log) scale:
  # y_{n+1} = y_{n+1-season} + ds_{n+1}
  yhat <- y_train[n + 1 - season] + pr_diff
  as.numeric(yhat)
}

################################################
# 5. Rolling setup (log scale)
################################################

# Use about half the data, but at least 365 days, as initial window
initial_window_agg <- max(365, floor(0.5 * n_agg))
if (initial_window_agg >= n_agg) {
  initial_window_agg <- floor(0.7 * n_agg)
}
cat("Initial rolling window (days):", initial_window_agg, "\n")
cat("Out-of-sample forecasts:", n_agg - initial_window_agg, "\n")

################################################
# 6. Run models on log(y_agg)
################################################

# 1) ARIMA(1,0,1)
err_log_arima <- rolling_errors(y_agg, initial_window_agg,
                                fit_forecast_arima)

# 2) Simple exponential smoothing
err_log_ets   <- rolling_errors(y_agg, initial_window_agg,
                                fit_forecast_ets)

# 3) Local level
err_log_ll    <- rolling_errors(y_agg, initial_window_agg,
                                fit_forecast_local_level)

# 4) Discount DLM for several deltas
deltas <- c(0.8, 0.9, 0.95, 0.99)
dlm_rows <- list()
for (d in deltas) {
  e_d <- rolling_errors(
    y_agg, initial_window_agg,
    function(z) fit_forecast_dlm(z, delta = d)
  )
  dlm_rows[[length(dlm_rows) + 1]] <- data.frame(
    Model = paste0("Discount DLM (delta=", d, ")"),
    MSE   = mse(e_d),
    MAE   = mae(e_d),
    stringsAsFactors = FALSE
  )
}
dlm_table <- do.call(rbind, dlm_rows)

# 5) Seasonal ARIMA with yearly seasonal difference
err_log_sarima <- rolling_errors(
  y_agg, initial_window_agg,
  fit_forecast_sarima
)

################################################
# 7. Summary table (log scale)
################################################

agg_model_comparison_log <- rbind(
  data.frame(
    Model = "ARIMA(1,0,1) on log(total sales)",
    MSE   = mse(err_log_arima),
    MAE   = mae(err_log_arima),
    stringsAsFactors = FALSE
  ),
  data.frame(
    Model = "Simple Exp. Smoothing (alpha=0.3) on log(total sales)",
    MSE   = mse(err_log_ets),
    MAE   = mae(err_log_ets),
    stringsAsFactors = FALSE
  ),
  data.frame(
    Model = "Local Level (StructTS) on log(total sales)",
    MSE   = mse(err_log_ll),
    MAE   = mae(err_log_ll),
    stringsAsFactors = FALSE
  ),
  data.frame(
    Model = "Seasonal ARIMA(1,0,1) with yearly diff (manual)",
    MSE   = mse(err_log_sarima),
    MAE   = mae(err_log_sarima),
    stringsAsFactors = FALSE
  ),
  dlm_table
)

cat("\n=== SECTION 1: Aggregate Model Comparison (Log Scale) ===\n")
print(agg_model_comparison_log, row.names = FALSE)
