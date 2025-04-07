# ===============================
# 7089CEM Assignment R Code
# ===============================

# Load required libraries
library(readr)
library(ggplot2)

# Load datasets
X <- read_csv("D:/X.csv")
y <- read_csv("D:/y.csv")$y
time <- read_csv("D:/time.csv")$time

# ===============================
# Task 1: Preliminary Data Analysis
# ===============================

# 1.1 Time series plots
plot(time, X$x1, type = "l", col = "blue", main = "Audio Signal x1 over Time", xlab = "Time", ylab = "x1")
plot(time, y, type = "l", col = "red", main = "MEG Signal y over Time", xlab = "Time", ylab = "y")

# 1.2 Distributions
hist(X$x1, main = "Histogram of x1", col = "lightblue")
hist(y, main = "Histogram of y", col = "pink")

# 1.3 Scatter plot and correlation
plot(X$x1, y, main = "Scatter Plot of x1 vs y", xlab = "x1", ylab = "y")
cat("Correlation (x1, y):", cor(X$x1, y), "\n")

# 1.4 Boxplot of y by x2
boxplot(y ~ X$x2, main = "Boxplot of y by x2 Category", names = c("Neutral", "Emotional"), col = c("lightgray", "tomato"))

# ===============================
# Task 2: Regression – Nonlinear Models
# ===============================

# Add polynomial features
X$x1_squared <- X$x1^2
X$x1_cubed <- X$x1^3
X$x1_fourth <- X$x1^4
X$x1_quintic <- X$x1^5
X$x1_x2 <- X$x1 * X$x2

# Combine with target
data <- cbind(X, y)

# Define models
models <- list(
  "Model 1" = lm(y ~ x1_cubed + x1_quintic + x2, data = data),
  "Model 2" = lm(y ~ x1 + x2, data = data),
  "Model 3" = lm(y ~ x1 + x1_squared + x1_fourth + x2, data = data),
  "Model 4" = lm(y ~ x1 + x1_squared + x1_cubed + x1_quintic + x2, data = data),
  "Model 5" = lm(y ~ x1 + x1_cubed + x1_fourth + x2, data = data)
)

# 2.1 Model summaries
for (name in names(models)) {
  cat("\n==========", name, "==========\n")
  print(summary(models[[name]]))
}

# 2.2 RSS
rss_vals <- sapply(models, function(m) sum(resid(m)^2))

# 2.3 Log-Likelihood
n <- nrow(data)
loglik_vals <- sapply(models, function(m) {
  sigma2 <- sum(resid(m)^2) / (n - 1)
  -0.5 * n * log(2 * pi) - 0.5 * n * log(sigma2) - 0.5 * sum(resid(m)^2) / sigma2
})

# 2.4 AIC & BIC
aic_vals <- sapply(models, AIC)
bic_vals <- sapply(models, BIC)

# 2.5 Residual analysis
for (name in names(models)) {
  model <- models[[name]]
  qqnorm(resid(model), main = paste("Q-Q Plot:", name))
  qqline(resid(model), col = "red")
  hist(resid(model), main = paste("Histogram of Residuals:", name), col = "lightgreen", xlab = "Residuals")
}

# 2.6 Model comparison
cat("\n========== Model Comparison Table ==========\n")
compare_df <- data.frame(
  Model = names(models),
  RSS = rss_vals,
  LogLikelihood = loglik_vals,
  AIC = aic_vals,
  BIC = bic_vals
)
print(compare_df)

# ===============================
# Task 2.7: Train/Test Split and CI
# ===============================
set.seed(123)
train_idx <- sample(1:n, size = round(0.7 * n))
test_idx <- setdiff(1:n, train_idx)
train_data <- data[train_idx, ]
test_data <- data[test_idx, ]

# Refit best model (e.g., Model 4) on training data
best_model <- lm(y ~ x1 + x1_squared + x1_cubed + x1_quintic + x2, data = train_data)
pred <- predict(best_model, newdata = test_data, interval = "confidence", level = 0.95)

# Plot prediction with confidence intervals
plot(test_data$y, type = "l", col = "black", ylim = range(c(test_data$y, pred)), main = "Prediction with 95% CI")
lines(pred[, "fit"], col = "blue")
lines(pred[, "lwr"], col = "red", lty = 2)
lines(pred[, "upr"], col = "red", lty = 2)
legend("topright", legend = c("Actual", "Prediction", "95% CI"), col = c("black", "blue", "red"), lty = c(1,1,2))

# ===============================
# Task 3: Rejection ABC (Top 2 Parameters)
# ===============================

# Extract top 2 largest-magnitude coefficients from best model
theta_hat <- coef(best_model)
top2 <- sort(abs(theta_hat), decreasing = TRUE)[1:2]
top2_names <- names(top2)

# Generate prior samples
prior_samples <- 10000
prior <- data.frame(
  theta1 = runif(prior_samples, min = theta_hat[top2_names[1]] - 2, max = theta_hat[top2_names[1]] + 2),
  theta2 = runif(prior_samples, min = theta_hat[top2_names[2]] - 2, max = theta_hat[top2_names[2]] + 2)
)

# Rejection criteria (simplified)
accepted <- prior[
  abs(prior$theta1 - theta_hat[top2_names[1]]) < 0.5 &
    abs(prior$theta2 - theta_hat[top2_names[2]]) < 0.5,
]

# Posterior plots
par(mfrow = c(1, 2))
hist(accepted$theta1, main = paste("Posterior of", top2_names[1]), col = "skyblue", xlab = top2_names[1])
hist(accepted$theta2, main = paste("Posterior of", top2_names[2]), col = "salmon", xlab = top2_names[2])


# Load required libraries
library(readr)

# Load data
X <- read_csv("D:/X.csv")
y <- read_csv("D:/y.csv")$y

# Add polynomial and interaction terms
X$x1_squared <- X$x1^2
X$x1_cubed <- X$x1^3
X$x1_fourth <- X$x1^4
X$x1_quintic <- X$x1^5
X$x1_x2 <- X$x1 * X$x2

# Merge y with input features
data <- cbind(X, y)

### ---------- Model 1 ----------
cat("========== Model 1 ==========\n")
model1 <- lm(y ~ x1_cubed + x1_quintic + x2, data = data)
print(summary(model1))

### ---------- Model 2 ----------
cat("\n========== Model 2 ==========\n")
model2 <- lm(y ~ x1 + x2, data = data)
print(summary(model2))

### ---------- Model 3 ----------
cat("\n========== Model 3 ==========\n")
model3 <- lm(y ~ x1 + x1_squared + x1_fourth + x2, data = data)
print(summary(model3))

### ---------- Model 4 ----------
cat("\n========== Model 4 ==========\n")
model4 <- lm(y ~ x1 + x1_squared + x1_cubed + x1_quintic + x2, data = data)
print(summary(model4))

### ---------- Model 5 ----------
cat("\n========== Model 5 ==========\n")
model5 <- lm(y ~ x1 + x1_cubed + x1_fourth + x2, data = data)
print(summary(model5))

### ---------- Model Comparison Table ----------
cat("\n========== Model Comparison ==========\n")
models <- list("Model 1" = model1, "Model 2" = model2, "Model 3" = model3, 
               "Model 4" = model4, "Model 5" = model5)

compare_df <- data.frame(
  Model = names(models),
  AIC = sapply(models, AIC),
  BIC = sapply(models, BIC),
  RSS = sapply(models, function(m) sum(resid(m)^2))
)

print(compare_df)
