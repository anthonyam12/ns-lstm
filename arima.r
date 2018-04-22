run <- function(train, test) {
    library(forecast)
    mdl <- auto.arima(train)
    hist <- train
    hist <- hist[-1]
    predictions <- c()
    for(i in 1:length(test)) {
        print(i)
        optim.control = list(maxit = 10000)
        mdl <- arima(as.matrix(hist), c(1,0,2))
        predictions <- c(predictions, forecast(mdl, 1)$mean[[1]])
        hist <- c(hist, test[i])
        hist <- hist[-1]
    }
    # print(predictions)
    mse <- sum((predictions- test)^2)
    mse <- mse/length(predictions)
    mae <- sum((predictions- test))
    mae <- mae/length(predictions)
    print(mse)
    print(mae)
    return(predictions)
}
