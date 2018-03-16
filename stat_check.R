run <- function() {
    #
    # Plots the variances of different chunks of data. Maybe not useful...
    #
    sunspots <- read.csv("./data/Sunspots.csv")
    data_chunks <- c()
    chunk_size <- 200
    chunks = as.integer((length(sunspots[, 2])/chunk_size))
    for (i in 0:(chunks-1)) {
        start = (i*chunk_size) + 1
        end = start + chunk_size - 1
        data_chunks <- c(data_chunks, list(sunspots[start:end, 2]))
    }
    data_chunks <- c(data_chunks, list(sunspots[(chunks*chunk_size):length(sunspots[,2]), 2]))
    variances <- c()
    for(i in 1:length(data_chunks)) {
        variances <- c(variances, var(data_chunks[[i]]))
    }
    plot(variances, type='l')
}
 
rolling_average <- function(data, chunk_size=100) {
    # Rolling mean plot
    
    library(zoo)
    plot(rollmean(data, chunk_size))
}

rolling_sd <- function(data, chunk_size=100) {
    # Rolling standard deviation plot
    
    library(zoo)
    plot(rollapply(data, chunk_size, FUN=sd))
}

rolling_variance <- function(data, chunk_size=100) {
    # Rolling variance plot
    
    library(zoo)
    plot(rollapply(data, chunk_size, FUN=var))
}

