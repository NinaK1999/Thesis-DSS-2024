########################################################################################################################
#                                                                                                                      #
#                                       Network analysis of own scales                                                 #
#                                                                                                                      #
########################################################################################################################

library(readxl)
library("huge")
library("igraph")
library(qgraph)
library(mgm)
library("MASS")
##### STUDY 1 #####

library(readxl)
dataset_study1_OSF <- read_excel("Library/Mobile Documents/com~apple~CloudDocs/Data Science and Society/Block 3/Thesis/Data versie 2/All_data/dataset_study1_OSF.xlsx")
View(dataset_study1_OSF)
Data1 <- dataset_study1_OSF

hist(Data1$FBI7, breaks = 20, main = "Histogram of FBI7", xlab = "Values")
hist(Data1$FBI8, breaks = 20, main = "Histogram of FBI7", xlab = "Values")

# Apply log transformation to the columns
Data1$FBI7_log <- log(Data1$FBI7 +1)
Data1$FBI8_log <- log(Data1$FBI8 +1)

# # Define the breaks for the categories
# breaks1_FBI7 <- quantile(Data1$FBI7,seq(0, 1, by=0.1))
# breaks1_FBI8 <- quantile(Data1$FBI8,seq(0, 1, by=0.1))
# 
# breaks1_FBI8
# 
# non_increasing_indices <- which(diff(breaks1_FBI8)<=0)
# 
# breaks1_FBI8[non_increasing_indices + 1] <- breaks1_FBI8[non_increasing_indices + 1] + 1
# breaks1_FBI8
# # Create a new column for FBI7
# Data1$FBI7_category <- cut(Data1$FBI7, breaks = breaks1_FBI7, labels = c(1:10), include.lowest = TRUE)
# Data1$FBI8_category <- cut(Data1$FBI8, breaks = breaks1_FBI8, labels = c(1:10), include.lowest = TRUE)
# barplot(table(Data1$FBI7_category), main="Frequency of Categories", xlab="Category", ylab="Frequency")
# barplot(table(Data1$FBI8_category), main="Frequency of Categories", xlab="Category", ylab="Frequency")
# 
# comparison1 <- data.frame(Data1$FBI7, Data1$FBI7_category)
# comparison2 <- data.frame(Data1$FBI8, Data1$FBI8_category)
#      
# Normalization
Data1_norm <- huge.npn(Data1)

# Remove columns by column names
Data1_norm <- subset(Data1_norm, select = -c(FBI, MSFUP, MSFUaprivate, MSFUapublic, COMF, CSS, 
                                             RSES, RRS, FBI7, FBI8))
# Remove columns by column names excluding those containing "DASS" and the specific DASS columns
Data1_norm <- Data1_norm[, !grepl("DASS", colnames(Data1_norm)) | colnames(Data1_norm) %in% 
                           c("DASS_stress", "DASS_anxiety", "DASS_depression")]
plot(Data1$FBI7)
plot(Data1$FBI8)


# Obtain GGM
Cor_Study1 <- cor_auto(Data1_norm)
print(Data1_norm)

group.item <- list(c(1:6), c(7:16), c(17:27), c(28:37), c(38:52), c(53:74), c(75:77), c(78:79))
Study1_GLASSO <- qgraph(Cor_Study1, layout = "spring", groups = group.item, graph =
                          "glasso", tuning = 0.5, sampleSize = 207, legend.cex =
                          0.4, color = c("olivedrab2", "darkseagreen1", "wheat3",
                                         "goldenrod2", "darkorange1", "darkslategray2",
                                         "mediumpurple","olivedrab2"), borders = FALSE,
                        theme = "colorblind", usePCH = TRUE,
                        minimum = 0.05)
WeightMatrix_Study1_GLASSO <- getWmat(Study1_GLASSO)
centralityPlot(Study1_GLASSO)
centralityTable(Study1_GLASSO, standardized = TRUE)

# Add predictability to the plot
Data1_norm_matrix <- as.matrix(Data1_norm)
p_1 <- ncol(Data1_norm_matrix)
dim(Data1_norm_matrix)

# Calculate R2
set.seed(1)
fit_obj1 <- mgm(data=Data1_norm_matrix,
                type = rep('g', p_1),
                level = rep('1', p_1),
                lambdSel = 'EBIC',
                ruleReg = 'OR')
pred_obj1 <- predict(object = fit_obj1, 
                     data = Data1_norm, 
                     errorCon = 'R2')

R2_study1 <- pred_obj1$error[, 2]  # Extract R^2 values
print(paste("R^2 for Study 1:", mean(R2_study1)))

print(pred_obj1)
dev.off()
R2_network1_Spring <- qgraph(WeightMatrix_Study1_GLASSO,
                             layout = "spring",
                             pie = as.numeric(as.character(pred_obj1$error[,2])), 
                             pieColor = rep('#377EB8', p_1),
                             labels = colnames(Data1_norm_matrix),
                             groups = group.item, color=c("olivedrab2", "darkseagreen1", 
                                                          "wheat3", "goldenrod2", "darkorange1", "darkslategray2",
                                                          "mediumpurple", "olivedrab2"), theme="colorblind", 
                             maximum=0.44, details=TRUE, minimum = 0.05)

# Check how predictability model fits with model obtained via qgraph:
cor(getWmat(Study1_GLASSO)[upper.tri(getWmat(Study1_GLASSO))], fit_obj1$pairwise$wadj[upper.tri(fit_obj1$pairwise$wadj)], method="pearson")

# Plot centrality for model:
centralityPlot(R2_network1_Spring)

##### STUDY 2 #####

# Data2 <- Data2_subscales #Insert csv file
library(readxl)
dataset_study2_OSF <- read_excel("Library/Mobile Documents/com~apple~CloudDocs/Data Science and Society/Block 3/Thesis/Data versie 2/All_data/dataset_study2_OSF.xlsx")
View(dataset_study2_OSF)
Data2 <- dataset_study2_OSF

# Due to impossible values, remove row 304 and row 161
Data2 <- Data2[-304,]
Data2 <- Data2[-161,]

plot(Data2$FBI7)
plot(Data2$FBI8)

hist(Data2$FBI7, breaks = 20, main = "Histogram of FBI7", xlab = "Values")
hist(Data2$FBI8, breaks = 20, main = "Histogram of FBI7", xlab = "Values")

# Apply log transformation to the columns
Data2$FBI7_log <- log(Data2$FBI7 +1)
Data2$FBI8_log <- log(Data2$FBI8 +1)

# # Define the breaks for the categories
# breaks2_FBI7 <- quantile(Data2$FBI7,seq(0, 1, by=0.1))
# breaks2_FBI8 <- quantile(Data2$FBI8,seq(0, 1, by=0.1))
# 
# # Create a new column for FBI7
# Data2$FBI7_category <- cut(Data2$FBI7, breaks = breaks2_FBI7, labels = c(1:10), include.lowest = TRUE)
# Data2$FBI8_category <- cut(Data2$FBI8, breaks = breaks2_FBI8, labels = c(1:10), include.lowest = TRUE)
# barplot(table(Data2$FBI7_category), main="Frequency of Categories", xlab="Category", ylab="Frequency")
# barplot(table(Data2$FBI8_category), main="Frequency of Categories", xlab="Category", ylab="Frequency")
# 
# comparison <- data.frame(Data2$FBI7, Data2$FBI7_category)

# Normalization
Data2_norm <- huge.npn(Data2)

# Remove columns by column names
Data2_norm <- subset(Data2_norm, select = -c(FBI, MSFUP, MSFUaprivate, MSFUapublic, COMF, CSS, 
                                             RSES, RRS, CI, CI2, FBI7, FBI8))
# Remove columns by column names excluding those containing "DASS" and the specific DASS columns
Data2_norm <- Data2_norm[, !grepl("DASS", colnames(Data2_norm)) | colnames(Data2_norm) %in% 
                           c("DASS_stress", "DASS_anxiety", "DASS_depression")]
print(Data2_norm)

#Obtain GGM
Cor_Study2 <- cor_auto(Data2_norm)

group.item <- list(c(1:6), c(7:16), c(17:27), c(28:37), c(38:52), c(53:74), c(75:77), c(78:79))
Study1_GLASSO <- qgraph(Cor_Study1, layout = "spring", 
                        groups=group.item, graph = "glasso", sampleSize = 207, 
                        color=c("olivedrab2", "darkseagreen1", "wheat3", "goldenrod2", 
                                "darkorange1", "darkslategray2", "mediumpurple", "olivedrab2"), 
                        borders=FALSE, details=T, usePCH=TRUE, minimum= 0.05) 
Study2_GLASSO <- qgraph(Cor_Study2, layout = "spring", groups=group.item, 
                        graph = "glasso", sampleSize = 468, color=c("olivedrab2", 
                        "darkseagreen1", "wheat3", "goldenrod2", "darkorange1", 
                        "darkslategray2", "mediumpurple", "olivedrab2"), borders=FALSE, details=T, usePCH=TRUE,
                        minimum = 0.05)
WeightMatrix_Study2_GLASSO <- getWmat(Study2_GLASSO) 
#-->Max. edge over studies = .43, constrain lay-out for visual comparison over both studies

L1 <- averageLayout(getWmat(Study1_GLASSO), getWmat(Study2_GLASSO))

#Add predictability
Data1_norm_matrix <- as.matrix(Data1_norm)
Data2_norm_matrix <- as.matrix(Data2_norm)
p_1 <- ncol(Data1_norm_matrix)
p_2 <- ncol(Data2_norm_matrix)
dim(Data1_norm_matrix)
dim(Data2_norm_matrix)

library(mgm)
set.seed(1)
fit_obj1 <- mgm(data=Data1_norm_matrix,
                type = rep('g', p_1),
                level = rep('1', p_1),
                lambdSel = 'EBIC',
                ruleReg = 'OR')
pred_obj1 <- predict(object = fit_obj1, 
                     data = Data1_norm, 
                     errorCon = 'R2')
print(pred_obj1)
dev.off()

R2_network1_L1 <- qgraph(WeightMatrix_Study1_GLASSO,
                         layout = L1,
                         pie = as.numeric(as.character(pred_obj1$error[,2])), 
                         pieColor = rep('#377EB8', p_1),
                         labels = colnames(Data1_norm_matrix),
                         groups = group.item, color=c("olivedrab2", "darkseagreen1", 
                         "wheat3", "goldenrod2", "darkorange1", "darkslategray2", 
                         "mediumpurple", "olivedrab2"), theme="colorblind", maximum=0.44, details=TRUE,
                         minimum = 0.05)

library(mgm)
set.seed(1)
fit_obj2 <- mgm(data=Data2_norm_matrix,
                type = rep('g', p_2),
                level = rep('1', p_2),
                lambdSel = 'EBIC',
                ruleReg = 'OR')
pred_obj2 <- predict(object = fit_obj2, 
                     data = Data2_norm, 
                     errorCon = 'R2')

R2_study2 <- pred_obj2$error[, 2]  # Extract R^2 values
print(paste("R^2 for Study 2:", mean(R2_study2))) 

print(pred_obj2)

R2_network2_L1 <- qgraph(WeightMatrix_Study2_GLASSO,
                         layout = L1,
                         pie = as.numeric(as.character(pred_obj2$error[,2])), 
                         pieColor = rep('#377EB8', p_2),
                         labels = colnames(Data2_norm_matrix),
                         groups = group.item, color=c("olivedrab2", "darkseagreen1", 
                         "wheat3", "goldenrod2", "darkorange1", "darkslategray2", 
                         "mediumpurple", "olivedrab2"), theme = "colorblind", maximum=0.44, 
                         details=TRUE, minimum = 0.05)

#For unconstrained model study 2
R2_network2_Spring <- qgraph(WeightMatrix_Study2_GLASSO,
                             layout ="spring",
                             pie = as.numeric(as.character(pred_obj2$error[,2])), 
                             pieColor = rep('#377EB8', p_2),
                             labels = colnames(Data2_norm_matrix),
                             groups = group.item, color=c("olivedrab2", 
                             "darkseagreen1", "wheat3", "goldenrod2", "darkorange1", 
                             "darkslategray2", "mediumpurple", "olivedrab2"), theme = "colorblind", 
                             maximum=0.44, details=TRUE, minimum = 0.05)

# Check how predictability model fits with model obtained via qgraph for Study 2:
cor(getWmat(Study2_GLASSO)[upper.tri(getWmat(Study2_GLASSO))], fit_obj2$pairwise$wadj[upper.tri(fit_obj2$pairwise$wadj)], method="pearson")

# Centrality for both studies:
centralityPlot(list("Study 1" = Study1_GLASSO, "Study 2" = Study2_GLASSO))
