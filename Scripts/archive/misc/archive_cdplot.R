# Read the csv file
df <- read.csv(file = '/Users/omid/Documents/GitHub/Causality/causal_drafts/Models/edges_df.csv')
# Remove redundant variables
df$'X' = NULL
df$'Graphs' = NULL

# Define my colors
my_col=c('#008001', #green
         '#BA0100', #red
         '#5F5F5F', #gray
         '#FFFF0A') #yellow

# Convert variables into factors
Scores=df$MMD_Loss #The model with max(MMD_Loss) has the min(Scores) 
Total_Edges=factor(df$Total_Edges)
OA_OB=factor(df$OA..OB-df$OB..OA, levels=c(+1, -1, 0), labels= c('OA-OB','OB-OA','Disconnected'))
OA_SA=factor(df$OA..SA-df$SA..OA, levels=c(-1, 0), labels= c('SA-OA','Disconnected'))
OA_SB=factor(df$OA..SB-df$SB..OA, levels=c(+1, -1, 0), labels= c('OA-SB','SB-OA','Disconnected'))
OB_SA=factor(df$OB..SA-df$SA..OB, levels=c(+1, -1, 0), labels= c('OB-SA','SA-OB','Disconnected'))
OB_SB=factor(df$OB..SB-df$SB..OB, levels=c(-1, 0), labels= c('SB-OB','Disconnected'))
SA_SB=factor(df$SA..SB-df$SB..SA, levels=c(0), labels= c('Disconnected'))


# Visualize the six conditional density plots 
par(mfrow=c(2,3))
cdplot(Scores, OA_OB, col=my_col[c(1, 2, 3)], ylevels = 3:1, xlab='',ylab='', main=bquote(O[A]~ 'vs.'~O[B]))
cdplot(Scores, OA_SA, col=my_col[c(2, 3)], ylevels = 2:1, xlab='',ylab='', main=bquote(O[A]~ 'vs.'~S[A]))
cdplot(Scores, OA_SB, col=my_col[c(1, 2, 3)], ylevels = 3:1, xlab='',ylab='', main=bquote(O[A]~ 'vs.'~S[B]))
cdplot(Scores, OB_SA, col=my_col[c(1, 2, 3)], ylevels = 3:1, xlab='',ylab='', main=bquote(O[B]~ 'vs.'~S[A]))
cdplot(Scores, OB_SB, col=my_col[c(2, 3)], ylevels = 2:1, xlab='',ylab='', main=bquote(O[B]~ 'vs.'~S[B]))
cdplot(Scores, Total_Edges, col= rainbow(length(levels(Total_Edges))), xlab='',ylab='', main='Total Number of Edges') 



# make labels and margins smaller
par(cex=0.7, mai=c(0.2,0.2,0.2,0.2))

# define area for the first
par(fig=c(0.05,0.30,0.55,.90))
cdplot(Scores, OA_OB, col=my_col[c(1, 2, 3)], ylevels = 3:1, xlab='',ylab='', main=bquote(bold(O[A]~ 'vs.'~O[B])))
par(fig=c(0.375,0.625,0.55,.90), new=TRUE)
cdplot(Scores, OA_SA, col=my_col[c(2, 3)], ylevels = 2:1, xlab='',ylab='', main=bquote(bold(O[A]~ 'vs.'~S[A])))
par(fig=c(0.70,0.95,0.55,.90), new=TRUE)
cdplot(Scores, OA_SB, col=my_col[c(1, 2, 3)], ylevels = 3:1, xlab='',ylab='', main=bquote(bold(O[A]~ 'vs.'~S[B])))
par(fig=c(0.05,0.30,0.10,.45), new=TRUE)
cdplot(Scores, OB_SA, col=my_col[c(1, 2, 3)], ylevels = 3:1, xlab='',ylab='', main=bquote(bold(O[B]~ 'vs.'~S[A])))
par(fig=c(0.375,0.625,0.10,.45), new=TRUE)
cdplot(Scores, OB_SB, col=my_col[c(2, 3)], ylevels = 2:1, xlab='',ylab='', main=bquote(bold(O[B]~ 'vs.'~S[B])))
par(fig=c(0.70,0.95,0.10,.45), new=TRUE)
cdplot(Scores, Total_Edges, col= rainbow(length(levels(Total_Edges))), xlab='',ylab='', main='Total Number of Edges') 


# To add a text:
#mtext("Outer Margin Area", line=22, cex=1.2, font=2)

# To use ggplot2
#ggplot(df, aes(Scores, after_stat(count)))+ geom_density(aes(fill = OA_OB), position='fill')



#-----------------------------------------------Pre-processing
# Convert all columns (except for "Scores") into factor: 
index <- 2:ncol(df)
df[ , index] <- lapply(df[ , index], as.factor)
#-----------------------------------------------cdplot(x,y)
# cdplot(y~x) where y is categorical factor and x is numerical 
# Interpretation: p(y=0 | x=15) = 0.8 and p(y=1 | x=15) = 0.2

#-----------------------------------------------# NASA Examples
fail <- factor(c(2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1,1, 2, 1, 1, 1, 1, 1),levels = 1:2, labels = c("no", "yes"))
temperature <- c(53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70,70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81)
## CD plot
cdplot(fail ~ temperature)
cdplot(fail ~ temperature, bw = 2)
cdplot(fail ~ temperature, bw = "SJ")
## highlighting for failures
cdplot(fail ~ temperature, ylevels = 1:2)
## scatter plot with conditional density
cdens <- cdplot(fail ~ temperature, plot = TRUE)
plot(I(as.numeric(fail) - 1) ~ jitter(temperature, factor = 2), xlab = "Temperature", ylab = "Conditional failure probability")
lines(53:81, 1 - cdens[[1]](53:81), col = 2)

#-----------------------------------------------Density Charts 
movies <- read.csv(file = '/Users/omid/Downloads/P2-Movie-Ratings.csv')
colnames(movies)=c('Film', 'Genre', 'CriticRating', 'AudienceRating', 'BudgetMillions', 'Year')
movies$Year=factor(movies$Year)
movies$Film=factor(movies$Film)
movies$Genre=factor(movies$Genre)
s = ggplot(data=movies, aes(x=BudgetMillions))
s+geom_density(aes(fill=Genre), position='stack')

#-----------------------------------------------Lattice Examples: kernel density plots by factor level
library(lattice)
Scores=df$Scores
OA.OB=factor(df$OA.OB,levels=c(0,1),labels=c("F","T"))
OA.SB=factor(df$OA.SB,levels=c(0,1),labels=c("F","T"))
densityplot(~Scores, main="Marginal Density Plot", xlab="MMD Loss")      
densityplot(~Scores|OA.OB*OA.SB, main="Coditional Density Plot by OA.OB and OA.SB Connections", xlab="MMD Loss") 
