setwd("/Users/deepthought42/Documents/Studium/ML1/MachLearn-Exercises/Jupyter")
list.files()
adult = read.csv('adult.data.csv', header = F, stringsAsFactors = F)
Columnames = read.csv("Colnames.txt", sep =":", header = F)
Columnames = Columnames[,1]
names(adult) = Columnames


#head(grepl('gov', adult$workclass))
library(stringr)


preprocessing = function(adult){
  
  #replace white spaces in character variables
  for(i in 1:ncol(adult)){
    if(class(adult[,i])=='character'){
      adult[,i] = str_replace_all(adult[,i], ' ','')
    }
  }
  
  #create summarizing categories for workclass
  adult$workclass[grepl('gov', adult$workclass)] = 'government-employed'
  adult$workclass[grepl('Self-emp', adult$workclass)] = 'self-employed'
  adult$workclass[grepl('Never', adult$workclass)| grepl('Without', adult$workclass)] = 'unemployed'
  adult$workclass[adult$workclass == '?'] = NA
  #unique(adult$workclass)
  
  he= c('Doctorate', 'Masters', 'Bachelors')
  hs= c('HS-grad', 'Some-college', 'Prof-school', 'Assoc-acdm', 'Assoc-voc')
  
  adult$education[adult$education %in% he] = 'higher education'
  adult$education[adult$education %in% hs] = 'highschool'
  adult$education[!(adult$education %in% c('higher education','highschool'))] = 'lower education'
  unique(adult$education)
  
  #summary(adult)
  #sapply(adult, function(x) sum(is.na(x)))
  
  #remove NAs
  adult =adult[!is.na(adult$workclass),]
  
  #generate new dataframe with continious variables = dummy variables 
  adult2 = adult[, sapply(adult, class) == 'integer']
  adult2$higher_educ = adult$education == 'higher education'
  adult2$lower_educ = adult$education == 'lower education'
  adult2$is_male = adult$sex == 'Male'
  
  adult2$gov_emp = adult$workclass == "government-employed"
  adult2$private = adult$workclass == "Private"
  adult2$self_emp = adult$workclass == "self-employed"
  
  #make sure all to have no spelling errors
  #sapply(adult2[,7:12], mean)
  
  adult2[,7:12] = sapply(adult2[,7:12], function(x) x = as.integer(x))
  adult2$high_income = ifelse(adult$class == '>50K',1,0)
  return(adult2)
}

pre_processed = preprocessing(adult = adult)

write.csv(pre_processed, 'adult.data.prepared.csv')

adult_test = read.csv("adult.test.txt", header = F, stringsAsFactors = F)
names(adult_test) = Columnames
adult_test$class = str_replace_all(adult_test$class, '\\.', '')


test_processed = preprocessing(adult = adult_test)
write.csv(test_processed, "adult.test.prepared.csv")
