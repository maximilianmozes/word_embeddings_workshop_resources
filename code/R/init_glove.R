###############################################################################
### Using pretrained GloVe models locally in R
### Models downloadable from: https://nlp.stanford.edu/projects/glove/
### author: B Kleinberg https://github.com/ben-aaron188
###############################################################################


###NOTE: you will need to download the pretrained GloVe vectors from:
###https://nlp.stanford.edu/projects/glove/


if (!require(data.table)){
  install.packages('data.table')
} 

if (!require(quanteda)){
  install.packages('quanteda')
} 

require(quanteda)
require(data.table)

setup_pipe = function(dir){
  print(paste("Looking for pretrained GloVe vectors in: ", dir, sep=""))
  
  currentwd = getwd()
  setwd(dir)
  
  files_in_dir = list.files(pattern = '*.txt')
  bool_sum = sum(1*grepl('glove', files_in_dir))
  if(bool_sum >= 1){
    print('Success - found GloVe objects in directory.')
    return(TRUE)
  } else {
    print('Failure = Could not find GloVe object.')
    return(FALSE)
  }

  setwd(currentwd)
  
}

init_glove = function(dir, which_model, dimensions){
  
  if(setup_pipe(dir) == T){
  
    if(which_model == '6B'){
      
      if(dimensions == 50){
        pt_glove = fread('./glove.6B.50d.txt', quote = "")  
        print('--- initialising the 50d model ---')
      } else if(dimensions == 100){
        pt_glove = fread('./glove.6B.100d.txt', quote = "")  
        print('--- initialising the 100d model ---')
      } else if(dimensions == 200){
        pt_glove = fread('./glove.6B.200d.txt', quote = "")  
        print('--- initialising the 200d model ---')
      } else if(dimensions == 300){
        pt_glove = fread('./glove.6B.300d.txt', quote = "")  
        print('--- initialising the 300d model ---')
      }
  
    } else if(which_model == '27B'){
      
      if(dimensions == 25){
        pt_glove = fread('./glove.twitter.27B.25d.txt', quote = "")  
        print('--- initialising the 27B 25d model ---')
      } else if(dimensions == 50){
        pt_glove = fread('./glove.twitter.27B.50d.txt', quote = "")  
        print('--- initialising the 27B 50d model ---')
      } else if(dimensions == 100){
        pt_glove = fread('./glove.twitter.27B.100d.txt', quote = "")  
        print('--- initialising the 27B 100d model ---')
      } else if(dimensions == 200){
        pt_glove = fread('./glove.twitter.27B.200d.txt', quote = "")  
        print('--- initialising the 27B 200d model ---')
      }
      
    } else if(which_model == '42B'){
      
      pt_glove = fread('./glove.42B.300d.txt', quote = "")  
      print('--- initialising the 42B 300d model ---')
      
    } else if(which_model == '840B'){
      
      pt_glove = fread('./glove.840B.300d.txt', quote = "")  
      print('--- initialising the 840B 300d model ---')
      
    } else if(!(which_model %in% c('6B', '27B', '42B', '840B'))){
      
      print('Choose from: 6B, 27B, 42B, 840B')
      print('For the 6B and 27B model, specify the dimensions too.')
      
    }
    
    pt_glove = pt_glove[!(is.na(V1)), ]
    feat_glove = pt_glove$V1
    pt_glove = pt_glove[, -1]
    names(pt_glove) = paste('vec', 1:dimensions, sep="_")
    
    pt_glove.matrix = as.matrix(pt_glove)
    rm(pt_glove)
    
    rownames(pt_glove.matrix) = feat_glove
    rm(feat_glove)
    
    pt_glove.dfm = as.dfm(rbind(pt_glove.matrix))
    rm(pt_glove.matrix)
    
    glove.pt <<- pt_glove.dfm
    rm(pt_glove.dfm)
    
    print('Success: initialised GloVe model as glove.pt')
  
  }
  
}

#CHANGELOG
#19 Feb - init
#24 Feb - added multi-model support
#28 Feb - fixed feat names bug
#29 Aug - changed path

#usage example:
# init_glove(dir = '.../glove', which_model = '6B', dim=100)
# 
# cos_sim_vals = textstat_simil(glove.pt
#                               , selection = c("man")
#                               , margin = "documents"
#                               , method = "cosine")
# head(sort(cos_sim_vals[,1], decreasing = TRUE), 5)

#load as:
#source('init_glove.R')

### END