card []testing_set;
card []training_set;

//////////////////////////This section of code is adapted from Charles Fried's tutorial "Let's code a Neural Network from scratch"                         
                        //Please see https://medium.com/typeme/lets-code-a-neural-network-from-scratch-part-1-24f0a30d7d62
                        //I owe a big "thank you" to his tutorial that helped me learn to Code a Neural Network
                        //loadData was kept as an object as I think it would not be easier to understand as functions
                        //Some variable names were changed

class card{ ///////organize images and labels as memory cards to keep them matched
  float [] img;  //images will load into a single dimension array
  float [] dig;  //label values as 0 and 1 on each 0 to 9 digits will be stored in this array
  int label;     //label as 0 to 9 digits will load into this variable
 
  
  card(){
    img = new float [196];       //each image has 196 pixels as it is reduced from standard MNIST by Charles Fried 
    dig = new float [10];
  }
  
  void imageLoad(byte [] images, int offset){  //load reduced MNIST images pixels into a 196 elements' array 
    
    for(int i=0; i<196; i++){
      img[i]=int(images[i+offset])/255.0+0.01; //normalize to 0,1 +0.01
    }
    
  }
  
  void labelLoad(byte [] labels, int offset){  //load labels to match the format of our outputs
    label = int(labels[offset]);
    
    for (int i=0; i<10; i++){
      if (i==label){                     
        dig[i]=1.0;
      }else{
        dig[i]=0.0;
      }
    }
  }
  
}

void loadData(){ //////////////////////////////this function load the reduced 14 x 14 MNIST into two sets--training set and testing set
  byte [] images =loadBytes("t10k-images-14x14.idx3-ubyte");
  byte [] labels =loadBytes("t10k-labels.idx1-ubyte");
  training_set = new card [8000];
  int tr_pos=0;
  testing_set = new card [2000];
  int te_pos=0;
  for(int i=0; i<10000; i++){
    if(i%5!=0){////////////////////////////////Pulling data evenly from the file
      training_set[tr_pos]=new card();                    // load training set
      training_set[tr_pos].imageLoad(images, 16+i*196);   // There is an offset of 16 bytes
      training_set[tr_pos].labelLoad(labels, 8 + i);      // There is an offset of 8 bytes
      tr_pos++;
    } else {
      testing_set[te_pos] = new card();                   // load testing set
      testing_set[te_pos].imageLoad(images, 16 + i*196);  // There is an offset of 16 bytes 
      testing_set[te_pos].labelLoad(labels, 8 + i);       // There is an offset of 8 bytes
      te_pos++;
    }
  }
}
