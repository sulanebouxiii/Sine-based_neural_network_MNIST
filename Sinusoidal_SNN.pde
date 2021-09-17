//////////////////////(LEFT) click stops training and test the current Neural Network (RIGHT) click resumes training

float [] input;
float [] lables;
float [][][] wIn;    //input layer to hidden layer 1 weights
float [] y1;         //hidden layer 1 output
float [][][] wIn1;   //hidden layer1 to hidden layer 2 weights 
float [] y2;         //hidden layer 2 output
float [][][] wOut;   //hidden layer 2 to output layer weights
float [] yHat;       //output layer or predictions in another name
float c = 0.01;      //learning rate or learning constant in another name
float sqerror;       //mean sqare error
float SD;            //mean sqare error across 200 samples

void setup() {
  size(700, 400);
  background(0);
  loadData();
  layers(196, 64, 49, 10);//setup the number of neurons in each layer
}

void draw() {
  
  SD=0;//reset Mean Square Error to zero before training the next 200 samples
  
  for (int i=0; i<200; i++) {
    int no=(int)random(0, training_set.length);       //select a random sample from the training set 
    input = training_set[no].img;
    lables = training_set[no].dig;
    feedforward(input, wIn, wIn1, wOut);              //feedforward with current weights 
    backprop(input, y1, y2, lables, wIn, wIn1, wOut); //backpropagate under current output, lables, and weights
    SD+=sqerror/200;                                  //calculate the mean square error for these 200 samples
  }
  
  fill(0);                                            //Display "SD" in drawing window
  rect(360,320,150,30);                               //"cover" the previous text with a rectangle
  fill(255);
  textSize(12);
  text("SD = "+SD,380,350);

  layout();                                           //calls the layout function to draw dots representing each neurons and outputs
}

void layers(int in, int hid, int hid2, int out) {
  input = new float [in];
  wIn = new float [in][2][hid];
  y1 = new float [hid];
  wIn1 = new float [hid][2][hid2];
  y2 = new float [hid2];
  wOut = new float [hid2][2][out];
  yHat = new float [out];

  
  for (int i=0; i<in; i++) { ////////////////populate weights with random numbers
    for (int j=0; j<hid; j++) {
      wIn [i][0][j]=randomGaussian()*0.1;  //initial two weights connecting input to hidden layer 1
      wIn [i][1][j]=randomGaussian()*0.1;  //The two weights connecting each signle neuron were called weight in and weight out in literatures. 
    }                                      //But this is confusion giving each neuron has its own input and output. 
  }                                        //Weight in and weight out in SNN means weight inside and outside of sinusoidal function to adjust the frequency and amplitude respectively.
                                           //So I use one 3d array to contain both weights and simply use 0 and 1 to differentiate between the two with 0 adjusting amplitudes and 1 adjusting the frequency.

  for (int i=0; i<hid; i++) {
    for (int j=0; j<hid2; j++) {
      wIn1 [i][0][j]=randomGaussian()*0.1;  //initial weights from hidden layer 1 to hidden layer 2
      wIn1 [i][1][j]=randomGaussian()*0.1;  //Gaussian random train faster 
    }
  }

  for (int i=0; i<hid2; i++) {
    for (int j=0; j<out; j++) {
      wOut [i][0][j]=randomGaussian()*0.1;  //inital weights from hidden layer 2 to output layer
      wOut [i][1][j]=randomGaussian()*0.1;
    }
  }
}

void feedforward(float [] input, float [][][] w1, float [][][] w2, float [][][] w3) {




  for (int j=0; j<y1.length; j++) {                         //feeding input to hidden layer1
    float sum=0;
    for (int i=0; i<input.length; i++) {
      sum += w1[i][0][j]*sin(input[i]*w1[i][1][j]*TWO_PI);  //node operation without needing activation function as sine is nonlinear
    }
    y1 [j] = sum;
  }

  for (int j=0; j<y2.length; j++) {                         //feeding hidden layer1 to hidden layer2
    float sum=0;
    for (int i=0; i<y1.length; i++) {
      sum += w2[i][0][j]*sin(y1[i]*w2[i][1][j]*TWO_PI);
    }
    y2 [j] = sum;
  }

  for (int j=0; j<10; j++) {                                //feeding hidden layer2 to output layer
    float sum=0;
    for (int i=0; i<y2.length; i++) {
      sum += w3[i][0][j]*sin(y2[i]*w3[i][1][j]*TWO_PI);
    }
    yHat [j] = sum;
  }
}


void backprop(float [] iput, float [] yhid1, float [] yhid2, float [] lables, float [][][]weights1, float [][][]weights2, float [][][]weights3) {
  
  sqerror = 0;
  
  float [][]error1 = new float [2][yhid2.length]; //initialize array to store d_error/d_hiddenlayer2output(as y2 in this case) to back propagate error to hidden layer2
  float [][]error2 = new float [2][yhid1.length]; //initialize array to store d_error/d_hiddenlayer1output(as y1 in this case) to back propagete error to hidden layer1
  
  
  for (int i=0; i<lables.length; i++) {           //calculate error from each value inside label and from 0 to 9 there will be one value "1" while others are "0"
    
    float error  =  yHat [i]-lables [i];          //simply subtracting desired value from the output (called yHat as in some literatures)
    sqerror += pow(error, 2)/lables.length;
    
    for (int j=0; j<yhid2.length; j++) { ///////////back propagating to hidden layer 2


      float dw1 = TWO_PI*yhid2[j]*weights3[j][0][i]*cos(yhid2[j]*TWO_PI*weights3[j][1][i]);
      float dw0 = sin(yhid2[j]*TWO_PI*weights3[j][1][i]);
      dw0 *= error*c;
      dw1 *= error*c;

      weights3[j][0][i]-=dw0;
      weights3[j][1][i]-=dw1;

      float dy_dx = TWO_PI*weights3[j][0][i]*weights3[j][1][i]*cos(yhid2[j]*TWO_PI*weights3[j][1][i]);  //gradient to the input of ouput layer which is the output of hidden layer 2 
      error1 [0][j] = +dy_dx*error;               //calcuate the error of hidden layer 2 using chain rule
      error1 [1][j] = +dy_dx*error;               //this is a non-threatening mistake there should be only one error for each neuron in the hidden layer2
      
    }
  }

  for (int i=0; i<yhid2.length; i++) { /////////////back propagating to hidden layer 1
    for (int j=0; j<yhid1.length; j++) {

      float dw1 = TWO_PI*yhid1[j]*weights2[j][0][i]*cos(yhid1[j]*TWO_PI*weights2[j][1][i]);
      float dw0 = sin(yhid1[j]*TWO_PI*weights2[j][1][i]);
      dw0 *= error1[0][i]*c;
      dw1 *= error1[1][i]*c;
      weights2[j][0][i]-=dw0;
      weights2[j][1][i]-=dw1;
      float dy_dx = TWO_PI*weights2[j][0][i]*weights2[j][1][i]*cos(yhid1[j]*TWO_PI*weights2[j][1][i]);
      error2 [0][j] += dy_dx*error1[0][i];         //calculate the error of hidden layer 1 using chain rule
      error2 [1][j] += dy_dx*error1[1][i];
      
    }
  }

  for (int i=0; i<yhid1.length; i++) { //////////////back propagating to input layer
    for (int j=0; j<input.length; j++) {
      float dw1 = TWO_PI*iput[i]*weights1[j][0][i]*cos(iput[i]*TWO_PI*weights1[j][1][i]);
      float dw0 = sin(iput[i]*TWO_PI*weights1[j][1][i]);
      dw0 *= error2 [0][i] *c;
      dw1 *= error2 [1][i] *c;
      weights1[j][0][i]-=dw0;
      weights1[j][1][i]-=dw1;
    }
  }
}

void display(float value) { //display each neuron as a circle and its gray scale coordinates with neural output value
  fill(128*(1-2*value));
  circle(0, 0, 5);
}

void layout() { //////////////Wrap neurons in each layers into rectangular matrices
  
  for (int i=0; i<input.length; i++) {    //input layer will display hand-written digits 
    pushMatrix();
    translate(i%14*5+50, i/14*8+100);     //coil up the array of neurons
    display(input[i]);
    popMatrix();
  }

  for (int k=0; k<y1.length; k++) {       //layout hidden layer 1
    pushMatrix();
    translate(k%8*5+342, k/8*8+156);
    display(y1[k]);
    //println(y1[k]);
    popMatrix();
  }

  for (int k=0; k<y2.length; k++) {       //layout hidden layer 2
    pushMatrix();
    translate(k%7*5+442, k/7*8+156);
    display(y2[k]);
    //println(y1[k]);
    popMatrix();
  }

  for (int j=0; j<yHat.length; j++) {     //layout output layer 
    pushMatrix();
    translate(600, j*10+156);
    display(yHat[j]);
    fill(196);
    textSize(9.5);
    text(j, 12, 3);//display 0 to 9 digits along the output dots
    popMatrix();
  }
}

void mousePressed() {
  if (mouseButton == LEFT) {
    noLoop();
    input = testing_set[(int)random(2000)].img;  //select random sample from testing set
    feedforward(input, wIn, wIn1, wOut);         //feed through trained neural net 
    redraw();                                    //show results
  }

  if (mouseButton == RIGHT) {                    //Left click resumes training
    loop();
  }
}
