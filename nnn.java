import java.io.Serializable;
import java.util.Random;

public class nnn implements Serializable{


	public int imageSize = 128*120;
	private static final long serialVUID = -500000000000000000L;

	//weights between the 3 layers Input, Hidden and Output
	private double [][]	wtsInputToHidden;
	private double [] wtsHiddenToOutput; 

	//Label for number of neurons or nodes in each layer
	private int inputDimension; //150000 something
	private int outputDimension; //1
	private int numOfHiddenNodes;  //32 upto me 
	public double[] data; // To hold the pixel values
	public double[] hiddenNodes; // Array to hold hidden node values
	public double output;
	public int mode; //Mode = 1 => training, mode = 2 => testing
	public double deltaK = 0;
	public double[] deltaH = {0,0,0,0,0};
	public double target;
	public double rate = 0.05;

	 	
	

	//NN Constructor to create object in main.

	public NN(int inputDimension, int numOfHiddenNodes, int outputDimension, int mode, double[] data, int sex){
		this.inputDimension = inputDimension;
		this.numOfHiddenNodes = numOfHiddenNodes;
		this.outputDimension = outputDimension;
		this.mode = mode;
		this.wtsInputToHidden = new double[numOfHiddenNodes][inputDimension]; 
		this.data = new double[this.inputDimension]; // giving size
		for (int t = 0; t < inputDimension; t++)  this.data[t] = (data[t]/85); // initializing

		this.hiddenNodes = new double[this.numOfHiddenNodes];
		this.wtsHiddenToOutput = new double[numOfHiddenNodes];
		
		this.target = target;

		if(sex == 0) this.target = 0.91;
		else if(sex == 1) this.target = 0.09;
		else System.out.println("This is not a gender");
// SINCE we are having object for every file we are initializing the weights in the constructor
		Random rng = new Random(System.currentTimeMillis());

		for(int i = 0 ; i < numOfHiddenNodes ; i++){
			for (int j = 0 ; j < imageSize ; j++ ){
				if (i == 0 ) {// eyes
					wtsInputToHidden[i][j] = rng.nextDouble() - 0.5;
				}
				else if (i == 1 ) {// nose
					wtsInputToHidden[i][j] = rng.nextDouble() - 0.5;
				}
				else if (i == 2 ) {//shoulder
					wtsInputToHidden[i][j] = rng.nextDouble() - 0.5;
				}
				else if (i == 3 ) {// jaws
					wtsInputToHidden[i][j] = rng.nextDouble() - 0.5;
				}
				else if(i==4){// lips
					wtsInputToHidden[i][j] = rng.nextDouble() - 0.5;
				}
			}
		}
		//System.out.println("numofhiddenNodes " + numOfHiddenNodes + " wtsHiddenToOutput " + wtsHiddenToOutput.length);
		int w = 0;
		for(w = 0; w < numOfHiddenNodes; w++){
			wtsHiddenToOutput[w] = 0.0000002;
		//	System.out.println(" wtsHiddenToOutput " + wtsHiddenToOutput[w]);
		}
	}
	//Now we need a function to train this neural network so it can learn. 
	//So we will have function name Train. For training the arguments we need are
	//training Data, training label to tell NN whether its a male of female. 

	//public void train (double[] trainData, double[][] trainLabels, double stepSize, double tolerance, double maxItr){
	public void train (double[] trainData, double cycle){

	while(cycle > 0){ //have to change this conditon may be between 0.8 and 0.9

		/* 
		COMPUTE THE VALUES FOR EACH NODE IN HIDDEN LAYER AND OUTPUT LAYER -----> CALLING MATRIX VECTOR PRODUCT
		*/

		//Computing values of nodes at hidden layer ---------------- this is an array ------------------------ (1)
		hiddenNodes = matrixVectorProduct(wtsInputToHidden, data);
		double [] finalHiddenNodes = new double[hiddenNodes.length];
		for(int q = 0; q < hiddenNodes.length; q++) finalHiddenNodes[q] = activation(hiddenNodes[q]);
		//prints what is saves in hiddenNodes
		//System.out.println( "What is saved in hiddenNodes array " + hiddenNodes[0] + "___" + hiddenNodes[1] + "___" + hiddenNodes[2] + "___" + hiddenNodes[3] + "___" + hiddenNodes[4] );
		
		//Computing values of nodes at output layer ---------------- this is just a variable ------------------------ (2)
		output = matrixVectorProductForOutput(wtsHiddenToOutput, finalHiddenNodes);

		/*
		CALLING SIGMOID FUNCTION TO CONVERT COMPUTED OUTPUT BETWEEN 0 AND 1 ----> RETURN FIANLOUTPUT
		*/
		double finaloutput;
		finaloutput = activation(output);
		System.out.println("Sex is " + target + " Final output -------------------->"+ finaloutput );
		/*
		CHECK IF THE FIANLOUTPUT IS THE SAME AS 0.8
		*/
		if(finaloutput == target){
			break;
		}

		else{

		/*
		UPDATE WEIGHTS FROM INPUT TO HIDDEN LAYER ---- USING STOCHASTIC GRADIENT DESCENT
		*/


		// FOR Hidden to Output ---- We are using deltaK and deltaWK
		// for difference of weights between hidden to output
		deltaK = finaloutput*(1 - finaloutput ) * ( target - finaloutput); 
		//System.out.println( "deltaK ------------------> "+ deltaK );
		// for difference of weights between input and hidden

		double deltaWK = rate * deltaK; 
		//System.out.println( "deltaWK ------------------> "+ deltaWK );
		// Updating weights between hidden and output
		for(int i = 0 ; i < numOfHiddenNodes ; i++){ //...............?? what should be the Xi
			//System.out.println( "wtsHiddenToOutput " + i + "     " + wtsHiddenToOutput[i] );
			wtsHiddenToOutput[i] = wtsHiddenToOutput[i] + (deltaWK * finalHiddenNodes[i]);
			//System.out.println( "wtsHiddenToOutput " + i + "     " + wtsHiddenToOutput[i] );
		}



		// FOR Input to hidden ---- We are using deltaH and deltaWH
		//Point 3
		for (int i = 0 ; i < hiddenNodes.length ; i++){
			//System.out.println( "deltaH ------------------> "+ deltaH );
			//	System.out.println( "hiddenNodes 1 to 5 ------------------> "+ hiddenNodes[i] );
			deltaH[i] = finalHiddenNodes[i] * (1 - finalHiddenNodes[i]);
		}

		//Point 4
		for( int i = 0 ; i < numOfHiddenNodes ; i++ ){
			double deltaWH = rate * deltaH[i];
			//System.out.println( "deltaWH ------------------> "+ deltaWH );
			for( int j = 0 ; j < imageSize ; j++ ) {
				wtsInputToHidden[i][j] = wtsInputToHidden[i][j] + (deltaWH * data[j]);
				//System.out.println( "wtsInputToHidden i,j  " + i + ",    " + j + "   " + wtsInputToHidden[i][j] );
				}
		}

		//COMPUTING THE VALUE AGAIN WITH NEW WEIGHTS
		hiddenNodes = matrixVectorProduct(wtsInputToHidden, data);
		for(int q = 0; q < hiddenNodes.length; q++) finalHiddenNodes[q] = activation(hiddenNodes[q]);
		output = matrixVectorProductForOutput(wtsHiddenToOutput, finalHiddenNodes);
		System.out.println( "Sex is " + target + " Regular output -------------------->"+ output );
		}
		cycle--;
	}
		
		//return wtsInputToHidden;
	}

	public double[][] getInputWeights(){
		return wtsInputToHidden;
	}
	public double[] getHiddenWeights(){
		return wtsHiddenToOutput;
	}


	double test( double[][] weightsItoH, double[] weightsHtoO ){

		hiddenNodes = matrixVectorProduct(weightsItoH, data);
		double [] finalHiddenNodes = new double[hiddenNodes.length];
		//calling activation on hidden nodes
		for(int q = 0; q < hiddenNodes.length; q++) finalHiddenNodes[q] = activation(hiddenNodes[q]);
		
		output = matrixVectorProductForOutput(weightsHtoO, finalHiddenNodes);
		double finaloutput = activation(output);
		System.out.println("Testdata " + " Final output -------------------->"+ finaloutput );

	 return finaloutput;
	}

	/* 
	Funtion that will predict on a given input
	*/

	/*
	public double test(double[] input) {

		double[] intermediate = new double[numOfHiddenNodes];
		matrixVectorProduct(wtsInputToHidden, input, intermediate);
		for (int i = 0 ; i < intermediate.length ; i++){
			intermediate[i] = activation(intermediate[i]);
		}

		double output = 0;
		matrixVectorProduct(wtsHiddenToOutput, intermediate, output);
		for (int i = 0 ; i < output.length ; i++){
			output[i] = activation(output[i]); 
		}
		 return output;
	}*/

	/*
	This is SIGMOID Activation function
	*/
	public static double activation (double n){
		double sig = (1.0d / (1.0d + Math.exp(-n)));
		//System.out.println("a = " + sig);
		return sig;
	}

	public static double derivative(double n) {
		double s = activation(n);
		return s * (1 - s);
	}

	public double[] matrixVectorProduct (double[][] weights, double[] nodesInPreviousLayer) 
	{
		//System.out.println("weights.length " + weights.length + " nodesInPreviousLayer.lengt " + nodesInPreviousLayer.length + " numOfHiddenNodes "+ numOfHiddenNodes);
		double[] valuesInLayer = new double[numOfHiddenNodes];

		for (int i = 0; i < weights.length ; i++){
			for (int j = 0; j < nodesInPreviousLayer.length; j++){
				valuesInLayer[i] = valuesInLayer[i] + (weights[i][j] * nodesInPreviousLayer[j]);
			}
		}
		//System.out.println( "What is saved in valuesInLayer array " + valuesInLayer[0] + "___" + valuesInLayer[1] + "___" + valuesInLayer[2] + "___" + valuesInLayer[3] + "___" + valuesInLayer[4] );
		return valuesInLayer;
	}
	public double matrixVectorProductForOutput (double[] weights, double[] nodesInPreviousLayer)  
	{
		//System.out.println("weights.length " + weights.length + " nodesInPreviousLayer.lengt " + nodesInPreviousLayer.length + " numOfHiddenNodes "+ numOfHiddenNodes);
		double valuesInLayer = 0 ;
		//System.out.println( "weights =========================== " + weights[0] + "___" + weights[1] + "___" + weights[2] + "___" + weights[3] + "___" + weights[4] );
		//System.out.println( "nodesInPreviousLayer>>>>>>>>>>>>>>>> " + nodesInPreviousLayer[0] + "___" + nodesInPreviousLayer[1] + "___" + nodesInPreviousLayer[2] + "___" + nodesInPreviousLayer[3] + "___" + nodesInPreviousLayer[4] );

		for (int i = 0; i < weights.length ; i++){
			valuesInLayer = valuesInLayer + (weights[i] * nodesInPreviousLayer[i]);
			//System.out.println("i --------------- " + i + "valuesInLayer " + valuesInLayer );
		}
		return valuesInLayer;
	}
}