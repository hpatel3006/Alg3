import javax.swing.*;
import java.awt.Point;
import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;
import java.util.TreeSet;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.Scanner;
import java.util.PriorityQueue;
import java.util.Map;
import java.util.HashMap;
import java.io.Serializable;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Collections;
import java.util.Random;

/// Entry point to program.
/**
 * This class acts as the entry point to the program and is responsible for parsing command line arguments
 * and setting up the GameController.  Run the program with the --help command-line parameter for more
 * information.
 * 
 * @author Leonid Shamis
 */
public class face
{
	final int imageSize = 128*120;
	static double[][] wightsforFemaleInputtoHidden;
	static double[][] wightsforMaleInputtoHidden;
	static double[] wightsforFemaleHiddentoOutput;
	static double[] wightsforMaleHiddentoOutput;
	public static void main(String[] args){
		 	
	 	int agrLength = args.length;
	 	int argIter = 0;
	 	wightsforFemaleInputtoHidden = new double[5][15360];
	 	wightsforMaleInputtoHidden = new double[5][15360];
	 	wightsforMaleHiddentoOutput = new double[5];
	 	wightsforFemaleHiddentoOutput = new double[5];


	 	while(argIter < agrLength){
	 		if(args[argIter].equals("-train")){
	 			listFilesforMale(args[argIter+1]);
	 			listFilesforFemale(args[argIter+2]);
	 		}

	 		else if(args[argIter].equals("-test")){

	 			listFilesforTest(args[argIter+1]);

	 		}



	 		argIter++;



	 	}



	 }



	 public static double[] readfile(String filename){

	 	double[] inputNeurons = new double[128*120];
	 	try{
	 		Scanner s = new Scanner(new File(filename));
	 		for(int i=0 ; i < 128*120 ; i++)
	 			inputNeurons[i] = s.nextInt();
	 		//throw new IDexception("ERROR");
	 	}
	 	catch (Exception e) {
			e.printStackTrace();
		}
	 	
	 	return inputNeurons;
	 }



	public static void listFilesforMale(String folder){
		File directory = new File(folder);
		File[] contents = directory.listFiles();
		int i = 0;
		NN male;
		for ( File f : contents) {
			if(i<5) {
				double[] arr = new double[ 128*120];
				arr = readfile(f.getAbsolutePath());
				male = new NN( 128*120, 5, 1, 1, arr, 0);
				//double[][] trainedweightsinputtohidden = new double[1][1];
	 			male.train(arr,300);
	 			double[][] temp1 = new double[5][15360];
	 			temp1 = male.getInputWeights();
	 			for(int y = 0; y < 5; y++){
	 				for(int j =0; j < 15360; j++){
	 					wightsforMaleInputtoHidden[y][j] = wightsforMaleInputtoHidden[y][j] + temp1[y][j];
	 				}
	 			}

	 			double[] temp2 = new double[5];
	 			temp2 = male.getHiddenWeights();
	 			for (int w = 0; w < 5; w++) wightsforMaleHiddentoOutput[w] = wightsforMaleHiddentoOutput[w] + temp2[w];


			}
			else break;
			i++;
		}
		for(int y = 0; y < 5; y++){
			for(int j =0; j < 15360; j++){
				wightsforMaleInputtoHidden[y][j] = wightsforMaleInputtoHidden[y][j]/i;
			}
		}
		for (int w = 0; w < 5; w++) wightsforMaleHiddentoOutput[w] = wightsforMaleHiddentoOutput[w]/i;	
		
	}

	public static void listFilesforTest(String folder){
		File directory = new File(folder);
		File[] contents = directory.listFiles();
		int i = 0;
		NN testdata;
		for ( File f : contents) {
			if(i<5) {
				double[] arr = new double[ 128*120];
				arr = readfile(f.getAbsolutePath());
				testdata = new NN( 128*120, 5, 1, 1, arr, 1);
				//double[][] trainedweightsinputtohidden = new double[5][128*120];
				testdata.test(wightsforMaleInputtoHidden, wightsforMaleHiddentoOutput);
	 			testdata.test(wightsforFemaleInputtoHidden, wightsforFemaleHiddentoOutput);
			}
			else break;
			i++;
		}

		
	}


	public static void listFilesforFemale(String folder){
		File directory = new File(folder);
		File[] contents = directory.listFiles();
		int i = 0;
		NN female;
		for ( File f : contents) {
			if(i<5) {
				double[] arr = new double[ 128*120];
				arr = readfile(f.getAbsolutePath());
				female = new NN( 128*120, 5, 1, 1, arr, 1);
				//double[][] trainedweightsinputtohidden = new double[5][128*120];
	 			female.train(arr, 300);
	 			double[][] temp1 = new double[5][15360];
	 			temp1 = female.getInputWeights();
	 			for(int y = 0; y < 5; y++){
	 				for(int j =0; j < 15360; j++){
	 					wightsforFemaleInputtoHidden[y][j] = wightsforFemaleInputtoHidden[y][j] + temp1[y][j];
	 				}
	 			}
	 			double[] temp2 = new double[5];
	 			temp2 = female.getHiddenWeights();
	 			for (int w = 0; w < 5; w++) wightsforFemaleHiddentoOutput[w] = wightsforFemaleHiddentoOutput[w] + temp2[w];
			}
			else break;
			i++;
		}

		for(int y = 0; y < 5; y++){
			for(int j =0; j < 15360; j++){
				wightsforFemaleInputtoHidden[y][j] = wightsforFemaleInputtoHidden[y][j]/i;
			}
		}
		for (int w = 0; w < 5; w++) wightsforFemaleHiddentoOutput[w] = wightsforFemaleHiddentoOutput[w]/i;	
	}

}
