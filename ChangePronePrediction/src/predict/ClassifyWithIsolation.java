package predict;

import Util.Result;
import java.io.File;
import java.sql.Date;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import classify.base.MyIsolationForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import Util.Config;
import Util.FileUtil;
public class ClassifyWithIsolation {
	private static int anomaly;
	private static double threshold = 0.5;
	public ClassifyWithIsolation(Instances ins) {
		super();
		this.anomaly = FileUtil.anomalyIndex(ins);
	}

	public static double getThreshold() {
		return threshold;
	}

	public static void setThreshold(double threshold) {
		ClassifyWithIsolation.threshold = threshold;
	}

	public static ArrayList<double[]> Classify(Instances ins,MyIsolationForest iso, ArrayList<double[]> prediction, double thres){

		int num = ins.numInstances();
      	double[][] pre = new double[num][3];
		double[] temps = new double[3];
		
    	int temp = 0;
    	String label = null;
    	for(int j=0;j < num; j++){
    		temps = iso.distributionForInstance(ins.instance(j));
    		pre[j][0] = temps[anomaly];
    		//pre[j][3] = temps[2];
    		//System.out.println("score is "+pre[j][0]);
    		//label = ins.classAttribute().toString();
    		temp = (int) ins.instance(j).classValue();
    	    pre[j][1] = temp;
    		pre[j][2] = -1;
    	}
    	for(int j =0; j < pre.length; j++){
    		//System.out.println("score is "+pre[j][0]);
    		pre[j][2] = (pre[j][0] >= thres) ? anomaly : (1-anomaly); 
    		prediction.add(pre[j]);
    	}
		return prediction;
		
		
	}
	
	public static Result Evaluate(ArrayList<double[]> predict, String e, String s) throws Exception{
		Object[] pre = predict.toArray();
		double[] temp = new double[3];
		double t_t = 0, t_f = 0, f_t = 0, f_f = 0;
		//StringBuffer text = new StringBuffer();
		for(int i = 0;i < pre.length; i++){
			temp = (double[])pre[i];
			if(temp[1] == 1 && temp[2] == 1) t_t++;
			else{
				if(temp[1] == 1 && temp[2] == 0) t_f++;
				else{
					if(temp[1] == 0 && temp[2] == 1) f_t++;
					else{
						if(temp[1] == 0 && temp[2] == 0) f_f++;
					}
				}
			}
		}
		
		double P = t_t + t_f;
		double N = f_t + f_f;
		double P1 = t_t + f_t;
		double N1 = t_f + f_f;
		double Matrix[][] = {{t_t,t_f,P},{f_t,f_f,N},{P1,N1,P+N}};
		//text.append("");
		double gmean = 0.0;
		double accuracy = 0.0;
		double[] recall = {0,0};
		double[] false_alarm = {0,0};
		double[] precision = {0,0};
		double[] fmeasure = {0,0};
		double[] balance = {0,0};
		double[] auc = {0,0};
		double mcc = 0.0;
		
		
		accuracy = (t_t + f_f)/(P + N);
		recall[0] = (f_f)/(N);
		precision[0] = (N1>0)?(f_f)/(N1):0;
		false_alarm[1] = f_t/N;
		false_alarm[0] = t_f/P;
		recall[1] = (t_t)/(P);
		precision[1] = (P1>0)?(t_t)/(P1):0;
		fmeasure[0] = (precision[0]+recall[0])>0?(precision[0]*recall[0]*2)/(precision[0]+recall[0]):0;
		fmeasure[1] = (precision[1]+recall[1])>0?(precision[1]*recall[1]*2)/(precision[1]+recall[1]):0;
		gmean = Math.sqrt(recall[0]*recall[1]);
		//gmean = Math.sqrt(precision[0]*precision[1]);
		balance[1] = 1-Math.sqrt((Math.pow((1-recall[1]), 2)+Math.pow((0-false_alarm[1]), 2))/2);
		balance[0] = 1-Math.sqrt((Math.pow((1-recall[0]), 2)+Math.pow((0-false_alarm[0]), 2))/2);
		auc[0] = new Calculation().calAUC(predict.toArray(),anomaly);
		auc[1] = auc[0];
		mcc = ((t_t * f_f) - (f_t * t_f)) / Math.sqrt((t_t + f_t) * (t_t + t_f) * (f_f + f_t) * (f_f + t_f));
		return new Result(0,0,e,s,anomaly,gmean,accuracy,recall,precision,fmeasure,balance,auc,Matrix,mcc);  
		
		
	}
	
	public static double findThreshold(Instances ins, int t) throws Exception{
	 	/*return 0.5;*/
	 	
    	int cnt =50; 
    	String[][] result = new String[cnt][17];   	
	    int numFolds = 10;
	    double bestThres = 0.0;
	    double thres = 0;
	    double gmean = 0.0, balance = 0.0, buggy_presicion = 0.0, fmeasure = 0.0;
	    Instances data = ins;    
	    //data = new LogPre().run(data,0.5);
	   	ArrayList<double[]> predict = new ArrayList<double[]>();
	   	String path = Config.result_folder + Config.file +"/temp/";
	   	File fold = new File(path);
	    data.setClassIndex(data.numAttributes()-1);
	   	if(! fold.exists()) fold.mkdirs();
    	String out_path = null;
    	
	    if (data.classAttribute().isNominal()) {    
	          data.stratify(numFolds);    
	    } 
	            
	    Result tempRes = null;
	       
	    MyIsolationForest iso = new MyIsolationForest();
        iso.setNumTrees(40);
        
    	data.randomize(new Random());
		for(int i = 0; i < cnt; i++){
			thres = 0.2 + i*0.01;
			predict.clear();
			for(int n = 0; n < numFolds; n++){
		         Instances train = data.trainCV(numFolds, n);    
		         Instances test = data.testCV(numFolds, n);
		         iso.setSubsampleSize(20);
		         iso.buildClassifier(train);
		         
		         predict = Classify(test,iso, predict, thres);
		     }
			
			 out_path = path + "findThres_" + t +".xls";
			 tempRes = Evaluate(predict,"","");
			 double extra =tempRes.recall(0)*tempRes.recall(1)*tempRes.precision(0)*tempRes.precision(1);
			 if(extra != 0 && tempRes.getGmean() > gmean){// ,tempRes.fMeasure(anomaly) > fmeasure
				 gmean = tempRes.getGmean();
				 balance = tempRes.getBalance()[anomaly];
				 fmeasure = tempRes.getFmeasure()[anomaly];
				 buggy_presicion = tempRes.getPrecision()[anomaly];
				 bestThres = thres;
			 }
			    result[i][0] = "0";
	            result[i][1] = String.valueOf(ins.numAttributes());
	            result[i][2] = String.valueOf(iso.getSubsampleSize());
	            result[i][3] = String.valueOf(iso.getNumTrees());
	            result[i][4] = String.valueOf(thres);
	            result[i][5] = String.valueOf(tempRes.getAccuracy());
	            result[i][6] = String.valueOf(tempRes.getGmean());
	            result[i][7] = String.valueOf(tempRes.getRecall()[0]);
	            result[i][8] = String.valueOf(tempRes.getRecall()[1]);
	            result[i][9] = String.valueOf(tempRes.getPrecision()[0]);
	            result[i][10] = String.valueOf(tempRes.getPrecision()[1]);
	            result[i][11] = String.valueOf(tempRes.getFmeasure()[0]);
	            result[i][12] = String.valueOf(tempRes.getFmeasure()[1]);
	            result[i][13] = String.valueOf(tempRes.getBalance()[0]);
	            result[i][14] = String.valueOf(tempRes.getBalance()[1]);
	            result[i][15] = String.valueOf(Calculation.calAUC(predict.toArray(),tempRes.getAnomalyClass()));
	            result[i][16] = String.valueOf(tempRes.getAnomalyClass());
	            //result[i][16] = String.valueOf(tempRes.getMcc());
		    			            			          	            
	     }
		FileUtil.outFile(out_path, result, Config.head);
	      //System.out.println("======File Writed======");
	     
		return bestThres;
		
		
	}
	public static Result crossValidateModel(MyIsolationForest isoF, Instances ins, int numFolds) throws Exception{
			Instances data = ins;
			data.randomize(new Random()); 
			if (data.classAttribute().isNominal()) {    
				  data.stratify(numFolds);    
			} 
			double thres = 0.0,	sum_thres = 0.0; 
			String[][] temp = new String[1][18];
			ArrayList<double[]> predict1 = new ArrayList<double[]>();
			Result tempRes = null;
			
			for(int n = 0; n < numFolds; n++){
				    //System.out.println("===Fold-"+n+"===");
				    Instances train = data.trainCV(numFolds, n);  
				    Instances test = data.testCV(numFolds, n);    	
				    //thres =  findThreshold(train,n);
				    thres = threshold;
				    sum_thres += thres;
				    isoF.buildClassifier(train);
				    predict1 = Classify(test,isoF, predict1, thres);				    			            	
			}
			sum_thres = (double)(sum_thres/numFolds);
			tempRes = Evaluate(predict1,"","");
			return tempRes;
	}
	
	

}
