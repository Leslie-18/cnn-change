package predict;

import java.io.File;
import Util.Config;
import Util.FileUtil;
import Util.Result;
import java.util.ArrayList;
import java.util.Random;

import classify.base.MyIsolationForest;
import classify.base.OtherEvaluation;
import weka.core.Instances;
public class ParaAnalyze {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		long start_time = System.currentTimeMillis();
		File ff = new File(Config.data_folder);//select_folder
    	//File ff = new File(select_folder);//
    	File[] files = ff.listFiles();
    	for(int i = 0;i < files.length; i++){
    		Instances data = FileUtil.ReadData(files[i].getAbsolutePath());
    		String 	out_path = Config.result_folder + files[i].getName() + "Para" + ".xls";
    		try {
				excute(new MyIsolationForest(), data,out_path, 1);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    		
    	}
		long end_time = System.currentTimeMillis();
	 	System.out.println("======Time Used:" + (end_time - start_time)/1000 + "s======");
	
		
	}
	
	public static double excute(MyIsolationForest iso, Instances ins, String outPath, int flag) throws Exception{
	 	/*return 0.5;*/
	 	
  	   	int cnt =41; 
  	   	String[][] result = new String[cnt][17];   	
	    int numFolds = 10;
	    double bestThres = 0.0;
	    double thres = 0.5;
	    double gmean = 0.0, balance = 0.0, buggy_presicion = 0.0, fmeasure = 0.0;
	    Instances data0 = ins;    
	    //data = new LogPre().run(data,0.5);
	   	ArrayList<double[]> predict = new ArrayList<double[]>();
	   	String path = Config.result_folder;
	   	File fold = new File(path);
	    data0.setClassIndex(data0.numAttributes()-1);
	   	if(! fold.exists()) fold.mkdirs();
	   	String out_path = null;
  	
	    if (data0.classAttribute().isNominal()) {    
	          data0.stratify(numFolds);    
	    } 
	     int anomaly = FileUtil.anomalyIndex(ins);      
	    Result tempRes = null;
	    data0.randomize(new Random());
		for(int i = 0; i < cnt; i++){
			
			ClassifyWithIsolation classifier = new ClassifyWithIsolation(data0);
			if(flag == 1){
				thres = 0.0 + i*0.025;
				//iso.setM_threshold(thres);
				classifier.setThreshold(thres);
			}else if(flag == 2){
				thres = 0 + i*10;
				iso.setNumTrees((int)thres);
			}else if(flag == 3){
				thres = 0 + i*0.025;
				iso.setSubsampleSize((int)(thres*ins.numInstances()));
			}
			
		 	//OtherEvaluation eval = new OtherEvaluation(data0,0);
			//eval.crossValidateModel(iso, data0, 10, new Random());

			Result eval = classifier.crossValidateModel(iso,data0, 10);

			    result[i][0] = "0";
	            result[i][1] = String.valueOf(ins.numAttributes());
	            result[i][2] = String.valueOf(iso.getSubsampleSize());
	            result[i][3] = String.valueOf(iso.getNumTrees());
	            result[i][4] = String.valueOf(thres);
	            result[i][5] = String.valueOf(1 - eval.errorRate());
	            result[i][6] = String.valueOf(eval.recall(0)*eval.recall(1));
	            result[i][7] = String.valueOf(eval.recall(0));
	            result[i][8] = String.valueOf(eval.recall(1));
	            result[i][9] = String.valueOf(eval.precision(0));
	            result[i][10] = String.valueOf(eval.precision(1));
	            result[i][11] = String.valueOf(eval.fMeasure(0));
	            result[i][12] = String.valueOf(eval.fMeasure(1));
	            result[i][13] = String.valueOf(0);
	            result[i][14] = String.valueOf(0);
	            result[i][15] = String.valueOf(eval.areaUnderROC(1));
	            result[i][16] = String.valueOf(anomaly);
		    			            			          	            
	     }
		FileUtil.outFile(outPath, result, Config.head);
	    System.out.println("-------> bestThres : " + bestThres);
		return bestThres;
		
	
	}
	
  

}
