package predict;
import Util.Result;

import java.io.File;
import java.io.IOException;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Random;

import classify.base.MyIsolationForest;
import compare.ResultStatis;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import Util.Config;
import Util.FileUtil;
import weka.core.converters.ConverterUtils.DataSink;
public class PreMain {

	/**
	 * @param args
	 */

	private static double threshold = 0.4;
	private static int ind = 0;
	private static double valueCnt[] = {0,0};
	private static double m_times = 10;
	private static int  m_width = 18;
	private static long start = 0;
	private static long end = 0;
	private static SimpleDateFormat dateFor = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	private static String date = null;

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Calendar calendar = Calendar.getInstance();
		start = System.currentTimeMillis(); 
		long start0 = System.currentTimeMillis();
		date =   dateFor.format(new Date(start0));
    	System.out.println("============Start Time:"+date+"============");
		//read the input data
    	//isoF.setNumTrees(30);
    	//isoF.setSubsampleSize(128);
    	//isoF.getDebug();
    	//isoF.listOptions();

        /*
         * 2.training
         * */
    	
    	File f = new File(Config.result_folder);
    	if(!f.exists()) f.mkdirs();
  
    	File ff = new File(Config.data_folder);//select_folder
    	File[] files = ff.listFiles();
    	for(int i = 0;i < files.length; i++){
    		//file = files[i];
    		Config.file = files[i].getName();
    		System.out.println("=========File:"+files[i]+"=======");
        	try {
				ExcuteIsoWithSub(files[i].getAbsolutePath(),i,false,"result");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	
    	}
    	
    	
    	/*
    	ArffLoader loaderNew = new ArffLoader();
		loaderNew.setFile(new File(select_folder+"antSelectedFilte_Linea.arff"));
		Instances Ins = loaderNew.getDataSet();
		Ins.setClassIndex(Ins.numAttributes()-1);
		runROC(Ins);
		*/
    	end = System.currentTimeMillis();
		calendar.setTimeInMillis(end - start);
    	System.out.println("======Total Time Used:"+(end-start)+"ms======");///(60*1000)+"min"+(end-start)%(60*1000)/1000+"s======");		
		//System.out.println("============Total Time Used:"+calendar.get(Calendar.MINUTE)+"min"+calendar.get(Calendar.SECOND)+"s==========");
	}
	
	/*static void ExcuteIsoWithSub(String datafile,int sheet,boolean if_Para,boolean if_Sel, String resultFileName) throws Exception{
    	double acc = 0,gmean = 0;
    	double[] recall = {0,0},preci = {0,0},fm = {0,0};
    	int tempNum = 0;
    	int cnt =10; //(int) (Math.log(attrNum)/Math.log(2));
		String[] Eval = Config.SubEvalS;
		String[] Sear = Config.SearchS;
		String FileFold = result_path+"SubEval_Search/";
		Instances newIns = null;

    	File f = new File(FileFold);
    	if(!f.exists()){
    		f.mkdirs();
    	}
    	String tempFold = result_path+"temp/";
    	File f1 = new File(tempFold);
    	if(!f1.exists()){
    		f1.mkdirs();
    	}
    	
    	String outPath = Config.result_path + resultFileName  + ".xls";
    	String[][] result = new String[Eval.length*Sear.length][18];
    	int m = 0;
    	int flag0 = 0;
    	for(int s = 0; s < Eval.length; s++)
    	{
    		for(int p = 0; p < Sear.length; p++)
    		{
    			String eval = Eval[s];
    			String search = Sear[p];
    			String tempOutPath = FileFold + eval.substring(0, 5) + "_"+search.substring(0, 5) + ".xls";
    			String outArffPath = Config.select_folder + datafile.substring(0, datafile.lastIndexOf(".")) + "_" + eval.substring(0, 5) + "_"+search.substring(0, 5) + ".arff";    	
    			
    			newIns = GetInputData(Config.data_folder+datafile, if_Sel, Eval[s], Sear[p], outArffPath);
    			newIns.setClassIndex(newIns.numAttributes()-1);
    			int anomalyClass = anomalyIndex(newIns);
    			
    			ClassifyWithIsolation classifier = new ClassifyWithIsolation(anomalyClass);
    			Calculation cal = new Calculation();
    			if(new File(tempOutPath).exists()) continue;
    			double undersam = 0;
			    double thres = 0;
			    //cnt = (int) ((trainIns.numInstances()/10)*0.9);
		        String[][] temp = new String[(int) m_times][18];
		        int numIns = newIns.numInstances();
		        ArrayList<double[]> predict1 = new ArrayList<double[]>();
		            	
		        System.out.println("======Start to evaluate======");
				int numFolds = 10;
				//data = new LogPre().run(data,0.5,0);
				                       
			    Result tempRes = null;
				//data = new Sample(datafile).overSample(data, 0.05);
				//data = new Sample(datafile).underSample(data, 0.05);
			        	    
				for(int i = 0 ; i < m_times ; i++){
				        	
				     start = System.currentTimeMillis();
				     //undersam = (0.01*(i+1));
				     undersam = 0.1;
				     double sampleRatio = 0.03;//0.03;//((i+1)*0.02);
				     isoF.setSubsampleSize((int) (numIns*sampleRatio));
				     //isoF.setSubsampleSize((int) 20);
				     //threshold = 0.0+(i-1)*0.02;
				     //isoF.setNumTrees((i+1)*10);
				        	
				     double sum_thres = 0;
				     Instances data = newIns;
				     data.randomize(new Random());    
				     if (data.classAttribute().isNominal()) {    
				         data.stratify(numFolds);    
				     } 
				     
				     for(int n = 0; n < numFolds; n++){
				          //System.out.println("===Fold-"+n+"===");
				          Instances train = data.trainCV(numFolds, n);  
				          Instances test = data.testCV(numFolds, n);
				            	
				          thres =  classifier.findThreshold(train,n);
				          //thres = 0.4;
				          //thres = threshold;
				          sum_thres += thres;
				          isoF.buildClassifier(train);
				          predict1 = classifier.Classify(test,isoF, predict1, thres );				    			            	
				     }
				            sum_thres = (double)(sum_thres/numFolds);
				            tempRes = classifier.Evaluate(predict1,eval,search);
				            
				            end = System.currentTimeMillis();
				            double time = end-start;
				            
				            temp[i][0] = datafile.substring(0, datafile.lastIndexOf(".")) + "_" + eval.substring(0, 5) + "_"+search.substring(0, 5);
				            temp[i][1] = String.valueOf(newIns.numAttributes());
				            temp[i][2] = String.valueOf(isoF.getSubsampleSize());//sampleRatio
				            temp[i][3] = String.valueOf(isoF.getNumTrees());
				            temp[i][4] = String.valueOf(sum_thres);
				            temp[i][5] = String.valueOf(tempRes.getAccuracy());
				            temp[i][6] = String.valueOf(tempRes.getGmean());
				            temp[i][7] = String.valueOf(tempRes.getRecall()[0]);
				            temp[i][8] = String.valueOf(tempRes.getRecall()[1]);
				            temp[i][9] = String.valueOf(tempRes.getPrecision()[0]);
				            temp[i][10] = String.valueOf(tempRes.getPrecision()[1]);
				            temp[i][11] = String.valueOf( tempRes.getFmeasure()[0]);
				            temp[i][12] = String.valueOf(tempRes.getFmeasure()[1]);
				            temp[i][13] = String.valueOf(tempRes.getBalance()[0]);
				            temp[i][14] = String.valueOf(tempRes.getBalance()[1]);
				            temp[i][15] = String.valueOf(cal.calAUC(predict1.toArray(),tempRes.getAnomalyClass()));
				            temp[i][16] = String.valueOf(tempRes.getAnomalyClass());
				            temp[i][17] = String.valueOf(time);     
				            predict1.clear();
					        //String out = result_path+"temp/"+eval+"_"+search+"_"+"i"+".xls";
				            //String[] head0 = {"score","label","prediction"} ;
					        //exportObjFile(predict1,out,head0);
				            OutPut.outFile(tempOutPath, temp, Config.head);
				            
						}
						
				        //判断是否是评价参数变化，如果不是，则计算每次运算的平均值写入结果
				         // 如果不是，则不取平均值
				        if(if_Para == true){
				        	for(int t = 0; t < m_width; t++){ 
					        	result[m][t] = "";
					        }
				        }else{
				        	String[] res = null;
				        	res = Calculation.calAverage(temp);
				        	for(int t = 0; t < m_width; t++){ 
					        	result[m][t] = res[t];
					        }
				        }
			            m++;

			            System.out.println("======Time Used:"+result[0][17]+"======");
   
			        
    		}
    	}
    	
    	OutPut.exportFile(result, outPath, datafile , sheet, Config.head);
        //outFile(outPath, result, head);
        System.out.println("======File Writed======");

       	end = System.currentTimeMillis();
    	date =   dateFor.format(new Date(end));
    	System.out.println("============End Time:"+date+"============");
	}
	*/
	static void ExcuteIsoWithSub(String datafile,int sheet,boolean if_Para, String resultFileName) throws Exception{
    	double acc = 0,gmean = 0;
    	double[] recall = {0,0},preci = {0,0},fm = {0,0};
    	int tempNum = 0;
    	int cnt =10; //(int) (Math.log(attrNum)/Math.log(2));
		String FileFold = Config.result_folder + Config.file + "/";//该项目的结果文件夹
		String FileName = datafile.substring(datafile.lastIndexOf("\\")+1, datafile.lastIndexOf("."));//获取文件名
    	String tempFold = FileFold + "temp/";//保存过程中产生的文件
    	String tempOutPath = FileFold + FileName + ".xls";//该项目结果保存的路径
    	String outPath = Config.result_folder  + resultFileName + ".xls";//最终统计结果保存的文件路径
    	
    	String trainoutPath = FileFold + "CV/"+ "train_"; //Config.result_folder  + "/CV/" + Config.file + "train_"; 
    	String testoutPath = FileFold + "CV/"+ "test_";//Config.result_folder  + "/CV/" + Config.file + "test_"; //+ ".arff";//
		Instances newIns = null;

    	File f = new File(FileFold);
    	if(!f.exists()){
    		f.mkdirs();
    	}else{}

    	File f1 = new File(tempFold);
    	if(!f1.exists()){
    		f1.mkdirs();
    	}
    	
    	
    	String[][] result = new String[1][18];
    	int m = 0;
    	int flag0 = 0;



    	//newIns = GetInputData(Config.data_folder+datafile, if_Sel, Eval[s], Sear[p], outArffPath);
    	newIns = FileUtil.ReadData(datafile);
    	newIns.setClassIndex(newIns.numAttributes()-1);
		int numIns = newIns.numInstances();
    	int anomalyClass = anomalyIndex(newIns);
    			
    	ClassifyWithIsolation classifier = new ClassifyWithIsolation(newIns);
    	Calculation cal = new Calculation();
  
		double thres = 0;
		String[][] temp = new String[(int) m_times][18];//temp数组保存运行m_times次的结果

		ArrayList<double[]> predict1 = new ArrayList<double[]>();
		System.out.println("======Start to evaluate======");
		int numFolds = 10;
		//data = new LogPre().run(data,0.5,0);
				                       
		Result tempRes = null;
		MyIsolationForest isoF = new MyIsolationForest();
    	isoF.setNumTrees(40);
    	isoF.setSubsampleSize(20);
    	System.out.println("Num of Trees:"+isoF.getNumTrees());
    	System.out.println("SubSample Size:"+isoF.getSubsampleSize());
    	
		for(int i = 0 ; i < m_times ; i++){
				        	
			start = System.currentTimeMillis();
			double sampleRatio = 0.03;//0.03;//((i+1)*0.02);
			isoF.setSubsampleSize((int) (numIns*sampleRatio));
				        	
			double sum_thres = 0;
			Instances data = newIns;
			data.randomize(new Random());    
			if (data.classAttribute().isNominal()) {    
				  data.stratify(numFolds);    
			} 
				     
			for(int n = 0; n < numFolds; n++){
				    //System.out.println("===Fold-"+n+"===");
				    Instances train = data.trainCV(numFolds, n);  
				    
				    DataSink.write(FileFold + "CV_Large_95/"+ i +"/"+"train_"+n+".arff", train);
				    	//DataSink.write(trainoutPath + n +".arff", train);
				    
				    Instances test = data.testCV(numFolds, n);
				    
				    DataSink.write(FileFold + "CV_Large_95/"+ i +"/"+"test_"+n+".arff", test);
				    	//DataSink.write(testoutPath + n +".arff", test);
				    
				    thres =  classifier.findThreshold(train,n);
				    sum_thres += thres;
				    isoF.buildClassifier(train);
				    predict1 = classifier.Classify(test,isoF, predict1, thres);				    			            	
			}
				    sum_thres = (double)(sum_thres/numFolds);
				    tempRes = classifier.Evaluate(predict1,"","");
				            
				    end = System.currentTimeMillis();
				    double time = end-start;
				            
				    temp[i][0] = FileName;
				    temp[i][1] = String.valueOf(newIns.numAttributes());
				    temp[i][2] = String.valueOf(isoF.getSubsampleSize());//sampleRatio
				    temp[i][3] = String.valueOf(isoF.getNumTrees());
				    temp[i][4] = String.valueOf(sum_thres);
				    temp[i][5] = String.valueOf(tempRes.getAccuracy());
				    temp[i][6] = String.valueOf(tempRes.getGmean());
				    temp[i][7] = String.valueOf(tempRes.getRecall()[0]);
				    temp[i][8] = String.valueOf(tempRes.getRecall()[1]);
				    temp[i][9] = String.valueOf(tempRes.getPrecision()[0]);
				    temp[i][10] = String.valueOf(tempRes.getPrecision()[1]);
				    temp[i][11] = String.valueOf( tempRes.getFmeasure()[0]);
				    temp[i][12] = String.valueOf(tempRes.getFmeasure()[1]);
				    temp[i][13] = String.valueOf(tempRes.getBalance()[0]);
				    temp[i][14] = String.valueOf(tempRes.getBalance()[1]);
				    temp[i][15] = String.valueOf(cal.calAUC(predict1.toArray(),tempRes.getAnomalyClass()));
				    //temp[i][16] = String.valueOf(tempRes.getAnomalyClass());
				    temp[i][16] = String.valueOf(tempRes.getMcc());
				    temp[i][17] = String.valueOf(time);     
				    
					String out = tempFold + "_" + i + ".xls";
				    String[] head0 = {"score","label","prediction"} ;
				    //将实例的score输出至临时文件
					FileUtil.exportObjFile(predict1,out,head0);
					//把每一个项目的结果输出至文件
				    FileUtil.outFile(tempOutPath, temp, Config.head);
				    predict1.clear();
			}
						
			/*判断是否是评价参数变化，如果不是，则计算每次运算的平均值写入结果
			* 如果是，则取平均值*/
			if(if_Para == true){
				for(int t = 0; t < m_width; t++){ 
				result[0][t] = "";
				}
			}else{
				String[] res = null;
				res = Calculation.calAverage(temp);
				for(int t = 0; t < m_width; t++){ 
					 result[0][t] = res[t];
				}
			}
			m++;

		System.out.println("======Time Used:"+result[0][17]+"======");
   
		//将该项目的若干次分类结果的平均值写入最终结果文件中
    	FileUtil.exportFile(result, outPath, datafile , sheet, Config.head);
        //outFile(tempOutPath, result, head);
        System.out.println("======File Writed======");

       	end = System.currentTimeMillis();
    	date =   dateFor.format(new Date(end));
    	System.out.println("============End Time:"+date+"============");
	}
	
	 private static int anomalyIndex(Instances ins){
		int valueCnt[] = {0,0};
	    int temp = 0;
	    int ind = 0;
	    valueCnt[0] = 0;
	    valueCnt[1] = 0;
	    for(int j=0;j < ins.numInstances(); j++){
	    		temp = (int) ins.instance(j).classValue();
	    		valueCnt[temp]++;
	    	}
	   	//ind = (valueCnt[0] > valueCnt[1]) ? valueCnt[1] : valueCnt[0];
	    ind = (valueCnt[0] > valueCnt[1]) ? 1 : 0;
	   	//System.out.println("======anomaly cnt="+valueCnt[0]+"======"+valueCnt[1]);
		return ind;
	 } 
	 
		 
	 private static Instances GetInputData(String filePath, boolean ifSelect, String evaluation, String search, String  outArffPath) throws Exception{

		if(!new File(filePath).exists()){
			System.out.println("======This File doesn't exist!" + "======");
			return null;
		}
		Instances ResultIns = null;
		//不做属性选择
		if(ifSelect == false){
			return FileUtil.ReadData(filePath);
		}else{//做属性选择
			File f0 = new File(outArffPath);
			if(f0.exists()){//已经属性选择过后的文件
				System.out.println("===Selected Attributes file already exists!===");
				ResultIns = FileUtil.ReadData(outArffPath);
			}else{//如果没有属性选择文件
				Instances newIns = FileUtil.ReadData(filePath);
		        //set the classIndex
		        System.out.println("======Attribute Selecting...======");
		        ResultIns = new AttrSelect(newIns, evaluation, search).SubsetSelect();
		        System.out.println("======New AttrNum="+(ResultIns.numAttributes())+"======");
	    		/*write to arff*/
	    		ArffSaver saver = new ArffSaver(); 
		        saver.setInstances(ResultIns);  
		        saver.setFile(new File(outArffPath));  
		        saver.writeBatch(); 
		        System.out.println("==Selected Attrbutes Writed:"+outArffPath+"==");
        	}
		}
		return ResultIns;
	 }
	 /*private static Instances AttributeSelection(String file, String evaluation, String search, String  outArffPath){
		Instances ResultIns = null;
    	System.out.println("======Evaluation:" + evaluation + "+++Search:"+ search+"======");
        
        try{
			File f0 = new File(outArffPath);
			if(f0.exists()){//已经属性选择过后的文件
				System.out.println("===Selected Attributes file already exists!===");
				ArffLoader loader0 = new ArffLoader();
				loader0.setFile(f0);
				Instances newIns = loader0.getDataSet();
				ResultIns = newIns;
			
		        int attrNum = newIns.numAttributes()-1;
		        newIns.numInstances();
		        System.out.println("=========Data Information=========");
		        System.out.println("======AttrNum:"+attrNum+"======");
		        System.out.println("======InstancesNum:"+newIns.numInstances()+"======");
			}else{//如果没有属性选择文件
					
				File fData = new File(data_path);
				//outArffPath = data_folder + datafile;
					
				ArffLoader loader0 = new ArffLoader();
				loader0.setFile(fData);
				Instances newIns = loader0.getDataSet();
				
		        int attrNum = newIns.numAttributes()-1;
		        System.out.println("=========Data Information=========");
		        System.out.println("======AttrNum:"+attrNum+"======");
		        System.out.println("======InstancesNum:"+newIns.numInstances()+"======");
		        //set the classIndex
		        System.out.println("======Attribute Selecting...======");
		        newIns = new AttrSelect(newIns, evaluation, search).SubsetSelect();
		        ResultIns = newIns;
    			System.out.println("======New AttrNum="+(newIns.numAttributes())+"======");
    			
    			//write to arff
    			ArffSaver saver = new ArffSaver(); 
	        	saver.setInstances(newIns);  
	        	saver.setFile(new File(outArffPath));  
	        	saver.writeBatch(); 
	        	System.out.println("==Selected Attrbutes Writed:"+outArffPath+"==");
        	}
	
    	} catch (Exception e) { 
    		System.out.println(e.toString());
    		
		} 
       
		return ResultIns;
		 
	 }*/
	 
}
