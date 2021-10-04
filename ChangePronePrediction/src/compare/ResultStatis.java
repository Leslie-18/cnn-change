package compare;
import Util.Config;
import Util.FileUtil;
import preprocess.Sample;
import java.io.File;
import java.util.Calendar;
import java.util.Random;

import classify.base.MyIsolationForest;
import classify.base.OtherBagging;
import classify.base.OtherEvaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class ResultStatis{
		//static String[] c = {"weka.classifiers.bayes.NaiveBayes",
			//"weka.classifiers.trees.J48",
			//"weka.classifiers.functions.SMO","classify.base.MyIsolationForest","weka.classifiers.functions.Logistic"};
			//,"weka.classifiers.trees.MyIsolationForest"//"classify.base.NewIsolationForest"};
		static String[] c = {"weka.classifiers.functions.SMO"};
				//,"weka.classifiers.trees.MyIsolationForest"//"classify.base.NewIsolationForest"};
		static String m_class = "change_prone";//"bug_introducing";//"bug_introducingCopy";//
		
		public static void main(String[] args) throws Exception {
			long start_time = System.currentTimeMillis();
			File ff = new File(Config.data_folder);//select_folder
	    	//File ff = new File(select_folder);//
	    	File[] files = ff.listFiles();
	    	
	    	if(!new File(Config.result_folder).exists()){
	    		new File(Config.result_folder).mkdirs();
	    	}
	    	if(!ff.exists()){
	    		System.out.println("======Directory:" + Config.data_folder + " doesn't exists======");
	    	}else{
		    	for(int i = 0;i < files.length; i++){
	
		    		run(Config.result_folder,i,files[i].getAbsolutePath());
		    	}
	    	}
			long end_time = System.currentTimeMillis();
		 	System.out.println("======Time Used:" + (end_time - start_time)/1000 + "s======");
		}
		public static void run(String resultFolder, int sheet,String project) throws Exception{

			Instances data = null; 
			int numIns = 0;
			int numAttr = 0;
			long start = 0;
			long end = 0;
			double time = 0;
			Calendar calendar = Calendar.getInstance();		
			
			String tempPath  = Config.data_folder.substring(Config.total_folder.length());
			String out = resultFolder + tempPath.substring(0,tempPath.indexOf("//")) + "_result"+"10"+".xls";
			System.out.println("Output Path:" + out);
			File output = new File(out);
	    	String[] head = {"project","classifier","bag","accuracy","gmean","recall-0","recall-1","precision-0","precision-1","fMeasure-0","fMeasure-1","AUC","MCC","time"};	    	
			//String[] methods = {"Simple","Bagging","Undersample","Oversample","UnderBag",	"OverBag", "SMOTE", "SMOTEBag","AdaBoost"};//,"SMOTELog","OverLog","UnderLog"
			String[] methods = {"UnderBag","OverBag","SMOTEBag"};
			int cnt = c.length;
			String[][] result = new String[(cnt)*methods.length][head.length];
			double[] temp = new double[head.length];
			//double[] tem = new double[head.length];
			String[][] tem = new String[1][head.length];
			String tempOutPath = Config.result_folder + Config.file + "/" + project.substring(project.lastIndexOf("\\")+1, project.lastIndexOf(".")) +"__"+ ".xls";//该项目结果保存的路径

			OtherEvaluation eval = null;
			Class<?> c1 = null;
			weka.classifiers.Classifier clas = null;
			int m =0;
			int classIndex = 0;
			double run_times = 10;
			
			//inputFile = Config.select_folder + project;
			System.out.println(project);
			data = FileUtil.ReadData(project);
			numIns = data.numInstances();
			numAttr = data.numAttributes();
			
			classIndex = numAttr-1;
			data.setClassIndex(classIndex);
			Sample sam = new Sample(m_class);//data.classAttribute().name()
			
			for(int i = 0;i < cnt; i++){
				System.out.println("Classifier:" + c[i]);
				c1 = Class.forName(c[i]);
				clas = (weka.classifiers.Classifier) Class.forName(c[i]).newInstance();
				if(c[i].contains("IsolationForest")){
					((MyIsolationForest)clas).setSubsampleSize(20);
					
				}
				for(int j = 0;j < methods.length;j++){
					start = System.currentTimeMillis();
					
					for(int t = 0;t < run_times; t++){
						
						data.randomize(new Random());	
						switch(j){
						case 0:{
							//start = System.currentTimeMillis();
							//inputFile = Config.select_folder + project.substring(0,project.lastIndexOf(".")) + "_" + "undersample.arff";
							/*Instances ins = null;
							if(c[i] == "weka.classifiers.trees.IsolationForest"){
								ins = sam.AntiUnderSample(data, 0.05);
							}else{
								ins = sam.UnderSample(data);
							}
							ins.randomize(new Random());
							
							Bagging c2 = new weka.classifiers.meta.Bagging();
							c2.setClassifier((weka.classifiers.Classifier) clas);
							c2.setBagSizePercent(50);
							
							eval = new Evaluation(ins);
							eval.crossValidateModel(c2, ins, 10, new Random());*/

							OtherBagging c2 = null;
							if(i!=3){
								c2 = new OtherBagging(clas, data,m_class,1);
							}else{
								c2 = new OtherBagging(clas, data,m_class,3);
							}
							c2.setClassifier((weka.classifiers.Classifier) clas);
							eval = new OtherEvaluation(data,0);
							eval.crossValidateModel(c2, data, 10, new Random());
							//end = System.currentTimeMillis();
							//time = (end-start);
							break;
						}
						case 1:{
							//start = System.currentTimeMillis();
							//inputFile = Config.select_folder + project.substring(0,project.lastIndexOf(".")) + "_" + "oversample.arff";
							/*Instances ins = null;
							if(c[i] == "weka.classifiers.trees.IsolationForest"){
								ins = sam.AntiOverSample(data, 0.05);
							}else{
								ins = sam.OverSample(data);
							}
							ins.randomize(new Random());
							
							Bagging c2 = new Bagging();
							c2.setClassifier((weka.classifiers.Classifier) Class.forName(c[i]).newInstance());
							c2.setBagSizePercent(50);
							
							eval = new Evaluation(ins);
							eval.crossValidateModel(c2, ins, 10, new Random());
							*/

							OtherBagging c2 = null;
							if(i!=3){
								c2 = new OtherBagging(clas, data,m_class,2);
							}else{
								c2 = new OtherBagging(clas, data,m_class,4);
							}
							c2.setClassifier((weka.classifiers.Classifier) clas);
							eval = new OtherEvaluation(data,0);
							eval.crossValidateModel(c2, data, 10, new Random());
							//end = System.currentTimeMillis();
							//time = (end-start);
							break;
						}
						     case 2:{
							     OtherBagging c2 = new OtherBagging(clas, data,m_class,5);
							     c2.setClassifier((weka.classifiers.Classifier) clas);
							     eval = new OtherEvaluation(data,0);
							     eval.crossValidateModel(c2, data, 10, new Random());
						       	break;
						     }
//							case 0:{
//								//start = System.currentTimeMillis();
//								
//								/*
//								eval = new Evaluation(data);
//								eval.crossValidateModel(clas, data, 10, new Random());
//								 */
//								
//								eval = new OtherEvaluation(data,0);
//								eval.crossValidateModel(clas, data, 10, new Random());
//								
//								//end = System.currentTimeMillis();
//								//time = (end-start);
//								break;
//							
//							}
//							case 1:{
//								//start = System.currentTimeMillis();
//								/*
//								Bagging c2 = new Bagging();
//								c2.setClassifier((weka.classifiers.Classifier) clas);
//								c2.setBagSizePercent(50);
//								eval = new Evaluation(data);
//								eval.crossValidateModel(c2, data, 10, new Random());
//								*/
//								OtherBagging c2 = new OtherBagging(clas, data, m_class, 0);
//								c2.setClassifier((weka.classifiers.Classifier) clas);
//								//c2.setBagSizePercent(50);
//								eval = new OtherEvaluation(data,0);
//								eval.crossValidateModel(c2, data, 10, new Random());
//								//System.out.println(eval.toClassDetailsString());
//								//System.out.println(c2.toString());
//								//end = System.currentTimeMillis();
//								//time = (end-start);
//								break;
//							}
//							case 2:{
//								//start = System.currentTimeMillis();
//								//inputFile = Config.select_folder + project.substring(0,project.lastIndexOf(".")) + "_" + "undersample.arff";
//								//System.out.println(inputFile);
//								/*
//								Instances ins = null;
//								if(c[i] == "weka.classifiers.trees.IsolationForest"){
//									ins = sam.AntiUnderSample(data, 0.05);
//								}else{
//									ins = sam.UnderSample(data);
//								}
//								
//								ins.randomize(new Random());
//								eval = new Evaluation(ins);
//								eval.crossValidateModel(clas, ins, 10, new Random());
//								*/
//
//								if(i!=3){
//									eval = new OtherEvaluation(data,1);
//								}else{
//									eval = new OtherEvaluation(data,3);
//								}
//								eval.crossValidateModel(clas, data, 10, new Random());
//								break;
//							}
//							case 3:{
//								//start = System.currentTimeMillis();
//								//inputFile = Config.select_folder + project.substring(0,project.lastIndexOf(".")) + "_" + "oversample.arff";
//								/*Instances ins = null;
//								if(c[i] == "weka.classifiers.trees.IsolationForest"){
//									ins = sam.AntiOverSample(data, 0.05);
//								}else{
//									ins = sam.OverSample(data);
//								}
//								ins.randomize(new Random());
//						
//								eval = new Evaluation(ins);
//								eval.crossValidateModel(clas, ins, 10, new Random());*/
//
//								if(i!=3){
//									eval = new OtherEvaluation(data,2);
//								}else{
//									eval = new OtherEvaluation(data,4);
//								}
//								eval.crossValidateModel(clas, data, 10, new Random());
//								//end = System.currentTimeMillis();
//								//time = (end-start);
//								break;
//							}
//							case 4:{
//								//start = System.currentTimeMillis();
//								//inputFile = Config.select_folder + project.substring(0,project.lastIndexOf(".")) + "_" + "undersample.arff";
//								/*Instances ins = null;
//								if(c[i] == "weka.classifiers.trees.IsolationForest"){
//									ins = sam.AntiUnderSample(data, 0.05);
//								}else{
//									ins = sam.UnderSample(data);
//								}
//								ins.randomize(new Random());
//								
//								Bagging c2 = new weka.classifiers.meta.Bagging();
//								c2.setClassifier((weka.classifiers.Classifier) clas);
//								c2.setBagSizePercent(50);
//								
//								eval = new Evaluation(ins);
//								eval.crossValidateModel(c2, ins, 10, new Random());*/
//
//								OtherBagging c2 = null;
//								if(i!=3){
//									c2 = new OtherBagging(clas, data,m_class,1);
//								}else{
//									c2 = new OtherBagging(clas, data,m_class,3);
//								}
//								c2.setClassifier((weka.classifiers.Classifier) clas);
//								eval = new OtherEvaluation(data,0);
//								eval.crossValidateModel(c2, data, 10, new Random());
//								//end = System.currentTimeMillis();
//								//time = (end-start);
//								break;
//							}
//							case 5:{
//								//start = System.currentTimeMillis();
//								//inputFile = Config.select_folder + project.substring(0,project.lastIndexOf(".")) + "_" + "oversample.arff";
//								/*Instances ins = null;
//								if(c[i] == "weka.classifiers.trees.IsolationForest"){
//									ins = sam.AntiOverSample(data, 0.05);
//								}else{
//									ins = sam.OverSample(data);
//								}
//								ins.randomize(new Random());
//								
//								Bagging c2 = new Bagging();
//								c2.setClassifier((weka.classifiers.Classifier) Class.forName(c[i]).newInstance());
//								c2.setBagSizePercent(50);
//								
//								eval = new Evaluation(ins);
//								eval.crossValidateModel(c2, ins, 10, new Random());
//								*/
//
//								OtherBagging c2 = null;
//								if(i!=3){
//									c2 = new OtherBagging(clas, data,m_class,2);
//								}else{
//									c2 = new OtherBagging(clas, data,m_class,4);
//								}
//								c2.setClassifier((weka.classifiers.Classifier) clas);
//								eval = new OtherEvaluation(data,0);
//								eval.crossValidateModel(c2, data, 10, new Random());
//								//end = System.currentTimeMillis();
//								//time = (end-start);
//								break;
//							}
//							case 6:{
//								eval = new OtherEvaluation(data,5);
//								eval.crossValidateModel(clas, data, 10, new Random());
//								break;
//							}
//							case 7:{
//								OtherBagging c2 = new OtherBagging(clas, data,m_class,5);
//								c2.setClassifier((weka.classifiers.Classifier) clas);
//								eval = new OtherEvaluation(data,0);
//								eval.crossValidateModel(c2, data, 10, new Random());
//								break;
//							}
//							case 8:{
//								weka.classifiers.meta.AdaBoostM1 booster = new AdaBoostM1();
//								booster.setClassifier((weka.classifiers.Classifier) clas);
//								eval = new OtherEvaluation(data,0);
//								eval.crossValidateModel(booster, data, 10, new Random()); 
//							}
						
						}
						//if(t == run_times-1) 
							//System.out.println(eval.toClassDetailsString());
						tem[0][0] = project.substring(project.lastIndexOf("\\")+1,project.lastIndexOf("."));
						tem[0][1] = c[i].substring(c[i].lastIndexOf(".")+1);
						tem[0][2] = methods[j];
						tem[0][3] = String.valueOf(1-eval.errorRate());
				        tem[0][4] = String.valueOf(Math.sqrt(eval.recall(0)*eval.recall(1)));
				        tem[0][5] = String.valueOf(eval.recall(0));//sampleRatio//
				        tem[0][6] = String.valueOf(eval.recall(1));
				        tem[0][7] = String.valueOf(eval.precision(0));
				        tem[0][8] = String.valueOf(eval.precision(1));
				        tem[0][9] = String.valueOf(eval.fMeasure(0));
				        tem[0][10] = String.valueOf(eval.fMeasure(1));
				        tem[0][11] = String.valueOf(eval.areaUnderROC(0));	
				        double[][] matri = eval.confusionMatrix();
				        tem[0][12] = String.valueOf((matri[0][0]*matri[1][1] - matri[0][1]*matri[1][0])/Math.sqrt((matri[0][0]+matri[1][0])*(matri[0][0]+matri[0][1])*(matri[1][1]+matri[1][0])*(matri[1][1]+matri[0][1])));
				        tem[0][13] = String.valueOf(0);
				        FileUtil.exportFile(tem, tempOutPath, project , sheet, head);
				       // for(int pl = 0 ; pl < 1; pl++) {
				        	//for(int pll = 0 ; pll < head.length; pl++) {
				        		//tem[pl][pll] = String.valueOf(0);}
				       // }
				        
						temp[0] = 0;
						temp[1] = 0;
						temp[2] = 0;
						temp[3] += 1-eval.errorRate();
				        temp[4] += Math.sqrt(eval.recall(0)*eval.recall(1));
				        temp[5] += eval.recall(0);//sampleRatio//
				        temp[6] += eval.recall(1);
				        temp[7] += eval.precision(0);
				        temp[8] += eval.precision(1);
				        temp[9] += eval.fMeasure(0);
				        temp[10] += eval.fMeasure(1);
				        temp[11] += eval.areaUnderROC(0);	
				        double[][] matrix = eval.confusionMatrix();
				        temp[12] += (matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0])/Math.sqrt((matrix[0][0]+matrix[1][0])*(matrix[0][0]+matrix[0][1])*(matrix[1][1]+matrix[1][0])*(matrix[1][1]+matrix[0][1]));
				        temp[13] += 0;
				        
				    }
					end = System.currentTimeMillis();
					time = (end-start);
					//System.out.println(eval.toClassDetailsString());
					result[m][0] = project.substring(project.lastIndexOf("\\")+1,project.lastIndexOf("."));
					result[m][1] = c[i].substring(c[i].lastIndexOf(".")+1);
					result[m][2] = methods[j];
					result[m][3] = String.valueOf(temp[3]/run_times);
			        result[m][4] = String.valueOf(temp[4]/run_times);
			        result[m][5] = String.valueOf(temp[5]/run_times);//sampleRatio//
			        result[m][6] = String.valueOf(temp[6]/run_times);
			        result[m][7] = String.valueOf(temp[7]/run_times);
			        result[m][8] = String.valueOf(temp[8]/run_times);
			        result[m][9] = String.valueOf(temp[9]/run_times);
			        result[m][10] = String.valueOf(temp[10]/run_times);
			        result[m][11] = String.valueOf(temp[11]/run_times);	
			        result[m][12] = String.valueOf(temp[12]/run_times);
			        result[m][13] = String.valueOf(time/10) + "ms";
			        m++;
			        for(int p = 0 ; p < head.length; p++) temp[p] = 0;
					
				}
				
	
			}
		    FileUtil.exportFile(result, out, project, sheet, head);
	}
	
		
}
	
