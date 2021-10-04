package classify.base;


import weka.classifiers.RandomizableClassifier;


import java.util.Enumeration;
import java.util.Random;
import java.util.ArrayList;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

import java.io.File;
import java.io.Serializable;

import predict.Calculation;
import Util.Config;
import Util.FileUtil;
import Util.Result;


/**
 <!-- globalinfo-start -->
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 9532 $
 */
public class NewIsolationForest extends MyIsolationForest {

  // For serialization

  /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
/** 
   * Returns distribution of scores.
   */
  double m_threshold = 0.5;


public double getM_threshold() {
	return m_threshold;
}



public void setM_threshold(double m_threshold) {
	this.m_threshold = m_threshold;
}
  
  public double[] distributionForInstance(Instance inst) {
    
    double avgPathLength = 0;
    for (int i = 0; i < m_trees.length; i++) {
      avgPathLength += m_trees[i].pathLength(inst);
    }
    avgPathLength /= (double) m_trees.length;

    double[] scores = new double[2];
    
    //scores[0] = Math.pow(2, - avgPathLength / c(m_subsampleSize));
     //scores[1] = 1.0 - scores[0];
    scores[m_anomaly] = Math.pow(2, - avgPathLength / c(m_subsampleSize)) + (0.5 - m_threshold);
    scores[1-m_anomaly] = 1.0 - scores[m_anomaly];
    //scores[2] = avgPathLength;
    //System.out.println("======c(m_subsampleSize):"+c(m_subsampleSize));
    return scores;      
  }

  public double findThreshold(NewIsolationForest iso,Instances ins) throws Exception{
	 	/*return 0.5;*/
	 	
  	   	int cnt =20; 
  	   	String[][] result = new String[cnt][17];   	
	    int numFolds = 10;
	    double bestThres = 0.0;
	    double thres = 0;
	    double gmean = 0.0, balance = 0.0, buggy_presicion = 0.0, fmeasure = 0.0;
	    Instances data0 = ins;    
	    //data = new LogPre().run(data,0.5);
	   	ArrayList<double[]> predict = new ArrayList<double[]>();
	   	String path = Config.result_folder + Config.file +"/temp/";
	   	File fold = new File(path);
	    data0.setClassIndex(data0.numAttributes()-1);
	   	if(! fold.exists()) fold.mkdirs();
	   	String out_path = null;
  	
	    if (data0.classAttribute().isNominal()) {    
	          data0.stratify(numFolds);    
	    } 
	            
	    Result tempRes = null;
	    data0.randomize(new Random());
		for(int i = 0; i < cnt; i++){
			thres = 0.4 + i*0.01;
			iso.setM_threshold(thres);
		 	OtherEvaluation eval = new OtherEvaluation(data0,0);
			eval.crossValidateModel(iso, data0, 10, new Random());
			
			 out_path = path + "findThres" + ".xls";
			 double extra = eval.recall(0)*eval.recall(1)*eval.precision(0)*eval.precision(1);
			 if(extra != 0 && eval.fMeasure(m_anomaly) > fmeasure){// tempRes.getGmean() > gmean,tempRes.getGmean() > gmean    tempRes.getPrecision()[anomaly] > buggy_presicion tempRes.getBalance()[anomaly] > balance
				 gmean = Math.sqrt(eval.recall(0)*eval.recall(1));
				 //balance = tempRes.getBalance()[anomaly];
				 fmeasure = eval.fMeasure(m_anomaly);
				 buggy_presicion = eval.precision(m_anomaly);
				 bestThres = thres;
				 System.out.print(fmeasure + "  ");
				
			 }
			    result[i][0] = "0";
	            result[i][1] = String.valueOf(ins.numAttributes());
	            result[i][2] = String.valueOf(iso.getSubsampleSize());
	            result[i][3] = String.valueOf(iso.getNumTrees());
	            result[i][4] = String.valueOf(thres);
	            result[i][5] = String.valueOf(1 - eval.errorRate());
	            result[i][6] = String.valueOf(gmean);
	            result[i][7] = String.valueOf(eval.recall(0));
	            result[i][8] = String.valueOf(eval.recall(1));
	            result[i][9] = String.valueOf(eval.precision(0));
	            result[i][10] = String.valueOf(eval.precision(1));
	            result[i][11] = String.valueOf(eval.fMeasure(0));
	            result[i][12] = String.valueOf(eval.fMeasure(1));
	            result[i][13] = String.valueOf(0);
	            result[i][14] = String.valueOf(0);
	            result[i][15] = String.valueOf(eval.areaUnderROC(1));
	            result[i][16] = String.valueOf(m_anomaly);
		    			            			          	            
	     }
		FileUtil.outFile(out_path, result, Config.head);
	    System.out.println("-------> bestThres : " + bestThres);
		return bestThres;
	}
	
  
}
