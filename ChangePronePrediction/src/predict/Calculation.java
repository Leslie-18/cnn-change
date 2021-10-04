package predict;

import java.awt.BorderLayout;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class Calculation {
	 private static Object[] sortArray(Object[] a){
		 double[] temp1 = new double[3];
		 double[] temp2 = new double[3];
		 Object temp = new Object();
		 for(int i = 0; i < a.length; i++){
			 for(int j = i; j < a.length; j++){
				 temp1 = (double[])a[i];
				 temp2 = (double[])a[j];
				 if(temp1[0] > temp2[0]){
					 temp = a[i];
					 a[i] = a[j];
					 a[j] = temp;
				 }
			 }
		 }
		 return a;
	 }
	 
	 static String[] calAverage(String[][] in){
		 int len = in.length, wid = in[0].length;
		int k = 4;
		String[] res = new String[wid];
		double[] cal = new double[wid - k];

		for(int i = 0 ;i < k; i++){
			res[i] = in[0][i];
		}
		for(int i = 0 ;i < len; i++){
			for(int j = k ; j < wid; j++){
				cal[j-k] += Double.parseDouble(in[i][j]);
			}
		}
		for(int j = k ; j < wid; j++){
			res[j] = String.valueOf(cal[j-k]/(double)len);
		}
		return res;
		 
	 }

	static double runROC(Instances Ins, Classifier c1) throws Exception{
		//isoF.buildClassifier(Ins);
		//Classifier c1 = isoF;
		
		//Classifier c1 = new RandomForest();
		Evaluation e1 = new Evaluation(Ins);
        e1.crossValidateModel(c1, Ins, 10, new Random(1));
		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;//Ins.numAttributes()-1;
		System.out.println("classIndex: " + classIndex);


		//System.out.println("e1.predictions(): " + e1.predictions().toArray()[2]);
		//String out = result_path+"temp/randomforest"+".xls";
    	//String[] head0 = {"score","label","prediction"} ;
    	//exportObjFile(e1.predictions().toArray(),out,head0);
		Instances curve = tc.getCurve(e1.predictions(), classIndex);
		System.out.println("curve: " + curve.classIndex());
		System.out.println("curve: " + curve.numInstances());
		double auc = e1.areaUnderROC(classIndex);
		System.out.println("The area under the ROC　curve: " + auc);
		
		/*
		* 在这里我们通过结果信息Instances对象得到包含TP、FP的两个数组
		* 这两个数组用于在SPSS中通过线图绘制ROC曲面
		*/
		int tpIndex = curve.attribute(ThresholdCurve.TP_RATE_NAME).index();
		int fpIndex = curve.attribute(ThresholdCurve.FP_RATE_NAME).index();
		double [] tpRate = curve.attributeToDoubleArray(tpIndex);
		double [] fpRate = curve.attributeToDoubleArray(fpIndex);
		//Util.writeArray(tpRate, fpRate, "d:\roc.txt");
		/*
		* 4.使用结果信息instances对象来显示ROC曲面
		*/
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		//这个获得AUC的方式与上面的不同，其实得到的都是一个共同的结果
		vmc.setROCString("(Area under ROC = " +
		Utils.doubleToString(tc.getROCArea(curve), 4) + ")");
		vmc.setName(curve.relationName());
		PlotData2D tempd = new PlotData2D(curve);
		tempd.setPlotName(curve.relationName());
		tempd.addInstanceNumberAttribute();
		vmc.addPlot(tempd);
		// 显示曲面
		String plotName = vmc.getName();
		final javax.swing.JFrame jf =
		new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
		jf.setSize(500,400);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(vmc, BorderLayout.CENTER);
		jf.addWindowListener(new java.awt.event.WindowAdapter() {
		public void windowClosing(java.awt.event.WindowEvent e) {
		jf.dispose();
		}
		});
		jf.setVisible(true);
		
		return auc;
	}
	
//public static double calAUC(Object[] roc, double d) throws Exception{
//		
//		double tp = 0, fp = 0, f = 0, p = 0;
//		double thres = 0.0, accPos = 0.0,accNeg = 0.0;
//		double[][] temp = new double[roc.length][3];
//		double[][] res = new double[roc.length][3];
//		double t_t = 0, t_f = 0, f_t = 0, f_f = 0;
//		double auc1 = 0.0;
//		double sum = 0, cnt = 0;
//		double[] rank = new double[roc.length];
//		//Arrays.sort(roc);
//		roc = sortArray(roc);
//		//System.out.println(roc.length);
//		rank[roc.length-1] = roc.length;
//		sum = rank[roc.length-1];
//		
//		for(int i = 0;i < roc.length; i++){
//			temp[i] = (double[])roc[i];
//			if(temp[i][1] == d){
//				p++;
//			}else{
//				f++;
//			}
//		}
//		
//		for(int i = roc.length-2;i > 0; i--){
//			temp[i] = (double[])roc[i];
//			rank[i] = i+1;
//			if(temp[i][0] == temp[i+1][0]){
//				sum += rank[i];
//				cnt++;
//			}
//			else
//			{
//				if(cnt > 0)
//				{
//					for(int j = i+1; j <= i+cnt; j++)
//					{						
//						rank[j] = sum/(cnt+1);
//					}
//					cnt = 0;
//					sum = i+1;
//				}
//			}
//			
//		}
//		
//		int cal = 0;
//		sum = 0;
//		for(int i = temp.length-1;i >= 0; i--){
//			if(temp[i][1] == d){
//				//accPos++;
//				//System.out.println(rank[i]);
//				sum += rank[i];
//			}
//		}
//		//System.out.println("p="+ p + ",f="+ f + ",sum = "+ sum);
//		auc1 = (sum - (p*(p+1))/2)/(p*f);
//		//auc0 = (sum - (f*(f-1))/2)/(p*f);
//
//		return auc1;
//
//	}

public static double calAUC(Object[] roc, double d) throws Exception{
	
	double tp = 0, fp = 0, f = 0, p = 0;
	double thres = 0.0, accPos = 0.0,accNeg = 0.0;
	double[][] temp = new double[roc.length][3];
	double[][] res = new double[roc.length][3];
	double t_t = 0, t_f = 0, f_t = 0, f_f = 0;
	double auc1 = 0.0;
	double sum = 0, cnt = 0;
	double[] rank = new double[roc.length];
	//Arrays.sort(roc);
	roc = sortArray(roc);
	//System.out.println(roc.length);
	rank[roc.length-1] = roc.length;
	//sum = rank[roc.length-1];
	
	for(int i = 0;i < roc.length; i++){
		temp[i] = (double[])roc[i];
		if(temp[i][1] == d){
			p++;
		}else{
			f++;
		}
	}
	
	for(int i = roc.length-2;i >= 0; i--){  //jia = 
		temp[i] = (double[])roc[i];
		rank[i] = i+1;
		if(temp[i][0] == temp[i+1][0]){
			//sum += rank[i];
			cnt++;
			if(i==0) {
				for(int m=i;m<i+cnt+1;m++) {
					sum+=rank[m];
				}
				for(int j = i;j<=i+cnt;j++)
				{
					rank[j] = sum /(cnt+1);
				}
			}
			
		}
		else
		{
			if(cnt > 0)
			{
				for(int m = i+1;m<i+cnt+2;m++) {
					sum += rank[m];
				}
				for(int j = i+1; j <= i+cnt+1; j++)
				{						
					rank[j] = sum/(cnt+1);
				}
				cnt = 0;
				sum = 0;
			}
		}
		
	}
	

	sum = 0;
	for(int i = temp.length-1;i >= 0; i--){
		if(temp[i][1] == d){
			//accPos++;
			//System.out.println(rank[i]);
			sum += rank[i];
		}
	}
	//System.out.println("p="+ p + ",f="+ f + ",sum = "+ sum);
	auc1 = (sum - (p*(p+1))/2)/(p*f);
	//auc0 = (sum - (f*(f-1))/2)/(p*f);

	return auc1;

}
}


