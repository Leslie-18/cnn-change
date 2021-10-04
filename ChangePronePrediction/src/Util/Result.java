package Util;

import java.io.File;
import java.io.IOException;

import jxl.Cell;
import jxl.Workbook;
import jxl.write.WritableCell;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import jxl.write.WriteException;
import jxl.write.biff.RowsExceededException;

public class Result{
	private static double time = 0.0;
	private static int attrNum = 0;
	private static String eval = null;
	private static String search = null;
	private static String path = null;
	private static int anomaly = 0;
	private static double gmean = 0;
	private static  double accuracy = 0;
	private static double[] recall = new double[2];
	private static double[] precision = new double[2];
	private static double[] fmeasure = new double[2];
	private static double[] balance = new double[2];
	private static double[] auc = new double[2];
	private static double mcc = 0;
	
	private static double matrix[][] = null;
	private static WritableWorkbook wbook = null; //建立excel文件  
	private static WritableSheet wsheet = null; //工作表名称  
	
	public Result(){
		
	}
	public Result(double t,int num, String e, String s, int an,double gm,double acc, double[] sen, double[] spe,double[] fm, double[] bal, double[] au,double[][] m,double c){
		time = t;
		attrNum = num;
		eval = e;
		search = s;
		anomaly = an;
		gmean = gm;
		accuracy = acc;
		recall =sen;
		precision = spe;
		fmeasure = fm;
		balance = bal;
		matrix = m;
		auc = au;
		mcc = c;
	}
	public static void end() throws IOException, WriteException{
		wbook.write(); //写入文件  
		wbook.close(); 
	}
	public static void setPath(String p){
		path = p;
	}
	public static void setGmean(double gm){
		gmean = gm;
	}
	public static void setAnomalyClass(int an){
		anomaly = an;
	}
	public static void setAccuracy(double acc){
		accuracy = acc;
	}
	public static void setRecall(double[] sen){
		recall = sen;
	}
	public static void setFmeasure(double[] fm){
		fmeasure = fm;
	}
	public static void setBalance(double[] bal){
		balance = bal;
	}
	public static void setPrecision(double[] spe){
		precision = spe;
	}
	public static void setAttrNum(int num){
		attrNum = num;
	}
	public static void setTime(double t){
		time = t;
	}
	public static void setMcc(double c) {
		mcc = c;
	}
	public double getAnomalyClass(){
		return  anomaly;
	}
	public static double getGmean(){
		return  gmean;
	}
	public static double getAccuracy(){
		return  accuracy;
	}
	public static double[] getRecall(){
		return recall;
	}
	public static double[] getFmeasure(){
		return fmeasure;
	}
	public static double[] getPrecision(){
		return precision;
	}
	public static double[] getBalance(){
		return balance;
	}
	public static double getTime(){
		return time;
	}
	public static double getMcc(){
		return mcc;
	}
	public static int getAttrNum(){
		return attrNum;
	}
	public static double recall(int ind){
		return recall[ind];
	}
	public static double precision(int ind){
		return precision[ind];
	}
	public static double fMeasure(int ind){
		return fmeasure[ind];
	}
	public static double errorRate(){
		return 1-accuracy;
	}
	public static double areaUnderROC(int ind){
		return auc[ind];
	}
}