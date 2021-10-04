package Util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import jxl.Workbook;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import jxl.write.WriteException;

public class FileUtil {
	
	public static void exportObjFile(ArrayList<double[]> array,String file,String[] head) throws Exception {  
		Object[] objects = array.toArray();
		WritableWorkbook wbook = Workbook.createWorkbook(new File(file)); //建立excel文件  
		WritableSheet wsheet = wbook.createSheet("result", 0); //工作表名称  
		jxl.write.Label content = null;
		for(int j = 0;j<head.length; j++){
			if(j<head.length){
			content = new jxl.write.Label(j, 0, head[j]);
			}else{
				content = new jxl.write.Label(j, 0, "");
			}
			
			wsheet.addCell(content);
		}
		double[] temp =  new double[5];
		for(int i = 0; i<objects.length; i++)
		{//行
			temp = (double[])objects[i];
			for(int j = 0; j<temp.length; j++){
				//System.out.println(j);
				content = new jxl.write.Label(j, i+1, String.valueOf(temp[j])); 
				//content = new jxl.write.Label(j, i+1, objects[i].toString()); 
				wsheet.addCell(content);
			}
		} /**/
	try { 
		wbook.write(); //写入文件  
		wbook.close();  
	} catch (Exception e) {  
		throw new Exception("导出文件出错");  
	}  
}
	//结果二维数组写入xls文件
	public static void outFile(String path, String[][] res, String[] head) throws IOException, WriteException, WriteException{

		WritableWorkbook wbook = Workbook.createWorkbook(new File(path)); //建立excel文件  
		WritableSheet wsheet = wbook.createSheet("result", 0); //工作表名称  
		//String[] temp = {"Evaluation","Search","AttrNum","accuracy","sensitive","precision","time"};
		jxl.write.Label content = null;

		for(int j = 0; j < head.length; j++){
			content = new jxl.write.Label(j, 0, head[j]);
			wsheet.addCell(content);
		}
		for(int i = 0; i < res.length; i++)
		{//行
			
			for(int j = 0; j < res[0].length; j++){
				content = new jxl.write.Label(j, i+1, res[i][j]);
				wsheet.addCell(content);
			}
		}
		
		wbook.write(); //写入文件  
		wbook.close(); 		

	}
	
	public static void exportFile(String[][] res,String file,String project, int sh, String[] head) throws Exception {  
		try { 
			File resultFile = new File(file);
			WritableWorkbook wbook = null;
			WritableSheet wsheet = null;
			if(!resultFile.exists()){
				wbook = Workbook.createWorkbook(resultFile); //建立excel文件  
				wsheet = wbook.createSheet("result",0); //工作表名称
			}else{
			
				Workbook wb = Workbook.getWorkbook(resultFile);
				wbook = Workbook.createWorkbook(resultFile,wb); //建立excel文件  
				wsheet = wbook.getSheet(0); //工作表名称  
			}
			
			//WritableSheet wsheet = wbook.createSheet("result",0); //工作表名称  

			
			int row = wsheet.getRows();
			System.out.println("======Rows:"+row+"======");
			
			jxl.write.Label content = null;
			if(row<=0){
				/*for(int j = 0;j<res[0].length; j++){
						if(j<head.length){
					content = new jxl.write.Label(j, 0, head[j]);
					}else{
						content = new jxl.write.Label(j, 0, "");
					}
					wsheet.addCell(content);
				}*/
				for(int j = 0; j < head.length; j++){
					content = new jxl.write.Label(j, 0, head[j]);
					wsheet.addCell(content);
				}
			}
			//向第一行写入项目路径
			row = wsheet.getRows();
			//content = new jxl.write.Label(0, row, project); 
			//wsheet.addCell(content);
			
			for(int i = 0; i<res.length; i++)
			{//行
				for(int j = 0;j<res[i].length; j++)
				{//列
					content = new jxl.write.Label(j, row+i, String.valueOf(res[i][j])); 
					wsheet.addCell(content);
				}
			} 

			wbook.write(); //写入文件  
			wbook.close();  
		} catch (Exception e) {  
			System.out.println(e.toString());
			throw new Exception();  
		}  
		
	}
	
	public static Instances ReadData(String filePath){
	    System.out.println("======Read data from " + filePath + "======");
		if(!new File(filePath).exists()){
			System.out.println("======This File doesn't exist!" + "======");
			return null;
		}
		Instances ResultIns = null;
		//不做属性选择
		File fData = new File(filePath);	
		ArffLoader loader = new ArffLoader();
		try {
			loader.setFile(fData);
			ResultIns = loader.getDataSet();
			System.out.println("=========Data Information=========");
		    System.out.println("======AttrNum:"+ResultIns.numAttributes()+"======");
		    System.out.println("======InstancesNum:"+ResultIns.numInstances()+"======");
		} catch (IOException e) {
				// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		return ResultIns;
}
	
	public static Instances ReadDataCSV(String filePath){
	    System.out.println("======Read data from " + filePath + "======");
		if(!new File(filePath).exists()){
			System.out.println("======This File doesn't exist!" + "======");
			return null;
		}
		Instances ResultIns = null;
		//不做属性选择
		File fData = new File(filePath);	
		CSVLoader loader = new CSVLoader();
		try {
			loader.setSource(fData);
			ResultIns = loader.getDataSet();
			System.out.println("=========Data Information=========");
		    System.out.println("======AttrNum:"+ResultIns.numAttributes()+"======");
		    System.out.println("======InstancesNum:"+ResultIns.numInstances()+"======");
		} catch (IOException e) {
				// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		return ResultIns;
}
	public static boolean WriteData(Instances ins,String filePath){
	    
		if(new File(filePath).exists()){
			System.out.println("======" + filePath + "already exist!======");
			return false;
		}
		ArffSaver saver = new ArffSaver(); 
	    saver.setInstances(ins);  
	    try {
			saver.setFile(new File(filePath));
			saver.writeBatch(); 
		    System.out.println("==Arff File Writed:"+filePath+"==");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}  
	    
	    return true;

		
}
public static boolean WriteArff2CSV(Instances ins,String filePath){
	    
		if(new File(filePath).exists()){
			System.out.println("======" + filePath + "already exist!======");
			return false;
		}
		weka.core.converters.CSVSaver saver = new CSVSaver();
	    saver.setInstances(ins);  
	    try {
			saver.setFile(new File(filePath));
			saver.writeBatch(); 
		    System.out.println("==CSV File Writed:"+filePath+"==");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}  
	    
	    return true;

		
}
	public static String getFileName(String filePath){
		return filePath.substring(filePath.lastIndexOf("\\")+1, filePath.lastIndexOf("."));//获取文件名
	}
	
	public static int anomalyIndex(Instances ins){
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

	

}
