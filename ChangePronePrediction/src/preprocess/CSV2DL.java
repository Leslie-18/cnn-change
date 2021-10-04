package preprocess;

import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;

import org.ujmp.core.objectmatrix.calculation.Convert;

import classify.base.NewIsolationForest;

public class CSV2DL {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String data_folder = "E:\\dataset\\change2.0\\dependency_graph\\";
    	String result_folder = "E:\\dataset\\change2.0\\dependency_graph\\";
    	File f = new File(result_folder);
    	if(!f.exists()) f.mkdirs();
    	
    	File ff = new File(data_folder);//select_folder
    	File[] files = ff.listFiles();
    	for(int i = 0;i < files.length; i++){
    		String fileName = files[i].getName();
    		System.out.println(fileName);
    		if(fileName.substring(fileName.lastIndexOf(".") + 1).equals("csv"))
    			convert(data_folder + fileName, result_folder  + fileName.substring(0,fileName.lastIndexOf(".")) + ".txt" );
    	}
	}
	
	private static void convert(String inPath, String outPath) throws IOException{
		BufferedReader bReader = new BufferedReader(new FileReader(inPath)); 
		BufferedWriter bWriter = new BufferedWriter(new FileWriter(outPath));
		ArrayList<String> nodesList = new ArrayList<String>(); 
		String head = "";
		StringBuffer sBuffer = new StringBuffer();
		try {
			int cnt = 0;
			String line = bReader.readLine();
			while ((line = bReader.readLine()) != null) {
				String[] temp = line.split(",");
				sBuffer.append(temp[0] + "    " + temp[1] + "\n");
				cnt++;
				if(!nodesList.contains(temp[0])){
					nodesList.add(temp[0]);
				}
				if(!nodesList.contains(temp[1])){
					nodesList.add(temp[1]);
				}
			}
			head = "DL n=" + nodesList.size() + "\n" + "format = nodelist1" + "\n"
					+ "labels embedded:" + "\n"
					+ "data:" + "\n";
			bWriter.write(head);
			bWriter.write(sBuffer.toString());
			bWriter.flush();
			bWriter.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
