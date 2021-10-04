package preprocess;

import java.io.File;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
/*
 * 该函数实现将原始arff文件中的名词性属性转换成数值型属性
 * */
public class NomToNum{
	private static String data_folder = "E:/dadaset/new data/temp0/";
	private static String result_path = "E:/dadaset/new data/output0/";
	//private static String data_folder = "E:/dadaset/data3/data - 副本/";
	//private static String result_path = "E:/dadaset/data3/data6/";
	public static void main(String[] args) throws Exception {
		File f = new File(result_path);
		if(!f.exists()) f.mkdirs();
		//trianAndEval(select_folder+"antSelectedFilte_Linea.arff","C:/Users/Administrator/Workspaces/weka-test/result/Various.xls");
		File ff = new File(data_folder);
		String[] files = ff.list();
		//String[] files = {"liferay1.arff"};
		String data = null,out_path = null;
		
		for(int i = 0;i < files.length; i++){
			//定义输入文件路径和输出文件路径
			data = data_folder + files[i];	
			File fData = new File(data);
			out_path = result_path + files[i].substring(0, (files[i].length()-4))+".arff";
			System.out.println("==Input Data:"+data+"==");
			
			if(!(new File(out_path).exists())){
				
				System.out.println("==Output Path:"+out_path+"==");
				//读取原始csv文件
				CSVLoader loader = new CSVLoader();
				loader.setSource(new File(data));
				Instances inputIns = loader.getDataSet();
				//输出原始csv文件的信息
				System.out.println("==Num Attr:"+inputIns.numAttributes());
				inputIns.deleteAttributeAt(0);
				inputIns.deleteAttributeAt(0);
				System.out.println("==Num Attr:"+inputIns.numAttributes());

				//File f2 = new File(data);
 				//ArffLoader loader0 = new ArffLoader();
 				//loader0.setFile(f2);
 				//Instances newIns = loader0.getDataSet();
				
				//调用转换接口，将名词性属性转化成数值型属性
				Instances tempIns = inputIns;
				Instances outputIns = convertAttr(tempIns,"is_bug_intro");
				
				//移除99%相同的冗余属性
				weka.filters.unsupervised.attribute.RemoveUseless remove = new weka.filters.unsupervised.attribute.RemoveUseless();
				remove.setMaximumVariancePercentageAllowed(0.99);
				//Instances tempIns = remove.useFilter(inputIns, remove);
				//输出取出冗余属性之后的属性个数
				
				System.out.println("==Removed Attr:"+tempIns.numAttributes());
				remove.setInputFormat(outputIns);
				outputIns = remove.useFilter(outputIns, remove);
				//输出结果，并保存至arff文件
				System.out.println("==Num Attr:"+outputIns.numAttributes());
				ArffSaver saver = new ArffSaver(); 
				saver.setInstances(outputIns);  
			    saver.setFile(new File(out_path));  
			    saver.writeBatch(); 
			    System.out.println("==Selected Attrbutes Writed:"+out_path+"==");
			}else{
				System.out.println("=="+out_path+"already exists!==");
			}
		}
	}
	
	public static Instances convertAttr(Instances ins, String c) throws Exception{
		Instances res = new Instances(ins);
		Instances insCopy = new Instances(ins);
		//res.deleteAttributeAt(0);
		res.delete();
		FastVector nomAttr =  new FastVector();
		FastVector nomInd =  new FastVector();
		int numAttr = ins.numAttributes();
		int numIns = ins.numInstances();
		int numNominal = 0;
		FastVector attInfo = new FastVector();
		Attribute label = ins.attribute(c);

		//record the nominal attributes
		for(int i = 0 ;i< numAttr; i++){
			if(ins.attribute(i).isNominal() ){//&& !ins.attribute(i).equals(label)
				Attribute temp = ins.attribute(i);
				if(temp.name().trim().equals(c)){//
					System.out.println("class attr:" + temp.name());
					label = temp;
					continue;
				}
				numNominal++;
				System.out.println("nominal attr:" + temp.name());
				nomAttr.addElement(temp);
				nomInd.addElement(i);
			}
		}
		System.out.println(numNominal+ "==="+numIns);
		Attribute newLabel = label.copy(c+"Copy");
		
		double[] cnt = new double[numNominal];
		Map[] map= new HashMap[numNominal];
		Attribute[] newAttr = new Attribute[numNominal];
		int[] newInd = new int[numNominal];
		for(int i = 0 ; i< numNominal;i++){
			map[i] = new HashMap<Double,Double>(); 
			newAttr[i] = new Attribute(((Attribute) nomAttr.elementAt(i)).name()+"Copy");
			ins.insertAttributeAt(newAttr[i], ins.numAttributes());
			System.out.println(newAttr[i].name());
			newInd[i] = ins.numAttributes()-1;
		}
		//Attribute att = new Attribute("label");		
		ins.insertAttributeAt(newLabel, ins.numAttributes());
		System.out.println("attr num" + res.numAttributes() + "   " + res.numInstances());
		numAttr = ins.numAttributes();
		//int time_daysInd = ins.attribute("time_days").index();
		for(int j = 0 ; j < numIns; j++){
			Instance temp = ins.instance(j);
			String  clas = temp.stringValue(label);
			//System.out.print(j+":"+clas);
			for(int i = 0 ; i< numNominal; i++){
				Attribute ind = (Attribute) nomAttr.elementAt(i);
				double value = temp.value(ind);
				//String value = temp.stringValue(ind);
				
				if(map[i].isEmpty() || !map[i].containsKey(value)){
					
					map[i].put(value, cnt[i]);
					cnt[i]++;
				}	
				//System.out.print(ind+"==="+value+"==="+(Double)map[i].get(value));
				ins.instance(j).setValue(newInd[i], (Double)map[i].get(value));//(ind, 1000);
				//ins.instance(j).setValue((Integer)nomInd.elementAt(i), ((Attribute) nomAttr.elementAt(i)).indexOfValue(value));//(ind, 1000);
				//ins.instance(j).setValue((Integer)nomInd.elementAt(i), (Double)map[i].get(value));
				
			}
			//System.out.println("class:"+ins.attribute(numAttr-1).name());
			ins.instance(j).setValue(numAttr-1, clas);
			//ins.instance(j).setValue(time_daysInd, temp.value(time_daysInd)+0.02);
			res.add(temp);
		}

		
		for(int i = 0 ; i< ins.numAttributes()-1; i++){
			if(ins.attribute(i).isNominal()){
				ins.deleteAttributeAt(i);
				i--;
			}
			
		}
		//ins.deleteAttributeAt(5);
		//weka.filters.unsupervised.attribute.NumericToNominal filter = new weka.filters.unsupervised.attribute.NumericToNominal();
		//filter.setInputFormat(ins);
		//filter.setInvertSelection(false);
		//String[] options = {"-R","last"};
		//filter.setOptions(options);
		//filter.setAttributeIndices("last");//setAttributeIndices("last");
		//Instances outputIns = filter.useFilter(ins, filter);
		System.out.println("============="+res.numAttributes()+"==="+res.numInstances()+"====================");
		return ins;
	}
	
	public static Instances binaryAttr(Instances ins) throws Exception{
		Instances res = new Instances(ins);
		//res.deleteAttributeAt(0);
		res.delete();
		FastVector nomAttr =  new FastVector();
		int numAttr = ins.numAttributes();
		int numIns = ins.numInstances();
		int numNominal = 0;
		String rangelist = "";
		//record the nominal attributes
		for(int i = 1 ;i< numAttr; i++){
			if(ins.attribute(i).isNominal()){
				numNominal++;
				nomAttr.addElement(i);
				rangelist += String.valueOf(i)+",";
			}
		}
		rangelist = rangelist.substring(0,rangelist.length()-2);


		//System.out.println(numNominal+ "==="+numIns);
		//Ajust Attributes order
		FastVector attInfo = new FastVector();
		Attribute label = ins.attribute(0);
		for(int i = 1;i<numAttr;i++)
		{
			weka.core.Attribute temp = ins.attribute(i);
			attInfo.addElement(temp);
		}
		attInfo.addElement(label);
		res = new Instances("res",attInfo,numIns);
		System.out.println("attr num" + res.numAttributes() + "   " + res.numInstances());

		weka.filters.unsupervised.attribute.NominalToBinary filter = new weka.filters.unsupervised.attribute.NominalToBinary();
		filter.setInputFormat(ins);
		filter.setAttributeIndices(rangelist);
		Instances outputIns = filter.useFilter(ins, filter);
		return outputIns;
		
		
	}

}