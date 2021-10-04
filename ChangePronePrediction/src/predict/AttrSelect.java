package predict;

import java.io.File;
import Util.Config;
import Util.FileUtil;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

/**
 * @author Heyueyang
 *
 */
public class AttrSelect {
	static Instances ins;
	static String Eval;
	static String Search;
	private static String data_folder = "E:/dataset/change2.0/com_net_other_bow_arff/";
	private static String select_folder = data_folder.substring(0,data_folder.lastIndexOf("/")) + "_selected/";//"E:/dataset/change2.0/com_net_arff_selected/";
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String[] Eval = Config.SubEvalS;
		String[] Sear = Config.SearchS;
    	File ff = new File(data_folder);//select_folder
    	File[] files = ff.listFiles();

	    	for(int s = 0; s < Eval.length; s++)
	    	{
	    		for(int p = 0; p < Sear.length; p++)
	    		{
	    			String eval = Eval[s];
	    			String search = Sear[p];
			    	String outDir = select_folder + eval.substring(0, 5) + "_"+search.substring(0, 5) + "/";
			    	if(!new File(outDir).exists()){
			    		new File(outDir).mkdirs();
			    	}
			    	for(int i = 0;i < files.length; i++){
			    		String file_path = files[i].getAbsolutePath();
			    		System.out.println("=========File:" + file_path + "=======");
			    		String file_name = file_path.substring(file_path.lastIndexOf("\\"), file_path.lastIndexOf("."));
				    	String outArffPath = outDir + file_name + ".arff";
				    	if(new File(outArffPath).exists()){
				    		continue;
				    	} 
				    	
				    	Instances newIns = FileUtil.ReadData(file_path);
					    //set the classIndex
					    System.out.println("======Attribute Selecting...======");
					    Instances ResultIns = null;
					    try{ 
					    	ResultIns = new AttrSelect(newIns, eval, search).SubsetSelect();
					    }catch(Exception e){
					    	e.printStackTrace();
					    	continue;
					    }
					    System.out.println("======New AttrNum="+(ResultIns.numAttributes())+"======");
				    	/*write to arff*/
				    	if(FileUtil.WriteData(ResultIns, outArffPath)){
				    		System.out.println("==Selected Attrbutes Writed:"+outArffPath+"==");
				    	}else{
				    		continue;
				    	}
			    	}
				    
	    		}
	    	}
    	
    	
    	
	}
	
	public AttrSelect(Instances ins, String eval, String search) {
		super();
		this.ins = ins;
		Eval = eval;
		Search = search;
	}

	public Instances RankSelect(int num) throws Exception{
        /*
         * 初始化搜索算法（search method）及属性评测算法（attribute evaluator）
         */
		/*
		 * 方法一：单一属性评估器+排序方法
		 * */
		Instances tempIns = ins;
			try{
				Class<?> c1 = Class.forName("weka.attributeSelection." + Eval);
				Class<?> c2 = Class.forName("weka.attributeSelection." + Search);	
				ASEvaluation eval = (ASEvaluation) c1.newInstance();
				Ranker rank = (Ranker) c2.newInstance();
				rank.setNumToSelect(num);
				
				AttributeSelection attrs1 = new AttributeSelection();
				
		        /*
		        * 3.根据评测算法评测各个属性
		        */
		
		        attrs1.setEvaluator(eval);
				attrs1.setSearch(rank);
				System.out.println("1");
				attrs1.SelectAttributes(ins);
				System.out.println("2");
				tempIns = attrs1.reduceDimensionality(ins);
				System.out.println("Evaluation:" + c1.getName());
				System.out.println("Search:" + c2.getName());
			}catch(Exception e){
				System.out.println("Attribute Selection Error::"+e.toString());
				throw new Exception();
			}
			
		
		return tempIns;
        
	}
	
	public Instances SubsetSelect() throws Exception{

		Instances tempIns = ins;
	
			/*
			 * 方法二：属性子集评估器+搜索方法
			 * */
			try{
				Class<?> c1 = Class.forName("weka.attributeSelection." + Eval);
				Class<?> c2 = Class.forName("weka.attributeSelection." + Search);	
				//WrapperSubsetEval wra = WrapperSubsetEval.class;
				//Ranker rank = (Ranker) c2.newInstance();
				//rank.setNumToSelect(num);

				ASEvaluation eval = (ASEvaluation) c1.newInstance();
				ASSearch search = (ASSearch) c2.newInstance();
				AttributeSelection attrs = new AttributeSelection();
				//if(Search == "LinearForwardSelection"){
				//	search = (weka.attributeSelection.LinearForwardSelection) c2.newInstance();
				//	((weka.attributeSelection.LinearForwardSelection)search).setNumUsedAttributes(1000);
				//}
				attrs.setEvaluator(eval);
				attrs.setSearch(search);
				attrs.SelectAttributes(ins);
				tempIns = attrs.reduceDimensionality(ins);
			}catch(Exception e){
				System.out.println("Attribute Selection Error::"+e.getMessage());
				throw new Exception();
			}
			
			

		
		return tempIns;
	}   



}
