package Util;

public class Config{
	
	public static String total_folder = "G://dataset//change6.0//";//"E://dataset//binbin//";//
	public static String result_folder = total_folder + "compare_result_logis_mcc//";
	//private static String data_folder = "E://dadaset//nasa-data//data//";
	//public static String data_folder = "E://dadaset//change//arff//";
	public static String data_folder = total_folder + "com_net_bow_arff//";//"com_net_arff_selected//CfsSu_BestF//"
	public static String select_folder = total_folder + "selected/";
	public static String sample_folder = total_folder + "sample/";
	//public static String data_folder = "E://dataset//binbin//binbin//";
	
	private static String[] EvalS = {"ChiSquaredAttributeEval"};
	//,"ReliefFAttributeEval","OneRAttributeEval","GainRatioAttributeEval"
	//,"CostSensitiveAttributeEval"
	private static String[] RankS = {"Ranker"};
	public static String[] SubEvalS = {"CfsSubsetEval"};//,"FilteredSubsetEval","WrapperSubsetEval","ClassifierSubsetEval","ConsistencySubsetEval","WrapperSubsetEval","CostSentitiveSubsetEval"
	//"ClassifierSubsetEval","ConsistencySubsetEval","CostSentitiveSubsetEval"}
	//private static String[] SearchS = {"BestFirst","ExhaustiveSearch","FCBFSearch","GeneticSearch","GreedyStepwise","LinearForwardSelection","RandomSearch",
    //		"RankSearch","ScatterSearchV1","SubsetSizeForwardSelection","TabuSearch"};
	public static String[] SearchS = {"BestFirst"};//"LinearForwardSelection","GreedyStepwise","SubsetSizeForwardSelection","GeneticSearch","RaceSearch"
	public static String[] head = {"file","AttrNum","SamSize","NumTrees","Threshold","accuracy","gmean","recall-0","recall-1","precision-0","precision-1","fMeasure-0","fMeasure-1","balance_0","balance_1","AUC","AnomalyClas","time","Evaluation","Search"};
	public static String file="";

}