package preprocess;
import Util.Config;
//采样 过采样 欠采样
import java.io.File;
import java.io.IOException;
import java.text.AttributedCharacterIterator.Attribute;
import java.util.Random;

import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class Sample{
	 private static String className = "bug_introducingCopy";//"change_prone";
	 public Sample(String claName){
		 className = claName;
	 }
	 public Sample(){
	 }
	 
	 public static Instances AntiUnderSample(Instances init, double samRatio) throws Exception{
		 double ratio = samRatio;
		 int numAttr = init.numAttributes();
			int numInstance = init.numInstances();

			FastVector attInfo = new FastVector();
			for (int i = 0; i < numAttr; i++) {
				weka.core.Attribute temp = init.attribute(i);
				attInfo.addElement(temp);
			}

			Instances NoInstances = new Instances("No", attInfo, numInstance);

			NoInstances.setClass(NoInstances.attribute(className));

			Instances YesInstances = new Instances("yes", attInfo, numInstance);
			YesInstances.setClass(YesInstances.attribute(className));

			init.setClass(init.attribute(className));
			int classIndex = init.classIndex();
			
			int numYes = 0;
			int numNo = 0;
			
			for (int i = 0; i < numInstance; i++) {
				Instance temp = init.instance(i);
				double Value = temp.value(classIndex);
				if (Value == 0) { // yes
					NoInstances.add(temp);
					numNo++;
				} else {
					YesInstances.add(temp);
					numYes++;
				}
			}
			
			Instances res;
			if (numYes > numNo) {
				if ((double)(numNo/numYes) <= ratio) {
					return init;
				}
				res = excuteSample(YesInstances, NoInstances, ratio);
				
			} else {
				if ((double)(numYes/numNo) <= ratio) {
					return init;
				}
				res = excuteSample(NoInstances, YesInstances, ratio);
			}
			return res;

		 /*
			Instances res = new Instances(ins);
			res.delete();
			int valueCnt[] = {0,0};
			int attNum = ins.numAttributes();
			int insNum = ins.numInstances();
			int[] label = new int[insNum];
			ins.setClassIndex(attNum-1);
			int temp = 0;
			for(int i =0; i < insNum; i++){		
				temp = (int) ins.instance(i).classValue();
				valueCnt[temp]++;
				label[i] = temp;
			}
			int anomCnt = (valueCnt[0] > valueCnt[1]) ? valueCnt[1] : valueCnt[0];
	    	int anomLab = (valueCnt[0] > valueCnt[1]) ? 1 : 0;
	    	
	    	java.util.Random r=new java.util.Random();
	    	int[] posInd = new int[anomCnt];
	    	int[] negInd = new int[insNum - anomCnt];
	    	int t = 0, p = 0;
	    	int samNum = 0;
	    	for(int i =0; i < insNum; i++){		
				temp = (int) ins.instance(i).classValue();
				if(label[i] == anomLab){
					posInd[t++] = i;
					
				}else{
					negInd[p++] = i;
				}	
			}
	    	//sampling in positive index
	    	if(((double)anomCnt/(double)insNum) <= samRatio){
	    		System.out.println(" (anomCnt"+anomCnt+")/insNum("+insNum+") "+((double)anomCnt/(double)insNum)+"<= samRatio "+samRatio);
	    		res = ins;
	    	}else{
		    	samNum = (int) ((insNum-anomCnt)*samRatio);
		    	if(samNum <= anomCnt){
			    	for(int i =0; i < samNum; i++){
			    		//System.out.println(posInd[r.nextInt(anomCnt)]);
			    		res.add(ins.instance(posInd[r.nextInt(anomCnt)]));
			    	}
			    	for(int i =0; i < negInd.length; i++){
			    		res.add(ins.instance(negInd[i]));
			    	} 
			    	
		    	}else{
		    		throw new Exception("samNum "+samNum+" must be smaller than number of positive instances "+anomCnt);
		    	}
		    	
	    	}
	    	String underPath = Config.select_folder + this.file.substring( 0,  file.lastIndexOf(".")) + "_undersample" + ".arff";
            System.out.println("===UnderSample Path==="+underPath);
            System.out.println("===minority num("+anomLab+"):"+anomCnt+" ===");
            System.out.println("===new minority num("+anomLab+"):"+samNum+" ===");
            ArffSaver saver = new ArffSaver(); 
    		saver.setInstances(res);  
    	    saver.setFile(new File(underPath));  
    	    saver.writeBatch(); 
	    	return res;*/
	    	
		 }
	 
	 public static Instances AntiOverSample(Instances init, double overTimes) throws Exception{
		 double ratio = overTimes;
		 FastVector attInfo = new FastVector();
			for (int i = 0; i < init.numAttributes(); i++) {
				weka.core.Attribute temp = init.attribute(i);
				attInfo.addElement(temp);
			}
			Instances YesInstances = new Instances("DefectSample1", attInfo,
					init.numInstances());// 閺夆晜鐟╅崳鐑芥儍閸曨偄鐏ュ┑顔碱儏椤旀劙鏌岃箛娑欎粯閻熸洑鐒﹂弫鐐哄箛韫囥儳绀夊☉鎾崇Х椤懐浜歌箛搴ｅ晩闁靛棴鎷�
			YesInstances.setClass(YesInstances.attribute(className));

			// YesInstances.setClassIndex(init.numAttributes() - 1);
			// 闁哄牜浜ｉ崗妯肩磼閻斿墎顏遍柣銊ュ閻ㄣ垻鐚剧紒妯煎灱缂佹稑褰炵紞鏃�绋夐悜妯讳粯闁告艾绨肩粩瀛樼▔椤忓嫮娼ｉ柟顑秶绀夐柛娆樺灥閸忔﹫鎷�?閼规澘姣�閻犱緤绱曢悾缁樼▔婵犲嫭鐣卞璺虹У濞煎懘鏁嶇仦鐐畳鐎垫澘鎳忛弫鍏兼交濞戞牭鎷�
			Instances Noinstances = new Instances("DefectSample2", attInfo,
					init.numInstances());
			Noinstances.setClass(Noinstances.attribute(className));
			init.setClass(init.attribute(className));
			int classIndex = init.classIndex();
			int numInstance = init.numInstances();
			int numYes = 0;
			int numNo = 0;
			for (int i = 0; i < numInstance; i++) {
				Instance temp = init.instance(i);
				double Value = temp.value(classIndex);
				if (Value == 1) { // weka闁汇劌瀚崬鎾焾閵娿儻鎷锋鐐存构缁楀绋夋惔锛勬剑闁诡儸鍛暠闁稿﹨鍋愬ù澶涙嫹?閻熸壆瀹夐柨娑樿嫰瀵剟鎳撻崓绺爇a api闁靛棴鎷�
					YesInstances.add(temp);
					numYes++;
				} else // clear change
				{
					Noinstances.add(temp);
					numNo++;
				}
			}
			// 濠碘�冲�归悘澶愬极娴兼潙娅ら柣鈺佹憸閻℃垿鏁嶇仦鐣屾澖闂傚嫬鎳嶇粭鍌炲及椤栨稓姊鹃柡鍫濐槹婢х晫鎮板畝鍐畺闂佹彃娲﹂悧閬嶆儍閸曗晪鎷�
			Instances res;
			if (numYes > numNo) {
				if ((double)(numNo/numYes) <= ratio) {
					return init;
				}
				res = excuteSample(Noinstances, YesInstances, 1/ratio);
			} else {
				if ((double)(numYes/numNo) <= ratio) {
					return init;
				}
				res = excuteSample(YesInstances, Noinstances, 1/ratio);
			}
			return res;
		    /*
			Instances res = ins;
			int valueCnt[] = {0,0};
			int attNum = ins.numAttributes();
			int insNum = ins.numInstances();
			int[] label = new int[insNum];
			ins.setClassIndex(attNum-1);
			int temp = 0;
			for(int i =0; i < insNum; i++){		
				temp = (int) ins.instance(i).classValue();
				valueCnt[temp]++;
				label[i] = temp;
			}
			int majorityCnt = (valueCnt[0] > valueCnt[1]) ? valueCnt[0] : valueCnt[1];
	    	int majorityLab = (valueCnt[0] > valueCnt[1]) ? 0 : 1;
	    	
	    	java.util.Random r=new java.util.Random();
	    	int[] posInd = new int[insNum-majorityCnt];
	    	int[] negInd = new int[majorityCnt];
	    	int t = 0, p = 0;
	    	
	    	for(int i =0; i < insNum; i++){		
				temp = (int) ins.instance(i).classValue();
				if(label[i] == majorityLab){
					negInd[t++] = i;
					
				}else{
					posInd[p++] = i;
				}	
			}
	    	//sampling in positive index
	    	int samNum = (int) (majorityCnt*(overTimes-1));
		    for(int i =0; i < samNum; i++){
		    	//System.out.println(posInd[r.nextInt(anomCnt)]);
		    	res.add(ins.instance(negInd[r.nextInt(majorityCnt)]));
		    }
		    String overPath = Config.select_folder + this.file.substring( 0,  file.lastIndexOf(".")) + "_oversample" + ".arff";
            System.out.println("===OverSample Path==="+overPath);
            System.out.println("===majority num("+majorityLab+"):"+majorityCnt+" ===");
            System.out.println("===new majority num("+majorityLab+"):"+samNum+" ===");
            ArffSaver saver = new ArffSaver(); 
    		saver.setInstances(res);  
    	    saver.setFile(new File(overPath));  
    	    saver.writeBatch(); 
	    	
	    	return res;*/
		 }
	 
	 /**
		 * 閺夆晛娲崳浼村冀闁垮鐓欐繛澶嬫磸閿燂拷?
		 * 
		 * @param init
		 * @return
		 * @throws IOException
		 */
		public static Instances OverSample(Instances init) throws IOException {
			FastVector attInfo = new FastVector();
			for (int i = 0; i < init.numAttributes(); i++) {
				weka.core.Attribute temp = init.attribute(i);
				attInfo.addElement(temp);
			}
			Instances YesInstances = new Instances("DefectSample1", attInfo,
					init.numInstances());// 閺夆晜鐟╅崳鐑芥儍閸曨偄鐏ュ┑顔碱儏椤旀劙鏌岃箛娑欎粯閻熸洑鐒﹂弫鐐哄箛韫囥儳绀夊☉鎾崇Х椤懐浜歌箛搴ｅ晩闁靛棴鎷�
			YesInstances.setClass(YesInstances.attribute(className));

			// YesInstances.setClassIndex(init.numAttributes() - 1);
			// 闁哄牜浜ｉ崗妯肩磼閻斿墎顏遍柣銊ュ閻ㄣ垻鐚剧紒妯煎灱缂佹稑褰炵紞鏃�绋夐悜妯讳粯闁告艾绨肩粩瀛樼▔椤忓嫮娼ｉ柟顑秶绀夐柛娆樺灥閸忔﹫鎷�?閼规澘姣�閻犱緤绱曢悾缁樼▔婵犲嫭鐣卞璺虹У濞煎懘鏁嶇仦鐐畳鐎垫澘鎳忛弫鍏兼交濞戞牭鎷�
			Instances Noinstances = new Instances("DefectSample2", attInfo,
					init.numInstances());
			Noinstances.setClass(Noinstances.attribute(className));
			init.setClass(init.attribute(className));
			int classIndex = init.classIndex();
			int numInstance = init.numInstances();
			int numYes = 0;
			int numNo = 0;
			for (int i = 0; i < numInstance; i++) {
				Instance temp = init.instance(i);
				double Value = temp.value(classIndex);
				if (Value == 1) { // weka闁汇劌瀚崬鎾焾閵娿儻鎷锋鐐存构缁楀绋夋惔锛勬剑闁诡儸鍛暠闁稿﹨鍋愬ù澶涙嫹?閻熸壆瀹夐柨娑樿嫰瀵剟鎳撻崓绺爇a api闁靛棴鎷�
					YesInstances.add(temp);
					numYes++;
				} else // clear change
				{
					Noinstances.add(temp);
					numNo++;
				}
			}
			// 濠碘�冲�归悘澶愬极娴兼潙娅ら柣鈺佹憸閻℃垿鏁嶇仦鐣屾澖闂傚嫬鎳嶇粭鍌炲及椤栨稓姊鹃柡鍫濐槹婢х晫鎮板畝鍐畺闂佹彃娲﹂悧閬嶆儍閸曗晪鎷�
			if (numYes == numNo) {
				return init;
			}
			Instances res;
			if (numYes > numNo) {
				res = excuteSample(YesInstances, Noinstances, 1);
			} else {
				res = excuteSample(Noinstances, YesInstances, 1);
			}
			return res;
		}

		/**
		 * 闁圭顦遍崣搴ｇ磼濞嗗繒鏆伴柣銊ュ閻︻喗绗熺�ｎ厾绠婚悶娑樼焷缁诲啴骞庨懞銉у闁靛棴鎷�
		 * 
		 * @param instances1
		 *            濞戞捁顕ч悿鍕瑹鐎ｎ喗鑲犻柨娑樿嫰瀹撳棙绗熷┑鍥хウ闁汇劌瀚悿鍕瑹鐎ｎ喗鑲犻柨娑樺缁″啰浜告潏銊π﹂柛蹇嬪姂閸庡瓨鎷呯捄銊︽殢闁汇劌瀚悿鍕瑹鐎ｎ喗鑲犻柕鍡嫹
		 * @param instances2
		 *            闁告搩鍨伴悿鍕瑹鐎ｎ喗鑲犻柨娑樺缁″啰浜告潏銊π﹂柣顏嗗枑椤掓粣鎷�?閻愰鏀介梺鎻掓处閻楅亶鎯冮崟顐ゆ澖濞撴艾顑夊▔锕傚Υ閿燂拷?		 * @param i
		 *            闁规儼濮ら悧閬嶅触鎼达紕绻侀柛鎺撳濞堟垶绋夊鍛�遍柣銊ュ鐞氼偊寮介崶鈺婂姰闁汇劌瀚惁顔界瑹鐎ｅ墎绀夐柛妤勬珪婵炲﹪寮藉畡鐗堝�祅um(yesInstances)/num(noinstances)闁汇劌瀚惁顔界瑹鐎ｅ墎绀夋繛澶堝妽閸撲即鏁嶉敓锟�?		 *            闁汇垹褰夌花顒佺▔鏉炴壆鍟� 闁告梻濞�閿熺晫绮欑�ｎ亞纰嶉弶鈺傚姌椤㈡垿鏌呴悢宄邦唺闁挎稑鏈〒鍫曞触鎼达紕鏉藉Δ鐘茬灱缁劑寮稿鍕枙闁哄秹鏀卞鍌滄媼閸撗呮瀭濞戞搫鎷烽柕鍡嫹
		 */
		private static Instances excuteSample(Instances instances1,
				Instances instances2, double ratio) {
			int numSample = (int) Math.ceil(instances1.numInstances() * ratio); // 濞村吋鐭粭澶嬪濮樿鲸鏆犲ù婊冮閻ゅ嫭绗熺�ｎ偅娈堕弶鈺佹搐椤у鎳撶仦鐣屸敍婵犙冨枦閿燂拷?
			int numNo = instances2.numInstances();
			Random rn = new Random();
			for (int i = 0; i < numSample; i++) {
				instances1.add(instances2.instance(rn.nextInt(numNo)));
			}
			return instances1;
		}

		/**
		 * 婵炲棛濞�閸ｄ即寮介柨瀣厵婵炲鎷�
		 * @param init 闁烩偓鍔嬬花顒勬煂閸ャ劎澹夐柣銊ュ閻ゅ嫭绗熺�ｎ喗鑲�.
		 * @return
		 * @throws IOException
		 */
		public static Instances UnderSample(Instances init) throws IOException {
			int numAttr = init.numAttributes();
			int numInstance = init.numInstances();

			FastVector attInfo = new FastVector();
			for (int i = 0; i < numAttr; i++) {
				weka.core.Attribute temp = init.attribute(i);
				attInfo.addElement(temp);
			}

			Instances NoInstances = new Instances("No", attInfo, numInstance);

			NoInstances.setClass(NoInstances.attribute(className));

			Instances YesInstances = new Instances("yes", attInfo, numInstance);
			YesInstances.setClass(YesInstances.attribute(className));

			init.setClass(init.attribute(className));
			int classIndex = init.classIndex();
			
			int numYes = 0;
			int numNo = 0;
			
			for (int i = 0; i < numInstance; i++) {
				Instance temp = init.instance(i);
				double Value = temp.value(classIndex);
				if (Value == 0) { // yes
					NoInstances.add(temp);
					numNo++;
				} else {
					YesInstances.add(temp);
					numYes++;
				}
			}
			if (numYes == numNo) {
				return init;
			}
			Instances res;
			if (numYes > numNo) {
				res = excuteSample(NoInstances, YesInstances, 1);
			} else {
				res = excuteSample(YesInstances, NoInstances, 1);
			}
			return res;
		}

		
		public static Instances SmoteSample(Instances ins, double ratio) throws Exception
		{
			int rat = 0;
			int classIndex = ins.classIndex();
			int numInstance = ins.numInstances();
			int numYes = 0;
			int numNo = 0;
			
			for (int i = 0; i < numInstance; i++) {
				Instance temp = ins.instance(i);
				double Value = temp.value(classIndex);
				if (Value == 0) { // yes
					numNo++;
				} else {
					numYes++;
				}
			}
			if (numYes == numNo) {
				rat = 100;
			}
			if (numYes > numNo) {
				rat = numYes/numNo;
			} else {
				rat = numNo/numYes;
			}
			
			weka.filters.supervised.instance.SMOTE smote = new  SMOTE();
			//smote.setPercentage(ratio*100);
			//smote.setPercentage(rat*100);
			//System.out.println("somte percentage : " + smote.getPercentage());
			ins.setClassIndex(ins.numAttributes()-1);
			smote.setInputFormat(ins);
			Instances res = Filter.useFilter(ins, smote);
			return res;
			
		}
		
		public Instances RandomSample(Instances init, double ratio) {
			int numAttr = init.numAttributes();
			int numInstance = init.numInstances();
			int totalNum = (int) (numInstance * ratio);
			
			FastVector attInfo = new FastVector();
			for (int i = 0; i < numAttr; i++) {
				weka.core.Attribute temp = init.attribute(i);
				attInfo.addElement(temp);
			}
			Instances res = new Instances("Res", attInfo, totalNum);
			Random rn = new Random();
			for (int i = 0; i <totalNum; i++) {
					res.add(init.instance(rn.nextInt(numInstance)));
			}
			res.setClass(res.attribute(className));
			return res;
		}
	
}